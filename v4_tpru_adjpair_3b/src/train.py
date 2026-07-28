"""
HNTV v4 Training (3B port) — Accelerate, works with 1 or N GPUs
v4 design: distribution-informed d=1..6 coverage + AdaptiveDistanceLoss + no_ordering aug
Single GPU (Colab):  python src/train.py --ckpt_name best_v4_3b
Multi GPU:           accelerate launch --num_processes=N --mixed_precision=bf16 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import csv
import time
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from itertools import permutations
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm

from config import (
    DATA_DIR, CKPT_DIR, LOG_DIR,
    EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, WARMUP_RATIO,
    LOGGING_STEPS, VAL_RATIO, SEED,
    LORA_R, LORA_ALPHA, MODEL_PATH,
    TRAIN_MINIBATCH, INFER_BATCH_SIZE,
    EMA_ALPHA, WEIGHT_TEMP, NO_ORDERING_WEIGHT,
)
from src.hard_negative import generate_hard_negative_samples
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn, load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import AdaptiveDistanceLoss

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def val_exact_match(model, processor, val_raw_df, yes_id, no_id, device, n_samples=None):
    model.eval()
    df = val_raw_df if n_samples is None else val_raw_df.sample(min(n_samples, len(val_raw_df)), random_state=SEED)
    correct, wrong_cases = 0, []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        gt        = ast.literal_eval(row["Answer"])
        img_dir   = DATA_DIR / "train" / sample_id
        files     = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / files[i])) for i in range(4)]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        scores = []
        for bs in range(0, len(ALL_PERMS), INFER_BATCH_SIZE):
            batch_perms = ALL_PERMS[bs: bs + INFER_BATCH_SIZE]
            texts, imgs_list = [], []
            for perm in batch_perms:
                imgs = reorder(list(perm))
                msgs = build_messages(imgs, sentence)
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                imgs_list.append(imgs)
            with torch.no_grad():
                inp = processor(text=texts, images=imgs_list,
                                return_tensors="pt", padding=True).to(device)
                s = forward_logit(model, processor, inp, yes_id, no_id)
            for i, perm in enumerate(batch_perms):
                scores.append((s[i].item(), list(perm)))

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    acc = correct / len(df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[Val] exact_match={acc:.4f} ({correct}/{len(df)})  "
          f"adj_swap_fail={adj}  other_fail={len(wrong_cases) - adj}")
    model.train()
    return acc


def forward_batch(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device):
    logits_list = []
    for bi in range(0, len(texts), TRAIN_MINIBATCH):
        inp = processor(
            text=texts[bi: bi + TRAIN_MINIBATCH],
            images=imgs_list[bi: bi + TRAIN_MINIBATCH],
            return_tensors="pt", padding=True,
        ).to(device)
        logits_list.append(forward_logit(model_unwrapped, processor, inp, yes_id, no_id))
    return torch.cat(logits_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r",     type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha", type=int,   default=LORA_ALPHA)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--ckpt_name",  type=str,   default="best_v4_3b")
    args = parser.parse_args()

    pg_kwargs   = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM,
                               kwargs_handlers=[pg_kwargs])
    device  = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v4-3B] Processes={accelerator.num_processes}  Device={device}")
        print(f"  MODEL_PATH={MODEL_PATH}  DATA_DIR={DATA_DIR}")
        print(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  lr={args.lr}  epochs={args.epochs}")
        print(f"  Loss: AdaptiveDistanceLoss (d=1..6 stratified coverage)")

    torch.manual_seed(SEED)

    # ── 데이터 ─────────────────────────────────────────────────────
    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val     = int(len(train_csv) * VAL_RATIO)
        val_raw   = train_csv[:n_val]
        trn_raw   = train_csv[n_val:]

        df_train = generate_hard_negative_samples(trn_raw)
        df_train.to_csv(CKPT_DIR / "_hn_v4.csv", index=False)
        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)

        n_groups = df_train["group_id"].nunique()
        print(f"Train groups: {n_groups}  Total rows: {len(df_train)}")

    accelerator.wait_for_everyone()

    df_train = pd.read_csv(CKPT_DIR / "_hn_v4.csv")
    val_raw  = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    train_ds = GroupedTemporalDataset(df_train)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2,
    )

    # ── 모델 ────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    yes_id, no_id    = get_yes_no_token_ids(processor)
    criterion        = AdaptiveDistanceLoss(ema_alpha=EMA_ALPHA, temperature=WEIGHT_TEMP)

    n_steps   = (len(train_dl) // GRAD_ACCUM) * args.epochs
    n_warmup  = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    best_val_acc = 0.0
    global_step  = 0

    weight_log_path = LOG_DIR / "dist_weights.csv"
    if is_main:
        with open(weight_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "w1","w2","w3","w4","w5","w6",
                "ema1","ema2","ema3","ema4","ema5","ema6",
                "L1","L2","L3","L4","L5","L6",
            ])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}  시작: {time.strftime('%H:%M:%S')}")
            print(f"  Effective batch = {BATCH_SIZE}×{GRAD_ACCUM}×{accelerator.num_processes}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                texts, imgs_list = [], []
                group_offsets   = []  # (start, size, is_no_ord, grp_dists)

                for grp_imgs, grp_sents, grp_dists, is_no_ord in zip(
                    batch["images"], batch["sentences"],
                    batch["dists"], batch["no_orderings"]
                ):
                    start = len(texts)
                    for imgs, sent in zip(grp_imgs, grp_sents):
                        msg  = build_messages(imgs, sent)
                        text = processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                        texts.append(text)
                        imgs_list.append(imgs)
                    group_offsets.append((start, len(grp_imgs), is_no_ord, grp_dists))

                logits = forward_batch(unwrapped, processor, texts, imgs_list, yes_id, no_id, device)

                ord_logits_parts, ord_dists, ord_sizes = [], [], []
                no_ord_pos_scores = []

                for start, size, is_no_ord, grp_dists in group_offsets:
                    g_logits = logits[start: start + size]
                    if is_no_ord:
                        no_ord_pos_scores.append(g_logits[0])
                    else:
                        ord_logits_parts.append(g_logits)
                        ord_dists.extend(grp_dists)
                        ord_sizes.append(size)

                if ord_sizes:
                    loss_ord, per_dist_loss, weights = criterion(
                        torch.cat(ord_logits_parts), ord_dists, ord_sizes
                    )
                else:
                    loss_ord      = torch.tensor(0.0, device=device)
                    per_dist_loss = {}
                    weights       = criterion.get_weights().cpu()

                if no_ord_pos_scores:
                    loss_no_ord = -F.logsigmoid(
                        torch.stack(no_ord_pos_scores)
                    ).mean()
                else:
                    loss_no_ord = torch.tensor(0.0, device=device)

                loss = loss_ord + NO_ORDERING_WEIGHT * loss_no_ord

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss  += loss.item()

                if is_main and global_step % LOGGING_STEPS == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg    = epoch_loss / (step + 1)
                    w = weights.tolist()
                    e = criterion.ema.tolist()
                    L = [per_dist_loss.get(d, float("nan")) for d in range(1, 7)]
                    print(
                        f"  step={global_step:5d}  loss={avg:.4f}  "
                        f"no_ord={loss_no_ord.item():.4f}  lr={lr_now:.2e}"
                        f"  w=[{','.join(f'{x:.3f}' for x in w)}]"
                        f"  L=[{','.join(f'{x:.3f}' if not (x!=x) else 'nan' for x in L)}]"
                    )
                    with open(weight_log_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [global_step] + w + e + L
                        )

        accelerator.wait_for_everyone()

        if is_main:
            elapsed = (time.time() - t0) / 60
            print(f"\n[Epoch {epoch}/{args.epochs} 완료]  소요: {elapsed:.1f}분  "
                  f"avg_loss: {epoch_loss / len(train_dl):.4f}")

            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, yes_id, no_id, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = CKPT_DIR / args.ckpt_name
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"  ★ Best saved (val_acc={val_acc:.4f})")

            last_path = CKPT_DIR / (args.ckpt_name + "_last")
            unwrapped.save_pretrained(last_path)
            processor.save_pretrained(last_path)
            print(f"  → Last saved to {last_path.name}")

        accelerator.wait_for_everyone()

    if is_main:
        print(f"\n학습 완료. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
