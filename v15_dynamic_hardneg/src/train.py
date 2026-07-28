"""
HNTV v15 Training — DDP via Accelerate
v15: v14(ListwiseSoftmaxLoss) 유지 + hard negative K 확장(N_EXTRA) + 동적 하드 네거티브 뱅크.
매 epoch 끝에 "이번에 모델이 제일 헷갈려한(positive 대비 점수차가 가장 작은) negative"를
sample_id별로 기록해뒀다가, 다음 epoch부터 그 negative를 그룹에 다시 포함시킨다 — 추가 forward
없이 이미 하는 학습 forward의 logit을 재활용(무료에 가까움). DDP 4-GPU라 rank별로 다른 샘플을
보게 되므로, all_gather_object로 전 rank의 관찰을 모아 병합해 동일한 전역 뱅크를 유지한다.
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import csv
import time
import argparse
import torch
import torch.distributed as dist
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
    TRAIN_MINIBATCH, INFER_BATCH_SIZE, N_EXTRA,
)
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn, load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import ListwiseSoftmaxLoss
from src.hard_negative import kendall_dist

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

        all_texts, all_imgs = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_imgs.append(imgs)
        with torch.no_grad():
            s = _forward_chunk(model, processor, all_texts, all_imgs, yes_id, no_id, device, INFER_BATCH_SIZE)
        scores = [(s[i].item(), list(perm)) for i, perm in enumerate(ALL_PERMS)]

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


def _forward_chunk(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, chunk_size):
    while True:
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(
                    text=texts[bi: bi + chunk_size],
                    images=imgs_list[bi: bi + chunk_size],
                    return_tensors="pt", padding=True,
                ).to(device)
                parts.append(forward_logit(model_unwrapped, processor, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] forward chunk_size -> {chunk_size}", flush=True)


def forward_batch(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device):
    return _forward_chunk(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, TRAIN_MINIBATCH)


def sync_bank(local_updates: dict, world_size: int) -> dict:
    """rank별 로컬 관찰(local_updates: {sample_id: (perm, score)})을 all_gather_object로 모아
    전역 뱅크로 병합. 여러 rank가 같은 sample_id를 봤으면 score가 더 높은(더 헷갈렸던)쪽을 채택.
    모든 rank가 동일한 입력 리스트를 병합하므로 결과도 전 rank에서 동일(별도 broadcast 불필요)."""
    if world_size <= 1:
        return local_updates
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_updates)
    merged = {}
    for d in gathered:
        for sid, (perm, score) in d.items():
            if sid not in merged or score > merged[sid][1]:
                merged[sid] = (perm, score)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r",     type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha", type=int,   default=LORA_ALPHA)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--n_extra",    type=int,   default=N_EXTRA)
    parser.add_argument("--train_minibatch", type=int, default=TRAIN_MINIBATCH)
    parser.add_argument("--ckpt_name",  type=str,   default="best_v15")
    parser.add_argument("--resume_from", type=str,  default=None)
    parser.add_argument("--start_epoch", type=int,  default=1)
    parser.add_argument("--prev_best_val", type=float, default=0.0)
    args = parser.parse_args()

    minibatch = args.train_minibatch

    pg_kwargs   = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM,
                               kwargs_handlers=[pg_kwargs])
    device  = accelerator.device
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v15] Processes={world_size}  Device={device}")
        print(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  lr={args.lr}")
        print(f"  Loss: ListwiseSoftmaxLoss (v14와 동일)")
        print(f"  Hard negatives: 기본 7(d1..6 전구간) + N_EXTRA={args.n_extra}(동적 뱅크+추가 랜덤)")
        print(f"  TRAIN_MINIBATCH={minibatch}")
        if args.resume_from:
            print(f"  [resume] {args.resume_from} 에서 재출발, start_epoch={args.start_epoch}")

    torch.manual_seed(SEED)

    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val     = int(len(train_csv) * VAL_RATIO)
        val_raw   = train_csv[:n_val]
        trn_raw   = train_csv[n_val:]
        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)
        print(f"Train samples (raw, pre-expansion): {len(trn_raw)}")

    accelerator.wait_for_everyone()

    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val     = int(len(train_csv) * VAL_RATIO)
    trn_raw   = train_csv[n_val:]
    val_raw   = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    def _worker_init_fn(worker_id):
        import random as _random
        seed = (SEED + accelerator.process_index * 1000 + worker_id
                + int(time.time() * 1000) % 100000)
        _random.seed(seed)

    train_ds = GroupedTemporalDataset(trn_raw, n_extra=args.n_extra)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2,
        worker_init_fn=_worker_init_fn,
    )

    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                                  resume_from=args.resume_from)
    yes_id, no_id    = get_yes_no_token_ids(processor)
    criterion        = ListwiseSoftmaxLoss()

    remaining_epochs = EPOCHS - args.start_epoch + 1
    n_steps   = (len(train_dl) // GRAD_ACCUM) * remaining_epochs
    n_warmup  = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    best_val_acc = args.prev_best_val
    global_step  = 0

    weight_log_path = LOG_DIR / "dist_weights.csv"
    if is_main and not (args.resume_from and weight_log_path.exists()):
        with open(weight_log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "step", "w1","w2","w3","w4","w5","w6",
                "ema1","ema2","ema3","ema4","ema5","ema6",
                "L1","L2","L3","L4","L5","L6",
            ])

    bank_log_path = LOG_DIR / "bank_stats.csv"
    if is_main and not (args.resume_from and bank_log_path.exists()):
        with open(bank_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "bank_size", "mean_hard_score"])

    for epoch in range(args.start_epoch, EPOCHS + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}  "
                  f"(bank_size={len(train_ds.bank)})")
            print(f"  Effective batch = {BATCH_SIZE}×{GRAD_ACCUM}×{world_size}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        local_bank_updates = {}  # 이번 epoch, 이 rank가 관찰한 sample_id -> (perm, score)

        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                texts, imgs_list = [], []
                group_offsets   = []  # (start, size, grp_dists, grp_perms, sample_id)

                for grp_imgs, grp_sents, grp_dists, grp_perms, sid in zip(
                    batch["images"], batch["sentences"], batch["dists"],
                    batch["perms"], batch["sample_ids"],
                ):
                    start = len(texts)
                    for imgs, sent in zip(grp_imgs, grp_sents):
                        msg  = build_messages(imgs, sent)
                        text = processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                        texts.append(text)
                        imgs_list.append(imgs)
                    group_offsets.append((start, len(grp_imgs), grp_dists, grp_perms, sid))

                logits = forward_batch(unwrapped, processor, texts, imgs_list, yes_id, no_id, device)

                ord_logits_parts, ord_dists, ord_sizes = [], [], []

                for start, size, grp_dists, grp_perms, sid in group_offsets:
                    g_logits = logits[start: start + size]
                    ord_logits_parts.append(g_logits)
                    ord_dists.extend(grp_dists)
                    ord_sizes.append(size)

                    # 하드 네거티브 관찰 기록: 이 그룹에서 negative 중 점수가 가장 높았던(=제일
                    # 헷갈렸던) permutation을 sample_id별로 저장. 추가 forward 없음 — 이미 계산된
                    # g_logits를 그대로 재사용.
                    neg_idxs = [i for i, d in enumerate(grp_dists) if d != 0]
                    if neg_idxs:
                        scores = g_logits.detach()
                        best_i = max(neg_idxs, key=lambda i: scores[i].item())
                        local_bank_updates[sid] = (grp_perms[best_i], scores[best_i].item())

                loss, per_dist_loss, weights = criterion(
                    torch.cat(ord_logits_parts), ord_dists, ord_sizes
                )

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
                        f"  step={global_step:5d}  loss={avg:.4f}  lr={lr_now:.2e}"
                        f"  avg_neg_prob=[{','.join(f'{x:.3f}' if not (x!=x) else 'nan' for x in L)}]"
                    )
                    with open(weight_log_path, "a", newline="") as f:
                        csv.writer(f).writerow([global_step] + w + e + L)

        accelerator.wait_for_everyone()

        # ── epoch 끝: rank별 로컬 관찰을 모아 전역 뱅크 갱신 ──
        global_bank = sync_bank(local_bank_updates, world_size)
        train_ds.bank = global_bank  # 다음 epoch의 fork된 worker가 이 상태를 물려받음

        if is_main:
            elapsed = (time.time() - t0) / 60
            mean_score = (sum(v[1] for v in global_bank.values()) / len(global_bank)
                          if global_bank else float("nan"))
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]  소요: {elapsed:.1f}분  "
                  f"avg_loss: {epoch_loss / len(train_dl):.4f}  "
                  f"bank_size={len(global_bank)}  mean_hard_score={mean_score:.3f}")
            with open(bank_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, len(global_bank), mean_score])

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
