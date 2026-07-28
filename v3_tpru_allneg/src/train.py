"""
HNTV Training - DDP via Accelerate
accelerate launch --num_processes=4 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import ast
import argparse
import torch
import pandas as pd
from itertools import permutations
from pathlib import Path
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
    MARGIN, RANKING_WEIGHT, LORA_R, LORA_ALPHA, MODEL_PATH,
    TRAIN_MINIBATCH, INFER_BATCH_SIZE,
)
from src.hard_negative import generate_hard_negative_samples
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn, load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import VerificationRankingLoss

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def val_exact_match(model, processor, val_raw_df, yes_id, no_id, device, n_samples=100):
    """rank 0에서만 실행. 24순열 스코어링으로 exact match 측정."""
    model.eval()
    df = val_raw_df.sample(min(n_samples, len(val_raw_df)), random_state=SEED)

    correct, wrong_cases = 0, []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        gt        = ast.literal_eval(row["Answer"])
        img_dir   = DATA_DIR / "train" / sample_id
        files     = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])

        def order_to_images(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [load_image(str(img_dir / files[inv[t]])) for t in range(4)]

        # 배치 validation: INFER_BATCH_SIZE개씩 묶어서 처리
        base_imgs = order_to_images(list(range(1, 5)))  # 원본 순서로 로드

        def order_to_images_fast(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        scores = []
        for batch_start in range(0, len(ALL_PERMS), INFER_BATCH_SIZE):
            batch_perms = ALL_PERMS[batch_start: batch_start + INFER_BATCH_SIZE]
            batch_texts, batch_imgs = [], []
            for perm in batch_perms:
                imgs = order_to_images_fast(list(perm))
                msgs = build_messages(imgs, sentence)
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                batch_texts.append(text)
                batch_imgs.append(imgs)
            with torch.no_grad():
                inp = processor(text=batch_texts, images=batch_imgs,
                                return_tensors="pt", padding=True).to(device)
                s_batch = forward_logit(model, processor, inp, yes_id, no_id)
            for i, perm in enumerate(batch_perms):
                scores.append((s_batch[i].item(), list(perm)))

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0]-diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    acc = correct / len(df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[Val] exact_match={acc:.4f} ({correct}/{len(df)})  "
          f"adj_swap_fail={adj}  other_fail={len(wrong_cases)-adj}")
    for w in wrong_cases[:3]:
        print(f"  {w['id']}  gt={w['gt']}  pred={w['pred']}  [{w['type']}]")

    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin",         type=float, default=MARGIN)
    parser.add_argument("--ranking_weight", type=float, default=RANKING_WEIGHT)
    parser.add_argument("--lora_r",         type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha",     type=int,   default=LORA_ALPHA)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--ckpt_name",      type=str,   default="best")
    args = parser.parse_args()

    pg_kwargs   = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Processes: {accelerator.num_processes}  Device: {device}")
        print(f"  margin={args.margin}  ranking_weight={args.ranking_weight}  "
              f"lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  "
              f"lr={args.lr}  ckpt={args.ckpt_name}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
                      f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")

    torch.manual_seed(SEED)

    # ── 데이터 ─────────────────────────────────────────────
    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val   = int(len(train_csv) * VAL_RATIO)
        val_raw = train_csv[:n_val]
        trn_raw = train_csv[n_val:]
        hn_df   = generate_hard_negative_samples(trn_raw)
        # 다른 프로세스도 쓸 수 있게 임시 저장
        hn_df.to_csv(CKPT_DIR / "_hn_df.csv", index=False)
        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)

    accelerator.wait_for_everyone()

    hn_df   = pd.read_csv(CKPT_DIR / "_hn_df.csv")
    val_raw = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    train_ds = GroupedTemporalDataset(hn_df, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    if is_main:
        print(f"Train groups: {len(train_ds)}  Val raw: {len(val_raw)}")

    # ── 모델 ────────────────────────────────────────────────
    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    # args.lr 반영 (config.py LR override)
    _lr = args.lr
    yes_id, no_id    = get_yes_no_token_ids(processor)
    criterion        = VerificationRankingLoss(margin=args.margin, ranking_weight=args.ranking_weight)

    n_steps  = (len(train_dl) // GRAD_ACCUM) * EPOCHS
    n_warmup = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=_lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    # ── 학습 루프 ────────────────────────────────────────────
    best_val_acc = 0.0
    global_step  = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}")
            print(f"  Processes={accelerator.num_processes}  "
                  f"Effective batch={BATCH_SIZE * GRAD_ACCUM * accelerator.num_processes}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(
            tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)
        ):
            with accelerator.accumulate(model):
                images_batch  = batch["images"]
                sentences_batch = batch["sentences"]
                labels        = batch["labels"].to(device)
                group_sizes   = batch["group_sizes"]

                # 모든 샘플 텍스트/이미지 준비
                all_texts, all_images_list = [], []
                for imgs, sent in zip(images_batch, sentences_batch):
                    msgs = build_messages(imgs, sent)
                    text = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    all_texts.append(text)
                    all_images_list.append(imgs)

                # TRAIN_MINIBATCH씩 묶어서 진짜 배치 forward
                logits_list = []
                unwrapped = accelerator.unwrap_model(model)
                for bi in range(0, len(all_texts), TRAIN_MINIBATCH):
                    sub_texts  = all_texts[bi: bi + TRAIN_MINIBATCH]
                    sub_images = all_images_list[bi: bi + TRAIN_MINIBATCH]
                    inp = processor(
                        text=sub_texts, images=sub_images,
                        return_tensors="pt", padding=True
                    ).to(device)
                    logits_list.append(
                        forward_logit(unwrapped, processor, inp, yes_id, no_id)
                    )

                logits = torch.cat(logits_list)
                loss, bce_l, rank_l = criterion(logits, labels, group_sizes)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item()

                if is_main and global_step % LOGGING_STEPS == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / (step + 1)
                    print(f"  step={global_step:5d}  loss={avg_loss:.4f}  "
                          f"bce={bce_l.item():.4f}  rank={rank_l.item():.4f}  "
                          f"lr={lr_now:.2e}")

        accelerator.wait_for_everyone()

        if is_main:
            elapsed = (time.time() - t0) / 60
            alloc   = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]")
            print(f"  소요: {elapsed:.1f}분  avg_loss: {epoch_loss/len(train_dl):.4f}")
            print(f"  GPU0 max_alloc: {alloc:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

            # ── Validation (rank 0만) ────────────────────
            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, yes_id, no_id, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = CKPT_DIR / args.ckpt_name
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"  ★ Best model saved  (val_acc={val_acc:.4f})")

        accelerator.wait_for_everyone()

    if is_main:
        print(f"\n학습 완료. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
