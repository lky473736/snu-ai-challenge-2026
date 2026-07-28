"""
HNTV v18 Training — DDP via Accelerate
Pairwise Bradley-Terry Temporal Ranking: 4프레임을 항상 원본 순서로 보여주고, 6개 쌍(4C2)에 대해
"Frame i가 Frame j보다 먼저냐"를 Yes/No로만 묻는다. vision encoder는 샘플당 1번만 태우고
(src/model.py의 pairwise_logits_fast) LLM 레이어만 6줄 배치로 통과시켜 로그오즈를 얻는다.
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import torch
import pandas as pd
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
    LOGGING_STEPS, VAL_RATIO, SEED, LORA_R, LORA_ALPHA, CKPT_NAME,
)
from src.dataset import PairwiseTemporalDataset, collate_fn, load_image, PAIRS
from src.model import load_model_and_processor, get_yes_no_token_ids, pairwise_logits_fast
from src.loss import pairwise_bt_loss
from src.aggregate import score_pairs_fast, aggregate_ranks


def val_exact_match(model, processor, val_raw_df, device, n_samples=None):
    import ast
    model.eval()
    df = val_raw_df if n_samples is None else val_raw_df.sample(min(n_samples, len(val_raw_df)), random_state=SEED)
    correct = 0
    raw_correct, raw_total = 0, 0
    adj_correct, adj_total = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sid, sentence = row["Id"], row["Sentence"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        pair_scores = score_pairs_fast(model, processor, base_imgs, sentence, device)
        for (i, j), z in pair_scores.items():
            pred_before = z > 0
            gt_before = gt[i - 1] < gt[j - 1]
            is_adj = abs(gt[i - 1] - gt[j - 1]) == 1
            raw_correct += int(pred_before == gt_before)
            raw_total += 1
            if is_adj:
                adj_correct += int(pred_before == gt_before)
                adj_total += 1

        pred = aggregate_ranks(pair_scores)
        if pred == gt:
            correct += 1

    acc = correct / len(df)
    raw_acc = raw_correct / raw_total
    adj_acc = adj_correct / adj_total if adj_total else float("nan")
    print(f"\n[Val] exact_match={acc:.4f} ({correct}/{len(df)})  "
          f"raw_pairwise_acc={raw_acc:.4f}  raw_adjacent(d=1)_acc={adj_acc:.4f}")
    if adj_total and adj_acc < raw_acc - 0.05:
        print(f"  [주의] d=1 인접쌍 정확도가 전체 대비 {raw_acc - adj_acc:.4f} 낮음 — "
              f"과거 loss reweighting들이 못 풀었던 것과 같은 패턴일 수 있음")
    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--ckpt_name", type=str, default=CKPT_NAME)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="job이 중간에 끊겼을 때 대비: 저장된 LoRA 체크포인트 경로에서 가중치만 이어받아 재출발.")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--prev_best_val", type=float, default=0.0)
    args = parser.parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v18] Processes={accelerator.num_processes}  Device={device}")
        print(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  lr={args.lr}")
        print(f"  Loss: pairwise_bt_loss (Bradley-Terry BCE, 6쌍/샘플, d=1 가중치 적용)")
        print(f"  Vision encoder: 샘플당 1번만 인코딩 (pairwise_logits_fast)")
        if args.resume_from:
            print(f"  [resume] {args.resume_from} 에서 재출발, start_epoch={args.start_epoch}")

    torch.manual_seed(SEED)

    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val = int(len(train_csv) * VAL_RATIO)
        val_raw = train_csv[:n_val]
        trn_raw = train_csv[n_val:]
        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)
        print(f"Train samples (raw): {len(trn_raw)}  Val: {len(val_raw)}")

    accelerator.wait_for_everyone()

    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val = int(len(train_csv) * VAL_RATIO)
    trn_raw = train_csv[n_val:]
    val_raw = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    train_ds = PairwiseTemporalDataset(trn_raw)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2,
    )

    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                                  resume_from=args.resume_from)

    remaining_epochs = EPOCHS - args.start_epoch + 1
    n_steps = (len(train_dl) // GRAD_ACCUM) * remaining_epochs
    n_warmup = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)

    best_val_acc = args.prev_best_val
    global_step = 0
    step_times = []

    for epoch in range(args.start_epoch, EPOCHS + 1):
        t0 = time.time()
        steps_per_epoch = len(trn_raw)
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}  (예상 step 수: {steps_per_epoch})")
            print(f"  Effective batch = {BATCH_SIZE}×{GRAD_ACCUM}×{accelerator.num_processes}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        t_step0 = time.time()

        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)
                logit_parts, labels, adj_flags = [], [], []
                for sid, grp_imgs, grp_sents, grp_labels, grp_adj in zip(
                        batch["sids"], batch["images"], batch["sentences"], batch["labels"], batch["adj_flags"]):
                    base_imgs, sentence = grp_imgs[0], grp_sents[0]
                    z = pairwise_logits_fast(unwrapped, processor, base_imgs, sentence, device)
                    logit_parts.append(z)
                    labels.extend(grp_labels)
                    adj_flags.extend(grp_adj)
                logits = torch.cat(logit_parts)

                loss = pairwise_bt_loss(logits, labels, adj_flags)
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
                epoch_loss += loss.item()

                step_dt = time.time() - t_step0
                step_times.append(step_dt)
                if len(step_times) > 20:
                    step_times.pop(0)
                if is_main and global_step % LOGGING_STEPS == 0:
                    avg_step = sum(step_times) / len(step_times)
                    remaining = steps_per_epoch - (step + 1)
                    eta_min = avg_step * remaining / 60
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / (step + 1)
                    print(f"  step={global_step:5d} ({step+1}/{steps_per_epoch})  loss={avg_loss:.4f}  "
                          f"lr={lr_now:.2e}  avg_step={avg_step:.2f}s  ETA(에폭 잔여)={eta_min:.1f}분")
                t_step0 = time.time()

        accelerator.wait_for_everyone()

        if is_main:
            elapsed = (time.time() - t0) / 60
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]  소요: {elapsed:.1f}분  "
                  f"avg_loss: {epoch_loss / len(train_dl):.4f}")

            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, device)

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
