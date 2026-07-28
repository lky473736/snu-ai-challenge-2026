"""
HNTV v21 Training — Qwen3-VL-32B-Instruct QLoRA(4bit), DDP via Accelerate
run_rank_sweep.sh가 VRAM 한계까지 찾은 rank(--lora_r)로 실제 3epoch 학습.
그 외 레시피(hard negative K=7, ListwiseSoftmaxLoss, LR, GRAD_ACCUM)는 v14/v20과 100% 동일.
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py --lora_r <r>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import time
import argparse
import torch
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
    LOGGING_STEPS, VAL_RATIO, SEED, MODEL_PATH,
    TRAIN_MINIBATCH, INFER_BATCH_SIZE, LORA_ALPHA_RATIO,
)
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn, load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import ListwiseSoftmaxLoss

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def val_exact_match(model, processor, val_raw_df, yes_id, no_id, device, n_samples=None):
    model.eval()
    df = val_raw_df if n_samples is None else val_raw_df.sample(min(n_samples, len(val_raw_df)), random_state=SEED)
    correct, wrong_cases = 0, []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sample_id = row["Id"]
        sentence = row["Sentence"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sample_id
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
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
            s = _chunked_forward(model, processor, all_texts, all_imgs, yes_id, no_id, device, _current_infer_batch)
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


_current_minibatch = [TRAIN_MINIBATCH]
_current_infer_batch = [INFER_BATCH_SIZE]


def _chunked_forward(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, size_holder):
    while True:
        chunk_size = size_holder[0]
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(
                    text=texts[bi: bi + chunk_size], images=imgs_list[bi: bi + chunk_size],
                    return_tensors="pt", padding=True,
                ).to(device)
                parts.append(forward_logit(model_unwrapped, processor, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            size_holder[0] = max(1, chunk_size // 2)
            print(f"[OOM] chunk_size {chunk_size} -> {size_holder[0]} (이후 계속 이 값 사용)", flush=True)


def forward_batch(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device):
    return _chunked_forward(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, _current_minibatch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--prev_best_val", type=float, default=0.0)
    args = parser.parse_args()

    lora_alpha = int(round(args.lora_r * LORA_ALPHA_RATIO))
    ckpt_name = args.ckpt_name or f"best_v21_r{args.lora_r}"

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v21] Processes={accelerator.num_processes}  Device={device}")
        print(f"  base=Qwen3-VL-32B-Instruct QLoRA(4bit-NF4)  lora_r={args.lora_r}  lora_alpha={lora_alpha}  lr={args.lr}")
        print(f"  Loss: ListwiseSoftmaxLoss (v14/v20과 동일)")
        print(f"  Hard negatives: LIVE resampled every __getitem__ (v14/v20과 동일, K=7 d=1..6)")
        print(f"  EPOCHS={EPOCHS}  ckpt_name={ckpt_name}")

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

    train_ds = GroupedTemporalDataset(trn_raw)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2,
    )

    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=lora_alpha,
                                                  resume_from=args.resume_from)
    yes_id, no_id = get_yes_no_token_ids(processor)
    criterion = ListwiseSoftmaxLoss()

    remaining_epochs = EPOCHS - args.start_epoch + 1
    n_steps = (len(train_dl) // GRAD_ACCUM) * remaining_epochs
    n_warmup = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)

    best_val_acc = args.prev_best_val
    global_step = 0

    for epoch in range(args.start_epoch, EPOCHS + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}")
            print(f"  Effective batch = {BATCH_SIZE}×{GRAD_ACCUM}×{accelerator.num_processes}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                texts, imgs_list = [], []
                group_offsets = []
                for grp_imgs, grp_sents, grp_dists in zip(
                    batch["images"], batch["sentences"], batch["dists"]
                ):
                    start = len(texts)
                    for imgs, sent in zip(grp_imgs, grp_sents):
                        msg = build_messages(imgs, sent)
                        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                        texts.append(text)
                        imgs_list.append(imgs)
                    group_offsets.append((start, len(grp_imgs), grp_dists))

                logits = forward_batch(unwrapped, processor, texts, imgs_list, yes_id, no_id, device)

                ord_logits_parts, ord_dists, ord_sizes = [], [], []
                for start, size, grp_dists in group_offsets:
                    ord_logits_parts.append(logits[start: start + size])
                    ord_dists.extend(grp_dists)
                    ord_sizes.append(size)

                loss, _, _ = criterion(torch.cat(ord_logits_parts), ord_dists, ord_sizes)

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                try:
                    accelerator.backward(loss)
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    if _current_minibatch[0] > 1:
                        _current_minibatch[0] = max(1, _current_minibatch[0] // 2)
                    print(f"[OOM-backward] step {global_step} 스킵, TRAIN_MINIBATCH->{_current_minibatch[0]}.", flush=True)
                    continue
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss += loss.item()

                if is_main and global_step % LOGGING_STEPS == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg = epoch_loss / (step + 1)
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  step={global_step:5d}  loss={avg:.4f}  lr={lr_now:.2e}  peakVRAM={peak:.1f}GB")

        accelerator.wait_for_everyone()

        if is_main:
            elapsed = (time.time() - t0) / 60
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]  소요: {elapsed:.1f}분  "
                  f"avg_loss: {epoch_loss / len(train_dl):.4f}")

            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, yes_id, no_id, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = CKPT_DIR / ckpt_name
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"  ★ Best saved (val_acc={val_acc:.4f})")

            last_path = CKPT_DIR / (ckpt_name + "_last")
            unwrapped.save_pretrained(last_path)
            processor.save_pretrained(last_path)
            print(f"  → Last saved to {last_path.name}")

        accelerator.wait_for_everyone()

    if is_main:
        print(f"\n학습 완료. Best val_acc: {best_val_acc:.4f}  ckpt_name={ckpt_name}")
        # 다음 단계(추론)에서 어떤 체크포인트를 쓸지 셸 스크립트가 읽을 수 있게 파일로 남김
        (CKPT_DIR / "LAST_CKPT_NAME.txt").write_text(ckpt_name)


if __name__ == "__main__":
    main()
