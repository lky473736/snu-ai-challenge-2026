"""
v5 — Qwen3-VL-8B-Instruct + LoRA, listwise 순열 직접 생성(SFT)
hard-negative/adaptive loss 없음. train.csv 각 행 = 학습 샘플 1개.
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
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
    LOGGING_STEPS, VAL_RATIO, SEED,
    LORA_R, LORA_ALPHA,
)
from src.dataset import ListwiseDataset, collate_fn, build_messages, load_image
from src.model import load_model_and_processor, get_digit_comma_ids, generate_permutation

IM_END = "<|im_end|>\n"


def build_batch_inputs(processor, batch, device):
    """샘플별로 prompt를 processor(images+text)로 처리 -> target 토큰을 뒤에 이어붙임 -> 배치 패딩"""
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    per_sample = []
    for imgs, sent, target in zip(batch["images"], batch["sentences"], batch["targets"]):
        msgs = build_messages(imgs, sent)
        prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p = processor(text=[prompt_text], images=[imgs], return_tensors="pt")
        target_ids = tok.encode(target + IM_END, add_special_tokens=False)
        target_ids_t = torch.tensor([target_ids], dtype=p["input_ids"].dtype)

        input_ids = torch.cat([p["input_ids"], target_ids_t], dim=1)
        attn      = torch.cat([p["attention_mask"], torch.ones_like(target_ids_t)], dim=1)
        mm_types  = torch.cat([p["mm_token_type_ids"], torch.zeros_like(target_ids_t)], dim=1)
        labels    = torch.cat([torch.full_like(p["input_ids"], -100), target_ids_t], dim=1)

        per_sample.append({
            "input_ids": input_ids[0], "attention_mask": attn[0],
            "mm_token_type_ids": mm_types[0], "labels": labels[0],
            "pixel_values": p["pixel_values"], "image_grid_thw": p["image_grid_thw"],
        })

    max_len = max(s["input_ids"].shape[0] for s in per_sample)

    def pad1d(t, pad_val):
        n = max_len - t.shape[0]
        if n == 0:
            return t
        return torch.cat([t, torch.full((n,), pad_val, dtype=t.dtype)])

    input_ids = torch.stack([pad1d(s["input_ids"], pad_id) for s in per_sample])
    attn      = torch.stack([pad1d(s["attention_mask"], 0) for s in per_sample])
    mm_types  = torch.stack([pad1d(s["mm_token_type_ids"], 0) for s in per_sample])
    labels    = torch.stack([pad1d(s["labels"], -100) for s in per_sample])
    pixel_values   = torch.cat([s["pixel_values"] for s in per_sample], dim=0)
    image_grid_thw = torch.cat([s["image_grid_thw"] for s in per_sample], dim=0)

    return {
        "input_ids": input_ids.to(device), "attention_mask": attn.to(device),
        "mm_token_type_ids": mm_types.to(device), "labels": labels.to(device),
        "pixel_values": pixel_values.to(device), "image_grid_thw": image_grid_thw.to(device),
    }


def val_exact_match(model, processor, val_df, digit_ids, comma_id, device, n_samples=None):
    model.eval()
    df = val_df if n_samples is None else val_df.sample(min(n_samples, len(val_df)), random_state=SEED)
    correct, wrong_cases = 0, []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sid = row["Id"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        imgs = [load_image(str(img_dir / f)) for f in files]
        try:
            pred = generate_permutation(model, processor, imgs, row["Sentence"], digit_ids, comma_id, device, build_messages)
        except Exception as e:
            pred = None
        if pred == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if pred is None or pred[i] != gt[i]] if pred and len(pred) == 4 else list(range(4))
            t = "adj_swap" if (pred and len(pred) == 4 and len(diffs) == 2 and abs(diffs[0]-diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sid, "gt": gt, "pred": pred, "type": t})
    acc = correct / len(df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[Val] exact_match={acc:.4f} ({correct}/{len(df)})  adj_swap_fail={adj}  other_fail={len(wrong_cases)-adj}")
    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--ckpt_name", type=str, default="best_v5")
    args = parser.parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v5] Processes={accelerator.num_processes}  Device={device}")
        print(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  lr={args.lr}")
        print("  방식: listwise 순열 직접 생성 SFT (hard-negative 없음)")

    torch.manual_seed(SEED)

    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val = int(len(train_csv) * VAL_RATIO)
        val_raw = train_csv[:n_val]
        trn_raw = train_csv[n_val:]
        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)
        trn_raw.to_csv(CKPT_DIR / "_trn_raw.csv", index=False)
        print(f"Train: {len(trn_raw)}  Val: {len(val_raw)}")

    accelerator.wait_for_everyone()
    trn_raw = pd.read_csv(CKPT_DIR / "_trn_raw.csv")
    val_raw = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    train_ds = ListwiseDataset(trn_raw)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True, prefetch_factor=2)

    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    digit_ids, comma_id = get_digit_comma_ids(processor)

    n_steps = (len(train_dl) // GRAD_ACCUM) * EPOCHS
    n_warmup = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}\nEpoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}\n{'='*60}")

        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)
                inputs = build_batch_inputs(processor, batch, device)
                out = unwrapped(**inputs)
                loss = out.loss

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

                if is_main and global_step % LOGGING_STEPS == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg = epoch_loss / (step + 1)
                    print(f"  step={global_step:5d}  loss={avg:.4f}  lr={lr_now:.2e}")

        accelerator.wait_for_everyone()
        if is_main:
            elapsed = (time.time() - t0) / 60
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]  소요: {elapsed:.1f}분  avg_loss: {epoch_loss/len(train_dl):.4f}")
            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, digit_ids, comma_id, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = CKPT_DIR / args.ckpt_name
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"  ★ Best saved (val_acc={val_acc:.4f})")
            last_path = CKPT_DIR / (args.ckpt_name + "_last")
            unwrapped.save_pretrained(last_path)
            processor.save_pretrained(last_path)
        accelerator.wait_for_everyone()

    if is_main:
        print(f"\n학습 완료. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
