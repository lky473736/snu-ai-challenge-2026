"""
TPRU-7B (또는 임의 VLM) zero-shot val 평가
LoRA 없이 base 모델만으로 24-permutation 스코어링 → val exact match

사용법:
  python src/eval_zero_shot.py
  python src/eval_zero_shot.py --n_samples 100  (빠른 체크)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import argparse
import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLProcessor

from config import DATA_DIR, MODEL_PATH, CKPT_DIR, SEED
from src.dataset import build_messages, load_image
from src.model import _get_model_class, get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--n_samples",  type=int, default=399, help="val에서 몇 개 평가할지")
    parser.add_argument("--val_csv",    type=str, default=str(CKPT_DIR / "_val_raw.csv"),
                        help="val CSV 경로 (없으면 train.csv에서 5% split)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {args.model_path}")
    print(f"Device: {device}")

    # ── val 데이터 로드 ──────────────────────────────────────
    val_csv_path = args.val_csv
    if os.path.exists(val_csv_path):
        val_df = pd.read_csv(val_csv_path)
        print(f"Val CSV loaded: {val_csv_path} ({len(val_df)} rows)")
    else:
        # fallback: train.csv에서 5% split
        print(f"Val CSV not found ({val_csv_path}), splitting from train.csv...")
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val = int(len(train_df) * 0.05)
        val_df = train_df[:n_val]
        print(f"  → {len(val_df)} val samples from train.csv")

    # No_ordering 제외 (정답이 항상 [1,2,3,4]인 샘플 — 평가 의미 없음)
    if "No_ordering" in val_df.columns:
        val_df = val_df[val_df["No_ordering"] != True].reset_index(drop=True)
        print(f"After filtering No_ordering: {len(val_df)} samples")

    val_df = val_df.sample(min(args.n_samples, len(val_df)), random_state=SEED)
    print(f"Evaluating: {len(val_df)} samples\n")

    # ── 모델 로드 (LoRA 없음) ────────────────────────────────
    ModelClass = _get_model_class(args.model_path)
    print(f"Loading {ModelClass.__name__}...")
    # VideoR1-7B is a training checkpoint with broken processor config;
    # fall back to TPRU-7B processor (same Qwen2.5-VL base, identical vocab/tokenizer)
    FALLBACK_PROCESSOR = "/data/gyuyeonlim/models/TPRU-7B"
    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
    except (ValueError, OSError):
        print(f"  processor load failed, using fallback: {FALLBACK_PROCESSOR}")
        processor = AutoProcessor.from_pretrained(FALLBACK_PROCESSOR)
    model = ModelClass.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)
    print(f"yes_id={yes_id}  no_id={no_id}\n")

    # ── 24-permutation 스코어링 ──────────────────────────────
    correct = 0
    wrong_cases = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
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

        scores = []
        for perm in ALL_PERMS:
            perm_list = list(perm)
            imgs  = order_to_images(perm_list)
            msgs  = build_messages(imgs, sentence)
            text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp   = processor(text=[text], images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                s = forward_logit(model, processor, inp, yes_id, no_id)
            scores.append((s.item(), perm_list))

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    acc = correct / len(val_df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n{'='*50}")
    print(f"[Zero-shot Val Result]")
    print(f"  Model     : {args.model_path}")
    print(f"  Samples   : {len(val_df)}")
    print(f"  Exact Match: {acc:.4f} ({correct}/{len(val_df)})")
    print(f"  adj_swap fail: {adj}  other fail: {len(wrong_cases) - adj}")
    print(f"{'='*50}")
    print(f"\n비교 기준:")
    print(f"  Qwen2-VL-7B + LoRA v1 fine-tuned: 0.5013 (200/399)")
    print(f"  TPRU-7B zero-shot (이번):         {acc:.4f} ({correct}/{len(val_df)})")


if __name__ == "__main__":
    main()
