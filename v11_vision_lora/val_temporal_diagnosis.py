"""
train/val 내부에서만 "temporal word 개수 vs 정확도" 상관관계 검증.
test는 전혀 안 봄 — v1의 diagnose_val.py(문장 길이 vs 정확도, r=0.376)와 동일한 방법론을
best_v6_5 체크포인트 + val 476개(동일 SEED/VAL_RATIO)에 적용.
"""
import ast
from itertools import permutations

import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor

from config import DATA_DIR, CKPT_DIR, MODEL_PATH, INFER_BATCH_SIZE
from src.dataset import load_image, build_messages
from src.model import get_yes_no_token_ids, forward_logit, _get_model_class

ALL_PERMS = list(permutations([1, 2, 3, 4]))
CKPT_PATH = CKPT_DIR / "best_v6_5"
device = torch.device("cuda:0")

TEMPORAL_WORDS = ["then", "before", "after", "while", "as", "next",
                   "finally", "first", "second", "third"]


def count_temporal_words(s: str) -> int:
    s = s.lower()
    return sum(s.count(w) for w in TEMPORAL_WORDS)


def _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, chunk_size):
    while True:
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(
                    text=texts[bi: bi + chunk_size],
                    images=imgs_list[bi: bi + chunk_size],
                    return_tensors="pt", padding=True,
                ).to(device)
                parts.append(forward_logit(model, processor, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] chunk_size -> {chunk_size}", flush=True)


def main():
    print(f"Loading base model + LoRA from {CKPT_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = PeftModel.from_pretrained(base_model, str(CKPT_PATH)).to(device)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv")
    val_df = val_df[val_df["No_ordering"] == False].reset_index(drop=True)
    print(f"Val samples (ordering only): {len(val_df)}")

    results = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        sample_id = row["Id"]
        sentence = row["Sentence"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sample_id
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        texts, imgs_list = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)

        with torch.no_grad():
            s = _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, INFER_BATCH_SIZE)
        scores = [(s[i].item(), list(perm)) for i, perm in enumerate(ALL_PERMS)]
        best = max(scores, key=lambda x: x[0])[1]

        results.append({
            "id": sample_id,
            "sentence": sentence,
            "correct": best == gt,
            "sent_len": len(sentence.split()),
            "temporal_words": count_temporal_words(sentence),
        })

    res_df = pd.DataFrame(results)
    out_path = CKPT_DIR / "val_temporal_diagnosis.csv"
    res_df.to_csv(out_path, index=False)

    acc = res_df["correct"].mean()
    print(f"\n=== Val Exact-Match Accuracy: {acc:.4f} ({res_df['correct'].sum()}/{len(res_df)}) ===")

    corr_len = res_df["sent_len"].corr(res_df["correct"].astype(int))
    corr_tw  = res_df["temporal_words"].corr(res_df["correct"].astype(int))
    print(f"\n상관계수 (sent_len vs correct)        = {corr_len:.3f}  (참고: v1 기준 0.376)")
    print(f"상관계수 (temporal_words vs correct)   = {corr_tw:.3f}  (신규 검증)")

    print("\ntemporal_words 구간별 정확도:")
    bins = [-1, 0, 1, 2, 3, 4, 100]
    labels = ["0", "1", "2", "3", "4", "5+"]
    res_df["tw_bin"] = pd.cut(res_df["temporal_words"], bins=bins, labels=labels)
    summary = res_df.groupby("tw_bin", observed=True).agg(
        accuracy=("correct", "mean"), n=("correct", "size")
    )
    print(summary)

    print("\nsent_len 구간별 정확도 (교차검증용):")
    bins2 = [0, 15, 20, 25, 30, 35, 200]
    labels2 = ["<=15", "15-20", "20-25", "25-30", "30-35", "35+"]
    res_df["len_bin"] = pd.cut(res_df["sent_len"], bins=bins2, labels=labels2)
    summary2 = res_df.groupby("len_bin", observed=True).agg(
        accuracy=("correct", "mean"), n=("correct", "size")
    )
    print(summary2)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
