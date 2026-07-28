"""
v6 진단: best_v5 체크포인트를 재학습 없이, "이미지 1번 인코딩 + 24개 후보 순열 우도 채점"
디코딩으로만 바꿔서 val 정확도를 다시 측정. 그리디 디코딩(0.4475)과 비교.
"""
import sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import build_messages, load_image
from src.model import _get_model_class, get_digit_comma_ids

ALL_PERMS = list(permutations([1, 2, 3, 4]))
IM_END = "<|im_end|>\n"


def score_all_candidates(model, processor, imgs, sentence, digit_ids, comma_id, device):
    """
    (진단용, KV캐시 재사용 없이 안전하게) 24개 후보 순열 문자열 전부에 대해
    이미지+프롬프트+후보를 매번 완전히 다시 forward해서 teacher-forced 우도 채점.
    이미지 재인코딩 최적화는 뺐지만(느림), 정확도 진단 목적엔 충분히 정확함.
    """
    msgs = build_messages(imgs, sentence)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    scores = []
    with torch.no_grad():
        for perm in ALL_PERMS:
            target = ",".join(str(x) for x in perm)
            target_ids = processor.tokenizer.encode(target, add_special_tokens=False)

            prompt_inputs = processor(text=[prompt_text], images=[imgs], return_tensors="pt")
            target_ids_t = torch.tensor([target_ids], dtype=prompt_inputs["input_ids"].dtype)
            input_ids = torch.cat([prompt_inputs["input_ids"], target_ids_t], dim=1).to(device)
            attn = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(target_ids_t)], dim=1).to(device)
            mm_types = torch.cat([prompt_inputs["mm_token_type_ids"], torch.zeros_like(target_ids_t)], dim=1).to(device)
            prefix_len = prompt_inputs["input_ids"].shape[1]

            out = model(
                input_ids=input_ids, attention_mask=attn, mm_token_type_ids=mm_types,
                pixel_values=prompt_inputs["pixel_values"].to(device),
                image_grid_thw=prompt_inputs["image_grid_thw"].to(device),
            )
            logp = torch.log_softmax(out.logits[0, prefix_len - 1: -1, :], dim=-1)  # (len(target_ids), vocab)
            logprob_sum = sum(logp[i, target_ids[i]].item() for i in range(len(target_ids)))
            scores.append((logprob_sum, list(perm)))

    best = max(scores, key=lambda x: x[0])[1]
    return best


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    ckpt = str(CKPT_DIR / "best_v5")
    processor = AutoProcessor.from_pretrained(ckpt)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_model, ckpt).to(device)
    model.eval()
    digit_ids, comma_id = get_digit_comma_ids(processor)

    # 원래 v5 val_exact_match와 동일하게 No_ordering 포함 476개 전부 사용 (공정 비교)
    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv")
    if args.n_samples:
        val_df = val_df.sample(args.n_samples, random_state=42).reset_index(drop=True)

    correct, wrong = 0, []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        sid = row["Id"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        imgs = [load_image(str(img_dir / f)) for f in files]
        pred = score_all_candidates(model, processor, imgs, row["Sentence"], digit_ids, comma_id, device)
        if pred == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if pred[i] != gt[i]]
            wrong.append("adj_swap" if len(diffs) == 2 and abs(diffs[0]-diffs[1]) == 1 else "other")

    acc = correct / len(val_df)
    adj = wrong.count("adj_swap")
    print(f"\n[24-way rescore] exact_match={acc:.4f} ({correct}/{len(val_df)})  adj_swap_fail={adj}  other_fail={len(wrong)-adj}")
    print(f"비교: greedy 디코딩(원래 v5) exact_match=0.4475 (213/476)")


if __name__ == "__main__":
    main()
