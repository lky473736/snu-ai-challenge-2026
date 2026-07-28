"""
Val 오답 분석: best checkpoint로 _val_raw.csv 전체(478개)를 24-perm scoring해서
gt와 pred 사이 켄달타우 거리(필요한 adjacent swap 횟수)로 오답을 분류.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import build_messages, load_image
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def kendall_tau_distance(pred: list, gt: list) -> int:
    """pred를 gt로 만들기 위해 필요한 adjacent swap(transposition) 횟수 = inversion count."""
    # gt 기준 상대 순위로 변환
    pos_in_gt = {v: i for i, v in enumerate(gt)}
    relative = [pos_in_gt[v] for v in pred]
    inversions = 0
    for i in range(len(relative)):
        for j in range(i + 1, len(relative)):
            if relative[i] > relative[j]:
                inversions += 1
    return inversions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading processor & model...")
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(CKPT_DIR / "best"))
    model.eval()

    yes_id, no_id = get_yes_no_token_ids(processor)

    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv")
    val_df = val_df[val_df["No_ordering"] == False].reset_index(drop=True)
    print(f"Val samples (ordering only): {len(val_df)}")

    results = []
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
        with torch.no_grad():
            for perm in ALL_PERMS:
                perm_list = list(perm)
                imgs = order_to_images(perm_list)
                msgs = build_messages(imgs, sentence)
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp  = processor(text=[text], images=imgs, return_tensors="pt").to(device)
                s = forward_logit(model, processor, inp, yes_id, no_id)
                scores.append((s.item(), perm_list))

        scores.sort(key=lambda x: -x[0])
        best = scores[0][1]
        gt_rank = next(i for i, (_, p) in enumerate(scores) if p == gt)
        dist = kendall_tau_distance(best, gt)

        results.append({
            "id": sample_id,
            "gt": gt,
            "pred": best,
            "correct": best == gt,
            "kendall_tau_dist": dist,
            "gt_rank": gt_rank,        # gt가 24개 중 몇 등으로 스코어링됐는지 (0=1등)
            "gt_score": next(s for s, p in scores if p == gt),
            "best_score": scores[0][0],
        })

    res_df = pd.DataFrame(results)
    out_path = CKPT_DIR / "val_diagnosis.csv"
    res_df.to_csv(out_path, index=False)

    acc = res_df["correct"].mean()
    print(f"\n=== Val Exact-Match Accuracy: {acc:.4f} ({res_df['correct'].sum()}/{len(res_df)}) ===")

    wrong = res_df[~res_df["correct"]]
    print(f"\n오답 {len(wrong)}개 / 켄달타우 거리 분포 (필요한 adjacent swap 횟수):")
    print(wrong["kendall_tau_dist"].value_counts().sort_index())

    print(f"\n오답 중 gt 순위 분포 (0=best였는데 다른 동점 처리 등 edge case, 1=2등...):")
    print(wrong["gt_rank"].value_counts().sort_index())

    near_miss = (wrong["kendall_tau_dist"] <= 1).sum()
    far_miss  = (wrong["kendall_tau_dist"] > 1).sum()
    print(f"\nnear-miss (거리<=1, 즉 한 번의 adjacent swap 차이): {near_miss}")
    print(f"far-miss  (거리>1, 더 멀리 틀림): {far_miss}")

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
