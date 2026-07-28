"""
HNTV Inference
24개 permutation 전부 스코어링 → 최고 Yes logit 선택
배치 처리로 속도 최적화
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from itertools import permutations
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from config import DATA_DIR, MODEL_PATH, CKPT_DIR, INFER_BATCH_SIZE
from src.dataset import build_messages, load_image
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))  # 24가지


def score_all_permutations(
    model, processor, yes_id, no_id, device,
    img_files: list, img_dir: Path, sentence: str
) -> list:
    """
    24가지 순서에 대한 Yes logit 계산 → [(score, perm_list), ...]
    INFER_BATCH_SIZE 단위로 묶어서 처리
    """
    results = []

    for batch_start in range(0, len(ALL_PERMS), INFER_BATCH_SIZE):
        batch_perms = ALL_PERMS[batch_start: batch_start + INFER_BATCH_SIZE]

        batch_texts  = []
        batch_images = []

        for perm in batch_perms:
            # perm[i] = Input_{i+1}의 temporal position
            # inverse: temporal t → input index
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(perm):
                inv[t_pos - 1] = inp_idx

            try:
                imgs = [load_image(str(img_dir / img_files[inv[t]])) for t in range(4)]
            except Exception as e:
                print(f"[inference] 이미지 로드 실패: {e}")
                imgs = None

            if imgs is None:
                results.append((float("-inf"), list(perm)))
                continue

            msgs = build_messages(imgs, sentence)
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)
            batch_images.append(imgs)

        if not batch_texts:
            continue

        # 배치 인코딩 (이미지 수가 달라서 개별 처리)
        with torch.no_grad():
            for i, (text, imgs) in enumerate(zip(batch_texts, batch_images)):
                inp   = processor(text=[text], images=imgs, return_tensors="pt").to(device)
                score = forward_logit(model, processor, inp, yes_id, no_id)
                perm  = list(batch_perms[i])
                results.append((score.item(), perm))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="best", help="checkpoints/ 하위 폴더명")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_DIR / args.ckpt_name
    print(f"Loading checkpoint: {ckpt_path}")
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    yes_id, no_id = get_yes_no_token_ids(processor)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Test samples: {len(test_df)}")

    submission = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        img_dir   = DATA_DIR / "test" / sample_id
        img_files = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])

        scores = score_all_permutations(
            model, processor, yes_id, no_id, device, img_files, img_dir, sentence
        )
        best_order = max(scores, key=lambda x: x[0])[1]

        submission.append({"Id": sample_id, "Answer": str(best_order)})

    sub_df = pd.DataFrame(submission)
    out_path = DATA_DIR.parent / "submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"\n제출 파일 저장: {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
