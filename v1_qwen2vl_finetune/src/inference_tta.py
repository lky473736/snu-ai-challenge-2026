"""
HNTV Inference with TTA (horizontal flip) + improved prompt
- 24 permutations × (original + H-flip) scores averaged → argmax
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from itertools import permutations
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import List

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import load_image
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))

PROMPT_TEMPLATE = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Carefully examine object positions, body postures, and state changes "
    "between consecutive frames to determine temporal direction.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)


def build_messages(images: List[Image.Image], sentence: str):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_TEMPLATE.format(sentence=sentence)})
    return [
        {
            "role": "system",
            "content": (
                "You are a temporal ordering assistant. "
                "Given 4 video frames in a specific order and a caption, "
                "determine if the frames are in the correct chronological order."
            ),
        },
        {"role": "user", "content": content},
    ]


def score_perm(model, processor, yes_id, no_id, device, imgs, sentence):
    msgs = build_messages(imgs, sentence)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[text], images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        return forward_logit(model, processor, inp, yes_id, no_id).item()


def score_all_permutations_tta(model, processor, yes_id, no_id, device,
                                img_files, img_dir, sentence):
    results = []
    for perm in ALL_PERMS:
        inv = [0] * 4
        for inp_idx, t_pos in enumerate(perm):
            inv[t_pos - 1] = inp_idx

        try:
            imgs_orig = [load_image(str(img_dir / img_files[inv[t]])) for t in range(4)]
        except Exception as e:
            print(f"[tta] 이미지 로드 실패: {e}")
            results.append((float("-inf"), list(perm)))
            continue

        # TTA: horizontal flip
        imgs_flip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs_orig]

        s_orig = score_perm(model, processor, yes_id, no_id, device, imgs_orig, sentence)
        s_flip = score_perm(model, processor, yes_id, no_id, device, imgs_flip, sentence)
        results.append(((s_orig + s_flip) / 2.0, list(perm)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="best_v1_score0.79057")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_DIR / args.ckpt_name
    print(f"Checkpoint: {ckpt_path}")
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    yes_id, no_id = get_yes_no_token_ids(processor)
    print(f"yes_id={yes_id}  no_id={no_id}")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Test samples: {len(test_df)}  (24 perms × 2 TTA = 48 forwards each)")

    submission = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        img_dir   = DATA_DIR / "test" / sample_id
        img_files = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])

        scores = score_all_permutations_tta(
            model, processor, yes_id, no_id, device, img_files, img_dir, sentence
        )
        best_order = max(scores, key=lambda x: x[0])[1]
        submission.append({"Id": sample_id, "Answer": str(best_order)})

    out_path = args.out if args.out else str(DATA_DIR.parent / "submission_tta.csv")
    pd.DataFrame(submission).to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
