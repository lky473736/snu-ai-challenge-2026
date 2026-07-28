"""
HNTV CoT Inference - Reasoning Prefix Injection
Step 1: 모델이 프레임 변화를 추론하는 텍스트 생성 (seed: REASON_SEED)
Step 2: 생성된 추론을 context로 추가 후 Yes/No logit 측정
재학습 없이 inference 단계에서만 CoT 효과 적용
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

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import build_messages, load_image
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))

# 모델이 reasoning을 먼저 생성하도록 유도하는 seed 텍스트
REASON_SEED = "Let me analyze the changes between consecutive frames: "

MIN_REASON_TOKENS = 5  # 이보다 짧으면 reasoning 생성 실패로 간주 → fallback


def score_one_perm(model, processor, yes_id, no_id, device, imgs, sentence, max_reason_tokens=80):
    """
    1단계: reasoning seed로 추론 텍스트 생성
    2단계: 추론 포함 전체 context로 Yes/No logit 측정
    생성 실패(너무 짧음) 시 기존 방식으로 fallback
    """
    msgs = build_messages(imgs, sentence)
    text_base = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ── 1단계: 추론 생성 ─────────────────────────────────────
    text_seeded = text_base + REASON_SEED
    inp = processor(text=[text_seeded], images=imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inp,
            max_new_tokens=max_reason_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_ids = generated[0][inp["input_ids"].shape[1]:]
    reasoning = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # 생성 실패 시 기존 방식 fallback
    if len(new_ids) < MIN_REASON_TOKENS:
        inp_base = processor(text=[text_base], images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            return forward_logit(model, processor, inp_base, yes_id, no_id).item()

    # ── 2단계: 추론 포함 context로 Yes/No logit 측정 ─────────
    text_with_reason = text_base + REASON_SEED + reasoning + "\nFinal answer:"
    inp2 = processor(text=[text_with_reason], images=imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        score = forward_logit(model, processor, inp2, yes_id, no_id)

    return score.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name",         type=str, default="best_v1_score0.79057")
    parser.add_argument("--max_reason_tokens", type=int, default=80)
    parser.add_argument("--out_name",          type=str, default="submission_cot.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_DIR / args.ckpt_name
    print(f"Checkpoint: {ckpt_path}")
    print(f"Max reasoning tokens: {args.max_reason_tokens}")

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
    sample_log = []  # 처음 3개 reasoning 출력용

    for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        img_dir   = DATA_DIR / "test" / sample_id
        img_files = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])

        def order_to_images(perm):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(perm):
                inv[t_pos - 1] = inp_idx
            return [load_image(str(img_dir / img_files[inv[t]])) for t in range(4)]

        scores = []
        for perm_idx, perm in enumerate(ALL_PERMS):
            imgs = order_to_images(list(perm))

            # 첫 번째 샘플의 첫 순열에서 reasoning 텍스트 샘플 출력
            if idx == 0 and perm_idx == 0:
                msgs = build_messages(imgs, sentence)
                text_base = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                text_seeded = text_base + REASON_SEED
                inp_tmp = processor(text=[text_seeded], images=imgs, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_tmp = model.generate(
                        **inp_tmp, max_new_tokens=args.max_reason_tokens,
                        do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
                    )
                new_ids_tmp = gen_tmp[0][inp_tmp["input_ids"].shape[1]:]
                sample_reason = processor.tokenizer.decode(new_ids_tmp, skip_special_tokens=True)
                print(f"\n[Sample reasoning (id={sample_id}, perm={list(perm)})]")
                print(f"  Sentence: {sentence[:80]}")
                print(f"  Reasoning: {sample_reason[:200]}")

            s = score_one_perm(
                model, processor, yes_id, no_id, device,
                imgs, sentence, args.max_reason_tokens
            )
            scores.append((s, list(perm)))

        best_order = max(scores, key=lambda x: x[0])[1]
        submission.append({"Id": sample_id, "Answer": str(best_order)})

    sub_df = pd.DataFrame(submission)
    out_path = DATA_DIR.parent / args.out_name
    sub_df.to_csv(out_path, index=False)
    print(f"\n제출 파일 저장: {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
