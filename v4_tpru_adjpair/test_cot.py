"""
재학습 없이, best_v4 체크포인트에 CoT 프롬프트만 추가해서 val 정확도 변화 확인.
방식: (1) 짧은 reasoning을 generate() (2) reasoning까지 이어붙인 텍스트로 1회 재forward해서
      log P(Yes)-log P(No) 점수 산출 (수동 KV캐시 이어붙이기 없이 안전하게 재인코딩)
24개 순열 각각에 대해 위 점수를 구해 argmax -> 기존 방식과 동일한 채점 로직.
"""
import sys, os, ast, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import load_image
from src.model import _get_model_class, get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))

SYSTEM = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
)

COT_PROMPT = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "In one short sentence, note the most obvious visual change between consecutive frames "
    "(e.g. object position, posture, state).\n"
    "Then answer: Is this the correct chronological order of events?\n"
    "Format your response as:\nReasoning: <one sentence>\nAnswer: <Yes or No>"
)


def build_cot_messages(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": COT_PROMPT.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def score_with_cot(model, processor, imgs, sentence, yes_id, no_id, device, max_new_tokens=50):
    msgs = build_cot_messages(imgs, sentence)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[imgs], return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id)
        reasoning = processor.tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # reasoning까지 포함해서 안전하게(수동 캐시 이어붙이기 없이) 재forward, "Answer:" 다음 위치의 Yes/No 점수
        full_text = prompt_text + reasoning
        if "Answer:" not in reasoning:
            full_text += "\nAnswer:"
        full_inputs = processor(text=[full_text], images=[imgs], return_tensors="pt").to(device)
        score = forward_logit(model, processor, full_inputs, yes_id, no_id)
    return score.item(), reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    ckpt = str(CKPT_DIR / "best_v4")
    processor = AutoProcessor.from_pretrained(ckpt)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_model, ckpt).to(device)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv").sample(args.n_samples, random_state=42).reset_index(drop=True)

    correct, wrong = 0, []
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        sid = row["Id"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        scores = []
        sample_reasonings = []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            s, reasoning = score_with_cot(model, processor, imgs, row["Sentence"], yes_id, no_id, device)
            scores.append((s, list(perm)))
            if len(sample_reasonings) < 1:
                sample_reasonings.append(reasoning)

        best = max(scores, key=lambda x: x[0])[1]
        ok = best == gt
        correct += ok
        if idx < 3:
            print(f"\n[sample {idx}] gt={gt} pred={best} ok={ok}")
            print(f"  reasoning 예시: {sample_reasonings[0][:150]}")
        if not ok:
            wrong.append((gt, best))

    acc = correct / len(val_df)
    print(f"\n[CoT 프롬프트, n={len(val_df)}] exact_match={acc:.4f} ({correct}/{len(val_df)})")


if __name__ == "__main__":
    main()
