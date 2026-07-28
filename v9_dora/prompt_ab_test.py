"""
프롬프트 A/B 테스트 — 재학습 없이 기존 best_v6_5 체크포인트로 val(476개)에서만 스크리닝.
"새 프롬프트가 확실히 낫다"를 증명하진 못함(모델이 그 프롬프트에 맞춰 학습된 게 아니므로) —
다만 최소한 망가뜨리지 않는지, 눈에 띄게 좋아지는지 정도의 저비용 신호를 얻기 위함.
단일 GPU로 실행: python prompt_ab_test.py
"""
import ast
from itertools import permutations

import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel

from config import DATA_DIR, CKPT_DIR, MODEL_PATH, INFER_BATCH_SIZE
from src.dataset import load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))
CKPT_PATH = CKPT_DIR / "best_v6_5"
device = torch.device("cuda:0")


# ── 프롬프트 변형 3종 ────────────────────────────────────────────────
SYSTEM_BASELINE = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
)
PROMPT_BASELINE = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Please carefully examine the changes between consecutive frames.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)

SYSTEM_V2 = (
    "You are an expert temporal ordering assistant specialized in human-action videos. "
    "Given 4 frames and a caption, determine whether the frames are arranged in the "
    "chronological order that matches the caption."
)
PROMPT_V2 = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n\n"
    "Compare each pair of CONSECUTIVE frames carefully. Pay special attention to:\n"
    "- the direction of motion or body-pose progression across frames\n"
    "- whether the middle two frames could be swapped without you noticing "
    "(adjacent-frame swaps are the most visually subtle type of error)\n\n"
    "Is this the correct chronological order of events matching the sentence?\n"
    "Answer only with \"Yes\" or \"No\"."
)

PROMPT_V3_SHORT_EXTRA = (
    "- Since the caption is brief, rely mainly on visual progression rather than the wording.\n"
)


def build_messages_baseline(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_BASELINE.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM_BASELINE}, {"role": "user", "content": content}]


def build_messages_v2(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_V2.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": content}]


def build_messages_v3(images, sentence):
    is_short = len(sentence.split()) <= 15
    prompt = PROMPT_V2.format(sentence=sentence)
    if is_short:
        marker = "(adjacent-frame swaps are the most visually subtle type of error)\n\n"
        prompt = prompt.replace(marker, marker[:-1] + PROMPT_V3_SHORT_EXTRA + "\n")
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": content}]


VARIANTS = {
    "A_baseline": build_messages_baseline,
    "B_adjswap_hint": build_messages_v2,
    "C_adjswap_plus_shortlen": build_messages_v3,
}


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


def val_exact_match(model, processor, val_df, yes_id, no_id, build_messages_fn):
    correct, wrong_cases = 0, []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=build_messages_fn.__name__, leave=False):
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

        texts, imgs_list = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_messages_fn(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)

        with torch.no_grad():
            s = _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, INFER_BATCH_SIZE)
        scores = [(s[i].item(), list(perm)) for i, perm in enumerate(ALL_PERMS)]
        best = max(scores, key=lambda x: x[0])[1]

        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append(t)

    acc = correct / len(val_df)
    adj = sum(1 for t in wrong_cases if t == "adj_swap")
    other = len(wrong_cases) - adj
    return acc, correct, adj, other


def main():
    print(f"Loading base model + LoRA from {CKPT_PATH} ...")
    processor = None
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    from src.model import _get_model_class
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = PeftModel.from_pretrained(base_model, str(CKPT_PATH)).to(device)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv")
    print(f"Val samples: {len(val_df)}")

    print(f"\n{'='*60}")
    print("프롬프트 A/B 테스트 (best_v6_5 체크포인트, 재학습 없음)")
    print(f"{'='*60}")

    results = {}
    for name, fn in VARIANTS.items():
        acc, correct, adj, other = val_exact_match(model, processor, val_df, yes_id, no_id, fn)
        results[name] = (acc, correct, adj, other)
        print(f"[{name}] exact_match={acc:.4f} ({correct}/{len(val_df)})  adj_swap_fail={adj}  other_fail={other}")

    print(f"\n{'='*60}")
    print("요약")
    print(f"{'='*60}")
    for name, (acc, correct, adj, other) in results.items():
        print(f"  {name:30s}  acc={acc:.4f}  adj={adj:3d}  other={other:3d}")


if __name__ == "__main__":
    main()
