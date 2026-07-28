"""
v13 Tier-0: n_nouns grounding 프롬프트 A/B 테스트 — 재학습 없음, best_v8 체크포인트 그대로.

배경 (EDA.md §10-2): n_nouns(문장에 언급된 명사 개수)가 val 정확도의 유일한 독립 신호로
6가지 방법(로지스틱 회귀/bootstrap CI/split-half 재현성 등)으로 확정됨(r=0.467). 문장이
명사가 적을수록 텍스트 단서가 부족해 이미지에만 의존해야 하고, 그럴수록 모델이 더 틀린다는
해석이었다. 지금까지(v7/v10) 이 발견은 전부 "loss 재가중치" 축으로만 활용됐고 둘 다 v8 대비
LB가 낮았다 — 이번엔 처음으로 "모델이 이미지에서 스스로 명사(대상)를 더 짚어보게" 유도하는
프롬프트 개입으로 같은 발견을 다른 방식으로 활용한다.

주의 (1단계 한계): 이 스크립트는 model.generate()로 실제 추론 텍스트를 생성시키지 않는다 —
기존 inference.py와 동일하게 전체 프롬프트를 teacher-forcing한 뒤 마지막 위치의 Yes/No
logit만 읽는 단일 forward 방식이다. 즉 "지시문"이 실제로 실행되는 CoT가 아니라, attention을
통해 logit에 영향을 줄 수 있는지만 저비용으로 스크리닝하는 것. 여기서 신호가 보이면 2단계로
실제 model.generate() 기반 CoT(비용이 24-permutation x 476 val이라 훨씬 큼)를 검증할 가치가
있다.

평가: val(476개, v8/v10과 동일 SEED=42 split) 전체 정확도 + n_nouns 구간별(EDA §10-2 버킷과
동일하게 1-4/5-6/7-8/9-10/11-15+) 정확도. "전체는 비슷한데 n_nouns 낮은 구간만 회복"되면 성공.

단일 GPU 실행: python prompt_ab_test.py
"""
import ast
from itertools import permutations

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig

from config import DATA_DIR, MODEL_PATH, CKPT_PATH, VAL_RAW_CSV, MAX_IMAGE_SIZE, INFER_BATCH_SIZE

ALL_PERMS = list(permutations([1, 2, 3, 4]))
device = torch.device("cuda:0")


# ── 이미지 로딩 (v8/dataset.py와 동일 로직) ──────────────────────────
def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _get_model_class(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path)
    mt = getattr(cfg, "model_type", "")
    if mt == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


# ── 프롬프트 변형 ────────────────────────────────────────────────────
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

SYSTEM_ADJSWAP = (
    "You are an expert temporal ordering assistant specialized in human-action videos. "
    "Given 4 frames and a caption, determine whether the frames are arranged in the "
    "chronological order that matches the caption."
)
PROMPT_ADJSWAP = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n\n"
    "Compare each pair of CONSECUTIVE frames carefully. Pay special attention to:\n"
    "- the direction of motion or body-pose progression across frames\n"
    "- whether the middle two frames could be swapped without you noticing "
    "(adjacent-frame swaps are the most visually subtle type of error)\n\n"
    "Is this the correct chronological order of events matching the sentence?\n"
    "Answer only with \"Yes\" or \"No\"."
)

# v13 신규: n_nouns(명사=구체적 시각 앵커) 발견을 직접 겨냥. 문장이 짧고 명사가 적을수록
# "이미지에서 스스로 대상을 짚어보라"는 지시가 상대적으로 더 큰 영향을 줄 것이라는 가설.
SYSTEM_NOUN_GROUNDING = (
    "You are a temporal ordering assistant. Given video frames in a specific order and a "
    "caption, determine if the frames are in the correct chronological order. Ground your "
    "judgment in the concrete people/objects visible in each frame, not only the wording of "
    "the caption."
)
PROMPT_NOUN_GROUNDING = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "For each frame, note the key people/objects and what they are doing, then track how "
    "they change from frame to frame. Use this visual progression together with the caption "
    "to judge order, especially when the caption is short or vague.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)


def _build(system, prompt_tpl, images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_tpl.format(sentence=sentence)})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


VARIANTS = {
    "A_baseline":        lambda imgs, s: _build(SYSTEM_BASELINE,      PROMPT_BASELINE,      imgs, s),
    "B_adjswap_hint":    lambda imgs, s: _build(SYSTEM_ADJSWAP,       PROMPT_ADJSWAP,       imgs, s),
    "D_noun_grounding":  lambda imgs, s: _build(SYSTEM_NOUN_GROUNDING, PROMPT_NOUN_GROUNDING, imgs, s),
}


def n_noun_bucket(n: int) -> str:
    if n <= 4:
        return "1-4"
    if n <= 6:
        return "5-6"
    if n <= 8:
        return "7-8"
    if n <= 10:
        return "9-10"
    return "11-15+"


BUCKET_ORDER = ["1-4", "5-6", "7-8", "9-10", "11-15+"]


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
                with torch.no_grad():
                    logits = model(**inp).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                parts.append((lp[:, yes_id] - lp[:, no_id]).clamp(-100.0, 100.0))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] chunk_size -> {chunk_size}", flush=True)


def eval_variant(model, processor, val_df, yes_id, no_id, build_fn, name):
    per_sample = []  # (n_nouns, correct: bool, fail_type: str|None)

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=name, leave=False):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        gt        = ast.literal_eval(row["Answer"])
        n_nouns   = int(row["n_nouns"])

        img_dir = DATA_DIR / "train" / sample_id
        files   = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        texts, imgs_list = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_fn(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)

        scores = _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, INFER_BATCH_SIZE)
        scored = [(scores[i].item(), list(perm)) for i, perm in enumerate(ALL_PERMS)]
        best = max(scored, key=lambda x: x[0])[1]

        correct = (best == gt)
        fail_type = None
        if not correct:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            fail_type = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
        per_sample.append((n_nouns, correct, fail_type))

    return per_sample


def summarize(name, per_sample):
    n = len(per_sample)
    correct = sum(1 for _, c, _ in per_sample if c)
    adj = sum(1 for _, c, f in per_sample if not c and f == "adj_swap")
    other = sum(1 for _, c, f in per_sample if not c and f == "other")
    print(f"\n[{name}] overall exact_match={correct/n:.4f} ({correct}/{n})  adj_swap_fail={adj}  other_fail={other}")

    print(f"  {'n_nouns bucket':12s}  {'n':>4s}  {'acc':>7s}")
    for b in BUCKET_ORDER:
        sub = [c for nn, c, _ in per_sample if n_noun_bucket(nn) == b]
        if not sub:
            continue
        acc = sum(sub) / len(sub)
        print(f"  {b:12s}  {len(sub):4d}  {acc:6.1%}")


def main():
    print(f"Loading base model + LoRA from {CKPT_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, str(CKPT_PATH)).to(device)
    model.eval()

    tok = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id  = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]

    val_df = pd.read_csv(VAL_RAW_CSV)
    print(f"Val samples: {len(val_df)}  (checkpoint: best_v8, LB 0.89528)")

    all_results = {}
    for name, fn in VARIANTS.items():
        per_sample = eval_variant(model, processor, val_df, yes_id, no_id, fn, name)
        all_results[name] = per_sample
        summarize(name, per_sample)

    print(f"\n{'='*70}")
    print("최종 요약 (overall / n_nouns 하위 구간 1-4,5-6 평균)")
    print(f"{'='*70}")
    for name, per_sample in all_results.items():
        n = len(per_sample)
        acc = sum(1 for _, c, _ in per_sample if c) / n
        low = [c for nn, c, _ in per_sample if n_noun_bucket(nn) in ("1-4", "5-6")]
        low_acc = sum(low) / len(low) if low else float("nan")
        print(f"  {name:20s}  overall={acc:.4f}  low_n_nouns(1-6)={low_acc:.4f}  (n_low={len(low)})")


if __name__ == "__main__":
    main()
