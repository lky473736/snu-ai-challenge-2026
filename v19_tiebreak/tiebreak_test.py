"""
v19 — d=1 인접 스왑 타이브레이크 (재학습 없음, best_v14 zero-shot 테스트)

배경(idea.md 5-1, 11절): d=1(인접 프레임 swap) 오답은 이 프로젝트 전체에서 가장 완고하게
안 풀리는 약점이다. 인접 시점 프레임은 시각적으로 실제로 더 비슷함(Cohen's d=0.349,
해상도 무관)이 확인됐는데, v14의 24-permutation 전수조사는 최고 점수 후보만 argmax로
뽑고 "1등과 2등이 얼마나 근접했는지, 그 차이가 정확히 어떤 인접쌍 때문인지"는 버린다.

이 스크립트는:
  1) best_v14로 val 476개에 대해 기존과 동일한 24-permutation 전수조사(baseline) 수행
  2) 1등·2등 후보의 Kendall distance가 정확히 1(인접 swap 관계)인 경우만 골라서,
     그 두 후보가 갈리는 정확히 그 인접 프레임 쌍 2장 + 픽셀 차이(diff) 이미지를 추가 입력으로
     "Frame A가 Frame B보다 먼저냐"를 다시 물어 override 여부를 결정
  3) baseline vs tie-break exact_match를 직접 비교

재학습 없음 — best_v14 그대로 씀. 프롬프트는 기존 스타일(Yes/No만) 유지, 표현력은
diff 이미지(새로운 시각 정보)로만 늘림 — v11/v12(용량 늘리기, 실패)와 v13(지시문 정교화, 실패)의
함정을 피하기 위한 설계 (세션 논의 참고).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
import time
from itertools import permutations

import pandas as pd
import torch
from PIL import Image, ImageChops, ImageOps
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig

from config import DATA_DIR, MODEL_PATH, V14_CKPT, V14_VAL_CSV, MAX_IMAGE_SIZE, INFER_BATCH_SIZE, LOG_DIR

ALL_PERMS = list(permutations([1, 2, 3, 4]))

PROMPT_4F = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Please carefully examine the changes between consecutive frames.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)
SYSTEM_4F = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
)

# 타이브레이크 전용 프롬프트 — 기존과 동일하게 Yes/No만, 지시문 정교화 없음(v13 교훈).
# 표현력은 프롬프트 문장이 아니라 diff 이미지(새 시각 정보)로만 늘린다.
PROMPT_TB = (
    "Sentence: {sentence}\n\n"
    "Frame A and Frame B show two moments from this event. The third image highlights the "
    "pixel-level difference between Frame A and Frame B.\n"
    "Does the event shown in Frame A happen before the event shown in Frame B?\n"
    "Answer only with \"Yes\" or \"No\"."
)
SYSTEM_TB = (
    "You are a temporal ordering assistant. "
    "Given two video frames, their pixel difference, and a caption, "
    "determine which frame happens first."
)


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


def load_model_and_processor(device):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            print(f"Loading model ({ModelClass.__name__}, bf16, attn={attn_impl})...")
            base_model = ModelClass.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            print(f"  -> {attn_impl} skipped: {e}")
    if base_model is None:
        raise RuntimeError("No attn_implementation worked")

    print(f"Loading v14 best checkpoint: {V14_CKPT}")
    model = PeftModel.from_pretrained(base_model, str(V14_CKPT))
    model.to(device)
    model.eval()
    return model, processor


def get_yes_no_token_ids(processor):
    tok = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]
    return yes_id, no_id


def forward_logit(model, inputs: dict, yes_id: int, no_id: int) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id] - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)


def _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, device, chunk_size):
    """OOM 나면 chunk_size 절반으로 줄여 재시도 (v14와 동일 안전장치)."""
    while True:
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(
                    text=texts[bi: bi + chunk_size],
                    images=imgs_list[bi: bi + chunk_size],
                    return_tensors="pt", padding=True,
                ).to(device)
                parts.append(forward_logit(model, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] chunk_size -> {chunk_size}", flush=True)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_messages_4f(images, sentence: str):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_4F.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM_4F}, {"role": "user", "content": content}]


def make_diff_image(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """픽셀 차이 이미지 — 크기 다르면 B를 A에 맞춰 리사이즈, autocontrast로 대비 강화."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    diff = ImageChops.difference(img_a.convert("RGB"), img_b.convert("RGB"))
    diff = ImageOps.autocontrast(diff, cutoff=1)
    return diff


def build_messages_tb(img_a, img_b, diff_img, sentence: str):
    content = [
        {"type": "text", "text": "Frame A:"}, {"type": "image", "image": img_a},
        {"type": "text", "text": "Frame B:"}, {"type": "image", "image": img_b},
        {"type": "text", "text": "Difference:"}, {"type": "image", "image": diff_img},
        {"type": "text", "text": PROMPT_TB.format(sentence=sentence)},
    ]
    return [{"role": "system", "content": SYSTEM_TB}, {"role": "user", "content": content}]


def reorder(order, base_imgs):
    """order(Answer 포맷: order[k-1]=Input_k의 정답위치) -> 시간순 이미지 리스트."""
    inv = [0] * 4
    for inp_idx, t_pos in enumerate(order):
        inv[t_pos - 1] = inp_idx
    return inv, [base_imgs[inv[t]] for t in range(4)]


def kendall_dist(p, q):
    rank = {v: i for i, v in enumerate(q)}
    arr = [rank[v] for v in p]
    inv = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def score_all_perms(model, processor, yes_id, no_id, device, base_imgs, sentence):
    all_texts, all_imgs, all_invs = [], [], []
    for perm in ALL_PERMS:
        inv, imgs = reorder(list(perm), base_imgs)
        msg = build_messages_4f(imgs, sentence)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)
        all_imgs.append(imgs)
        all_invs.append(inv)
    scores = _forward_chunk(model, processor, all_texts, all_imgs, yes_id, no_id, device, INFER_BATCH_SIZE)
    ranked = sorted(
        [(scores[i].item(), list(ALL_PERMS[i]), all_invs[i]) for i in range(len(ALL_PERMS))],
        key=lambda x: -x[0],
    )
    return ranked  # [(score, perm, inv), ...] 내림차순


def find_disputed_pair(inv1, inv2):
    """inv1(top1 display order) vs inv2(top2 display order) — 정확히 인접한 두 슬롯이 다르면
    그 두 슬롯의 (물리적 Input idx, 슬롯 위치) 반환. 아니면 None."""
    diff_positions = [t for t in range(4) if inv1[t] != inv2[t]]
    if len(diff_positions) != 2:
        return None
    a, b = diff_positions
    if abs(a - b) != 1:
        return None
    if inv1[a] != inv2[b] or inv1[b] != inv2[a]:
        return None  # 진짜 transposition이 아님
    pos_early, pos_late = (a, b) if a < b else (b, a)
    return inv1[pos_early], inv1[pos_late]  # top1 기준 (더 이른 슬롯의 Input idx, 더 늦은 슬롯의 Input idx)


def main():
    device = torch.device("cuda:0")
    model, processor = load_model_and_processor(device)
    yes_id, no_id = get_yes_no_token_ids(processor)

    # v14 학습 때와 동일한 val 476개(No_ordering 포함) 그대로 사용 — v14 공식 val=0.5987(epoch5)과
    # baseline 수치가 직접 비교되도록 필터링하지 않음.
    val_df = pd.read_csv(V14_VAL_CSV)
    print(f"val: {len(val_df)}개")

    baseline_correct = 0
    tiebreak_correct = 0
    triggered = 0
    overridden = 0
    override_helped = 0
    override_hurt = 0
    rows = []

    t0 = time.time()
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="val"):
        sample_id = row["Id"]
        sentence = row["Sentence"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sample_id
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        ranked = score_all_perms(model, processor, yes_id, no_id, device, base_imgs, sentence)
        top1_score, top1_perm, top1_inv = ranked[0]
        top2_score, top2_perm, top2_inv = ranked[1]

        base_ok = (top1_perm == gt)
        baseline_correct += int(base_ok)

        final_perm = top1_perm
        d = kendall_dist(top1_perm, top2_perm)
        trig = False
        override = False

        if d == 1:
            disputed = find_disputed_pair(top1_inv, top2_inv)
            if disputed is not None:
                trig = True
                triggered += 1
                early_idx, late_idx = disputed  # top1 주장: early_idx가 late_idx보다 먼저
                img_a, img_b = base_imgs[early_idx], base_imgs[late_idx]
                diff_img = make_diff_image(img_a, img_b)
                msg = build_messages_tb(img_a, img_b, diff_img, sentence)
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                tb_score = _forward_chunk(
                    model, processor, [text], [[img_a, img_b, diff_img]], yes_id, no_id, device, 1
                )[0].item()
                # tb_score > 0 -> "A가 B보다 먼저" -> top1과 일치 -> 유지
                # tb_score <= 0 -> "B가 A보다 먼저" -> top2와 일치 -> override
                if tb_score <= 0:
                    override = True
                    overridden += 1
                    final_perm = top2_perm

        final_ok = (final_perm == gt)
        tiebreak_correct += int(final_ok)

        if override:
            if final_ok and not base_ok:
                override_helped += 1
            elif base_ok and not final_ok:
                override_hurt += 1

        rows.append({
            "id": sample_id, "gt": gt, "top1": top1_perm, "top2": top2_perm,
            "base_ok": base_ok, "triggered": trig, "override": override, "final_ok": final_ok,
        })

    n = len(val_df)
    elapsed = (time.time() - t0) / 60
    print(f"\n소요: {elapsed:.1f}분")
    print(f"\n[Baseline]  exact_match={baseline_correct/n:.4f} ({baseline_correct}/{n})")
    print(f"[Tiebreak]  exact_match={tiebreak_correct/n:.4f} ({tiebreak_correct}/{n})")
    print(f"\n트리거(1등-2등 d=1): {triggered}/{n} ({triggered/n:.1%})")
    print(f"오버라이드 발생: {overridden}/{triggered if triggered else 1}")
    print(f"  override로 오답->정답: {override_helped}")
    print(f"  override로 정답->오답: {override_hurt}")
    print(f"  순증감: {override_helped - override_hurt:+d}")

    out_path = LOG_DIR / "tiebreak_detail.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n상세 로그: {out_path}")


if __name__ == "__main__":
    main()
