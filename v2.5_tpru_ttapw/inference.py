"""
HNTV v2.5 — TPRU-7B Server Inference (DDP)

Features:
  1. 560px resolution
  2. Batch-24 inference (all 24 perms at once)
  3. TTA: horizontal flip
  4. Reverse score correction (free, uses already-computed scores)
  5. Pairwise adjacent frame scoring (12 pairs batched)
  6. DDP: each GPU handles a shard of 819 test samples
"""

import os
import torch
import torch.distributed as dist
import pandas as pd
from itertools import permutations
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ── 경로 ──────────────────────────────────────────────────────
MODEL_PATH = "/data/gyuyeonlim/models/TPRU-7B"
CKPT_DIR   = "/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/checkpoints/best_tpru"
DATA_DIR   = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
OUT_DIR    = Path("/data/gyuyeonlim/snu_ai_challenge/v2.5_tpru_ttapw")

# ── 설정 ──────────────────────────────────────────────────────
MAX_SIZE    = 560
INFER_BATCH = 24
PAIR_WEIGHT = 0.3

ALL_PERMS   = list(permutations([1, 2, 3, 4]))
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}

PROMPT_4F = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Carefully examine object positions, body postures, and state changes "
    "between consecutive frames to determine temporal direction.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)

PROMPT_2F = (
    "Sentence: {sentence}\n\n"
    "Frame 1 and Frame 2 are shown in this exact order.\n"
    "Does Frame 1 come BEFORE Frame 2 in the actual chronological sequence?\n"
    "Answer only with \"Yes\" or \"No\"."
)

SYSTEM = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
)


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_msgs(images, sentence, prompt_tmpl):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_tmpl.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content}]


def batch_logits(model, processor, yes_id, no_id, device, texts, images_list, batch_size=INFER_BATCH):
    scores = []
    for bi in range(0, len(texts), batch_size):
        sub_t = texts[bi:bi+batch_size]
        sub_i = images_list[bi:bi+batch_size]
        with torch.no_grad():
            inp = processor(text=sub_t, images=sub_i,
                            return_tensors="pt", padding=True).to(device)
            logits = model(**inp).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            scores.extend((lp[:, yes_id] - lp[:, no_id]).cpu().tolist())
    return scores


def score_24perms(model, processor, yes_id, no_id, device, base_imgs, sentence):
    texts, imgs_list = [], []
    for perm in ALL_PERMS:
        inv = [0] * 4
        for k, t in enumerate(perm):
            inv[t-1] = k
        imgs = [base_imgs[inv[t]] for t in range(4)]
        msg  = build_msgs(imgs, sentence, PROMPT_4F)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
    return batch_logits(model, processor, yes_id, no_id, device, texts, imgs_list)


def pairwise_scores(model, processor, yes_id, no_id, device, base_imgs, sentence):
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    texts, imgs_list = [], []
    for (i, j) in pairs:
        imgs = [base_imgs[i], base_imgs[j]]
        msg  = build_msgs(imgs, sentence, PROMPT_2F)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
    raw = batch_logits(model, processor, yes_id, no_id, device, texts, imgs_list, batch_size=12)
    pw = [[0.0]*4 for _ in range(4)]
    for k, (i, j) in enumerate(pairs):
        pw[i][j] = raw[k]
    return pw


def pairwise_bonus_for_perm(perm, pw):
    inv = [0] * 4
    for k, t in enumerate(perm):
        inv[t-1] = k
    return sum(pw[inv[t]][inv[t+1]] for t in range(3))


def main():
    # ── DDP 초기화 ────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}  |  Resolution: {MAX_SIZE}px")
        print(f"PAIR_WEIGHT: {PAIR_WEIGHT}  |  TTA: ON  |  Reverse: ON")

    # ── 모델 로드 ─────────────────────────────────────────────
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(base_model, CKPT_DIR)
    model.eval()

    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id  = processor.tokenizer.convert_tokens_to_ids("No")

    # ── reverse perm 인덱스 ───────────────────────────────────
    rev_idx = [PERM_TO_IDX[tuple(reversed(p))] for p in ALL_PERMS]

    # ── 데이터 샤딩 ───────────────────────────────────────────
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    shard   = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"Total: {len(test_df)} samples → {len(shard)} per GPU")

    # ── 추론 ──────────────────────────────────────────────────
    submission = []
    img_dir_root = DATA_DIR / "test"

    for _, row in tqdm(shard.iterrows(), total=len(shard),
                       desc=f"[rank{rank}]", position=rank):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        img_dir   = img_dir_root / sample_id
        img_files = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])

        try:
            base_imgs = [load_image(str(img_dir / f)) for f in img_files]
        except Exception:
            submission.append({"Id": sample_id, "Answer": str([1, 2, 3, 4])})
            continue

        flip_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in base_imgs]

        # 1. 24 perms × orig + flip
        s_orig = score_24perms(model, processor, yes_id, no_id, device, base_imgs, sentence)
        s_flip = score_24perms(model, processor, yes_id, no_id, device, flip_imgs, sentence)

        # 2. TTA 평균
        s_tta = [(s_orig[i] + s_flip[i]) / 2.0 for i in range(24)]

        # 3. Reverse score correction (공짜)
        s_corr = [s_tta[i] - s_tta[rev_idx[i]] for i in range(24)]

        # 4. Pairwise adjacent bonus
        pw = pairwise_scores(model, processor, yes_id, no_id, device, base_imgs, sentence)

        # 5. Final score
        final = [
            s_corr[i] + PAIR_WEIGHT * pairwise_bonus_for_perm(perm, pw)
            for i, perm in enumerate(ALL_PERMS)
        ]

        best = max(range(24), key=lambda i: final[i])
        submission.append({"Id": sample_id, "Answer": str(list(ALL_PERMS[best]))})

    # ── 부분 결과 저장 ────────────────────────────────────────
    partial_path = OUT_DIR / f"partial_{rank}.csv"
    pd.DataFrame(submission).to_csv(partial_path, index=False)

    if world_size > 1:
        dist.barrier()

    # ── rank 0: 병합 ──────────────────────────────────────────
    if rank == 0:
        dfs = [pd.read_csv(OUT_DIR / f"partial_{i}.csv") for i in range(world_size)]
        final_df = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        out_path = OUT_DIR / "submission_v2.5.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}")
        print(final_df.head())

        # 임시 partial 파일 정리
        for i in range(world_size):
            (OUT_DIR / f"partial_{i}.csv").unlink(missing_ok=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
