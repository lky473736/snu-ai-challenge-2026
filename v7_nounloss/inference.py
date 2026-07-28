"""
v7 Inference — DDP, 24-permutation exhaustive search (Qwen3-VL-8B base, 448px, AdaptiveDistanceNounLoss)
best_v7 and best_v7_last checkpoints → submission_v7_best.csv / submission_v7_last.csv
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
from transformers import AutoProcessor, AutoConfig

MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"
CKPT_BASE  = Path("/data/gyuyeonlim/snu_ai_challenge/v7_nounloss/checkpoints")
DATA_DIR   = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
OUT_DIR    = Path("/data/gyuyeonlim/snu_ai_challenge/v7_nounloss")

CKPT_LIST = [
    ("best_v7",      "submission_v7_best"),
    ("best_v7_last", "submission_v7_last"),
]

MAX_SIZE    = 448  # train.py MAX_IMAGE_SIZE와 반드시 일치 (v6 때 512로 어긋났던 문제 재발 방지)
INFER_BATCH = 24  # OOM 나면 자동으로 줄어듦 (여유 3.2GB로 빡빡해서 안전장치 필수)

ALL_PERMS = list(permutations([1, 2, 3, 4]))

PROMPT_4F = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Please carefully examine the changes between consecutive frames.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)

SYSTEM = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
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


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_msgs(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_4F.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content}]


def _score_chunk(model, processor, yes_id, no_id, device, texts, imgs_list, chunk_size):
    """OOM 나면 chunk_size 절반으로 줄여 재시도"""
    while True:
        try:
            out = []
            for bs in range(0, len(texts), chunk_size):
                inp = processor(text=texts[bs: bs + chunk_size], images=imgs_list[bs: bs + chunk_size],
                                 return_tensors="pt", padding=True).to(device)
                logits = model(**inp).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                out.extend((lp[:, yes_id] - lp[:, no_id]).cpu().tolist())
            return out
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] infer chunk_size -> {chunk_size}", flush=True)


def score_24perms(model, processor, yes_id, no_id, device, base_imgs, sentence):
    texts, imgs_list = [], []
    for perm in ALL_PERMS:
        inv = [0] * 4
        for k, t in enumerate(perm):
            inv[t - 1] = k
        imgs = [base_imgs[inv[t]] for t in range(4)]
        msg  = build_msgs(imgs, sentence)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
    with torch.no_grad():
        scores = _score_chunk(model, processor, yes_id, no_id, device, texts, imgs_list, INFER_BATCH)
    return list(zip(scores, [list(p) for p in ALL_PERMS]))


def run_inference(model, processor, yes_id, no_id, device, shard, rank, world_size, out_name):
    submission   = []
    img_dir_root = DATA_DIR / "test"

    for _, row in tqdm(shard.iterrows(), total=len(shard),
                       desc=f"[{out_name}][rank{rank}]", position=rank):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        img_dir   = img_dir_root / sample_id
        img_files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")

        try:
            base_imgs = [load_image(str(img_dir / f)) for f in img_files]
        except Exception:
            submission.append({"Id": sample_id, "Answer": str([1, 2, 3, 4])})
            continue

        scores = score_24perms(model, processor, yes_id, no_id, device, base_imgs, sentence)
        best   = max(scores, key=lambda x: x[0])[1]
        submission.append({"Id": sample_id, "Answer": str(best)})

    partial_path = OUT_DIR / f"partial_{rank}_{out_name}.csv"
    pd.DataFrame(submission).to_csv(partial_path, index=False)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dfs      = [pd.read_csv(OUT_DIR / f"partial_{i}_{out_name}.csv") for i in range(world_size)]
        final_df = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        out_path = OUT_DIR / f"{out_name}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}  ({len(final_df)} rows)")
        for i in range(world_size):
            (OUT_DIR / f"partial_{i}_{out_name}.csv").unlink(missing_ok=True)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    )

    tok    = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id  = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    shard   = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"World size: {world_size}  |  Resolution: {MAX_SIZE}px")
        print(f"Total: {len(test_df)} samples → {len(shard)} per GPU")

    for ckpt_name, out_name in CKPT_LIST:
        ckpt_path = CKPT_BASE / ckpt_name
        if not ckpt_path.exists():
            if rank == 0:
                print(f"[skip] {ckpt_path} not found")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Checkpoint: {ckpt_path}")

        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        run_inference(model, processor, yes_id, no_id, device, shard, rank, world_size, out_name)

        del model
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
