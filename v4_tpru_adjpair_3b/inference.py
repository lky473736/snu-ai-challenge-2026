"""
HNTV v4 Inference (3B port) — 24-permutation exhaustive search
Single GPU (Colab):  python inference.py --data_dir ./data/snuaichallenge_data
Multi GPU:           torchrun --nproc_per_node=N inference.py --data_dir ...
"""

import os
import argparse
import torch
import torch.distributed as dist
import pandas as pd
from itertools import permutations
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MAX_SIZE    = 448
INFER_BATCH = 24

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


def score_24perms(model, processor, yes_id, no_id, device, base_imgs, sentence):
    results = []
    for bs in range(0, len(ALL_PERMS), INFER_BATCH):
        batch_perms = ALL_PERMS[bs: bs + INFER_BATCH]
        texts, imgs_list = [], []
        for perm in batch_perms:
            inv = [0] * 4
            for k, t in enumerate(perm):
                inv[t - 1] = k
            imgs = [base_imgs[inv[t]] for t in range(4)]
            msg  = build_msgs(imgs, sentence)
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)
        with torch.no_grad():
            inp    = processor(text=texts, images=imgs_list,
                               return_tensors="pt", padding=True).to(device)
            logits = model(**inp).logits[:, -1, :].float()
            lp     = torch.log_softmax(logits, dim=-1)
            scores = (lp[:, yes_id] - lp[:, no_id]).cpu().tolist()
        for i, perm in enumerate(batch_perms):
            results.append((scores[i], list(perm)))
    return results


def run_inference(model, processor, yes_id, no_id, device, shard, rank, world_size,
                   data_dir: Path, out_dir: Path, out_name: str):
    submission   = []
    img_dir_root = data_dir / "test"

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

    partial_path = out_dir / f"partial_{rank}_{out_name}.csv"
    pd.DataFrame(submission).to_csv(partial_path, index=False)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dfs      = [pd.read_csv(out_dir / f"partial_{i}_{out_name}.csv") for i in range(world_size)]
        final_df = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        out_path = out_dir / f"{out_name}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}  ({len(final_df)} rows)")
        for i in range(world_size):
            (out_dir / f"partial_{i}_{out_name}.csv").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=os.environ.get("HNTV_DATA_DIR", "./data/snuaichallenge_data"),
                         help="snuaichallenge_data 폴더 (test.csv, test/ 포함)")
    parser.add_argument("--model_path", type=str, default=os.environ.get("HNTV_MODEL_PATH", "Stephengzk/TPRU-3B"),
                         help="로컬 경로 또는 HuggingFace repo id")
    parser.add_argument("--ckpt_dir",   type=str, default="./checkpoints",
                         help="LoRA 체크포인트들이 들어있는 상위 폴더")
    parser.add_argument("--ckpt_names", type=str, default="best_v4_3b,best_v4_3b_last",
                         help="ckpt_dir 아래 서브폴더명, 콤마로 여러 개 지정 시 각각 제출파일 생성")
    parser.add_argument("--out_dir",    type=str, default=".",
                         help="submission csv 저장 위치")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    ckpt_base = Path(args.ckpt_dir)
    out_dir   = Path(args.out_dir)
    ckpt_list = [(name, f"submission_{name}") for name in args.ckpt_names.split(",")]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"Model: {args.model_path}  |  Data: {data_dir}  |  Resolution: {MAX_SIZE}px")

    processor  = AutoProcessor.from_pretrained(args.model_path)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": device},
    )

    tok    = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id  = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]

    test_df = pd.read_csv(data_dir / "test.csv")
    shard   = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"World size: {world_size}  |  Total: {len(test_df)} samples → {len(shard)} per GPU")

    for ckpt_name, out_name in ckpt_list:
        ckpt_path = ckpt_base / ckpt_name
        if not ckpt_path.exists():
            if rank == 0:
                print(f"[skip] {ckpt_path} not found")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Checkpoint: {ckpt_path}")

        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        run_inference(model, processor, yes_id, no_id, device, shard, rank, world_size,
                      data_dir, out_dir, out_name)

        del model
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
