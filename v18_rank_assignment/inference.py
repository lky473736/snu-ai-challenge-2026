"""
v18 Inference (Pairwise Bradley-Terry) — DDP, 샘플당 6-forward + 닫힌 형태 집계 (Qwen3-VL-8B base, 448px)
best_v18 and best_v18_last checkpoints → submission_v18_best.csv / submission_v18_last.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig

from config import DATA_DIR, CKPT_DIR, MODEL_PATH
from src.dataset import load_image
from src.aggregate import predict_permutation

OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

CKPT_LIST = [
    ("best_v18", "submission_v18_best"),
    ("best_v18_last", "submission_v18_last"),
]


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


def run_inference(model, processor, device, shard, rank, world_size, out_name):
    submission = []
    img_dir_root = DATA_DIR / "test"

    for _, row in tqdm(shard.iterrows(), total=len(shard), desc=f"[{out_name}][rank{rank}]", position=rank):
        sample_id = row["Id"]
        sentence = row["Sentence"]
        img_dir = img_dir_root / sample_id
        img_files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")

        try:
            base_imgs = [load_image(str(img_dir / f)) for f in img_files]
        except Exception:
            submission.append({"Id": sample_id, "Answer": str([1, 2, 3, 4])})
            continue

        pred = predict_permutation(model, processor, base_imgs, sentence, device)
        submission.append({"Id": sample_id, "Answer": str(pred)})

    partial_path = OUT_DIR / f"partial_{rank}_{out_name}.csv"
    pd.DataFrame(submission).to_csv(partial_path, index=False)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dfs = [pd.read_csv(OUT_DIR / f"partial_{i}_{out_name}.csv") for i in range(world_size)]
        final_df = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        out_path = OUT_DIR / f"{out_name}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}  ({len(final_df)} rows)")
        for i in range(world_size):
            (OUT_DIR / f"partial_{i}_{out_name}.csv").unlink(missing_ok=True)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device})

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    shard = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Total: {len(test_df)} samples → {len(shard)} per GPU")

    for ckpt_name, out_name in CKPT_LIST:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            if rank == 0:
                print(f"[skip] {ckpt_path} not found")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Checkpoint: {ckpt_path}")

        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model.eval()

        run_inference(model, processor, device, shard, rank, world_size, out_name)

        del model
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
