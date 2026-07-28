"""
v20 Inference — Qwen3-VL-32B QLoRA(4bit), DDP, 24-permutation 전수조사 (v14와 동일 스코어링).
best_v20 체크포인트 하나만 사용(탐색 실행이라 last는 생략 — 필요하면 CKPT_LIST에 추가).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
from itertools import permutations
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from accelerate import dispatch_model
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig

from config import (
    DATA_DIR, CKPT_DIR, MODEL_PATH,
    BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT, LLM_INT8_SKIP_MODULES,
    INFER_BATCH_SIZE,
)
from src.dataset import load_image, build_messages
from src.model import get_yes_no_token_ids, forward_logit

OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ALL_PERMS = list(permutations([1, 2, 3, 4]))

CKPT_LIST = [
    ("best_v20", "submission_v20_best"),
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


def reorder(order, base_imgs):
    inv = [0] * 4
    for inp_idx, t_pos in enumerate(order):
        inv[t_pos - 1] = inp_idx
    return [base_imgs[inv[t]] for t in range(4)]


def _chunked_forward(model, processor, texts, imgs_list, yes_id, no_id, device, size_holder):
    while True:
        chunk_size = size_holder[0]
        try:
            parts = []
            with torch.no_grad():
                for bi in range(0, len(texts), chunk_size):
                    inp = processor(text=texts[bi:bi + chunk_size], images=imgs_list[bi:bi + chunk_size],
                                     return_tensors="pt", padding=True).to(device)
                    parts.append(forward_logit(model, processor, inp, yes_id, no_id))
            return torch.cat(parts)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            size_holder[0] = max(1, chunk_size // 2)
            print(f"[OOM] infer chunk_size -> {size_holder[0]}", flush=True)


def run_inference(model, processor, device, shard, rank, world_size, out_name, yes_id, no_id, size_holder):
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

        texts, imgs_list = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm), base_imgs)
            msg = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)

        scores = _chunked_forward(model, processor, texts, imgs_list, yes_id, no_id, device, size_holder)
        best = max(zip(scores.tolist(), [list(p) for p in ALL_PERMS]), key=lambda x: x[0])[1]
        submission.append({"Id": sample_id, "Answer": str(best)})

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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        llm_int8_skip_modules=LLM_INT8_SKIP_MODULES,
    )
    if world_size == 1:
        offload_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
            llm_int8_skip_modules=LLM_INT8_SKIP_MODULES,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        device_map = {"model.language_model": 0, "model.visual": "cpu", "lm_head": "cpu"}
        base_model = ModelClass.from_pretrained(
            MODEL_PATH, quantization_config=offload_bnb_config, torch_dtype=torch.bfloat16, device_map=device_map,
        )
        print("hf_device_map:", getattr(base_model, "hf_device_map", None))
        print(f"GPU 메모리(로드 직후): {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("visual dtype:", next(base_model.model.visual.parameters()).dtype)
        print("lm_head dtype:", next(base_model.lm_head.parameters()).dtype)
    else:
        base_model = ModelClass.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={"": device},
        )
    torch.cuda.empty_cache()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    shard = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Total: {len(test_df)} samples -> {len(shard)} per GPU")

    yes_id, no_id = get_yes_no_token_ids(processor)
    size_holder = [INFER_BATCH_SIZE]

    for ckpt_name, out_name in CKPT_LIST:
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            if rank == 0:
                print(f"[skip] {ckpt_path} not found")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Checkpoint: {ckpt_path}")

        if rank == 0:
            torch.cuda.reset_peak_memory_stats()
            print(f"GPU 메모리(어댑터 로드 직전): {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
            for name, p in base_model.named_parameters():
                if p.device.type == "cuda" and "language_model" not in name:
                    print(f"  [의외로 GPU에 있음] {name}: {p.device}, {p.dtype}, {p.numel()}", flush=True)

        try:
            if world_size == 1:
                model = PeftModel.from_pretrained(
                    base_model, str(ckpt_path), low_cpu_mem_usage=True,
                )
            else:
                model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
            if rank == 0:
                print(f"GPU 메모리(OOM 시점 peak): {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
                for name, p in base_model.named_parameters():
                    if p.device.type == "cuda" and "language_model" not in name:
                        print(f"  [OOM 시점, GPU로 이동됨] {name}: {p.device}, {p.dtype}", flush=True)
            raise
        model.eval()

        if world_size == 1:
            dispatch_model(model, device_map={
                "": 0,
                "base_model.model.model.visual": "cpu",
                "base_model.model.lm_head": "cpu",
            })
            if rank == 0:
                for name, p in model.named_parameters():
                    if "visual" not in name and "lm_head" not in name and p.device.type != "cuda":
                        print(f"[여전히 CPU] {name}: {p.device}", flush=True)

        run_inference(model, processor, device, shard, rank, world_size, out_name, yes_id, no_id, size_holder)

        del model
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
