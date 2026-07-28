"""
본선 진출 후보자 대상 — 외부 데이터셋(snuaichallenge_test_data) 추론.
추가 학습 금지 규정에 따라, 예선 최종 제출(submission_v20_best.csv, public 0.91099 /
private 0.90650)을 만든 것과 100% 동일한 체크포인트(best_v20)·동일 로직(4bit QLoRA,
24-permutation 전수조사)만 그대로 재사용한다. 재학습 없음, 코드 변경은 데이터 경로/포맷
어댑터뿐(Input_1~4를 test.csv에서 직접 읽어 파일시스템 정렬 의존을 제거 — 재현성 강화).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import permutations
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig

from config import (
    CKPT_DIR, MODEL_PATH,
    BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT, LLM_INT8_SKIP_MODULES,
    INFER_BATCH_SIZE,
)
from src.dataset import load_image, build_messages
from src.model import get_yes_no_token_ids, forward_logit

OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ALL_PERMS = list(permutations([1, 2, 3, 4]))

# 예선 최종 제출과 동일한 체크포인트만 사용(추가 학습 금지)
CKPT_NAME = "best_v20"
OUT_NAME = "submission_semifinal_v20"

EXT_DATA_DIR = Path("/data/gyuyeonlim/snu_ai_challenge/semifinal_external_test/snuaichallenge_test_data")
EXT_TEST_CSV = EXT_DATA_DIR / "test.csv"
EXT_IMG_ROOT = EXT_DATA_DIR / "test"


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
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            size_holder[0] = max(1, chunk_size // 2)
            print(f"[OOM] infer chunk_size -> {size_holder[0]}", flush=True)


def run_inference(model, processor, device, shard, rank, world_size, out_name, yes_id, no_id, size_holder):
    submission = []

    for _, row in tqdm(shard.iterrows(), total=len(shard), desc=f"[{out_name}][rank{rank}]", position=rank):
        sample_id = row["Id"]
        sentence = row["Sentence"]
        img_dir = EXT_IMG_ROOT / sample_id
        # 재현성 강화: 파일시스템 정렬(sorted) 대신 test.csv의 Input_1~4를 그대로 사용
        img_files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]

        try:
            base_imgs = [load_image(str(img_dir / f)) for f in img_files]
        except Exception as e:
            print(f"[warn] {sample_id} 이미지 로드 실패({e}) -> 기본값 [1,2,3,4]", flush=True)
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
        # 단일 GPU(예: RTX 3090 24GB)에서 베이스 모델(4bit) + LoRA 어댑터가 다 안 들어갈 수 있어,
        # 일부 레이어를 CPU RAM으로 오프로드한다. 연산 자체(양자화 방식, 정밀도)는 동일하게
        # 유지되므로 예측 결과에는 영향이 없다 — forward pass 때 CPU에 있는 레이어만 그때그때
        # GPU로 옮겨 계산하는 방식이라 속도만 느려진다.
        base_model = ModelClass.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory={0: "20GiB", "cpu": "200GiB"},
        )
    else:
        base_model = ModelClass.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={"": device},
        )
    torch.cuda.empty_cache()

    test_df = pd.read_csv(EXT_TEST_CSV)
    shard = test_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"World size: {world_size}  (외부 데이터셋, 추가 학습 없음, {CKPT_NAME} 그대로 사용)")
        print(f"Total: {len(test_df)} samples -> {len(shard)} per GPU")

    yes_id, no_id = get_yes_no_token_ids(processor)
    size_holder = [INFER_BATCH_SIZE]

    ckpt_path = CKPT_DIR / CKPT_NAME
    if not ckpt_path.exists():
        raise RuntimeError(f"{ckpt_path} 없음 — 예선 최종 체크포인트가 맞는지 확인 필요")

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Checkpoint: {ckpt_path}")

    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    run_inference(model, processor, device, shard, rank, world_size, OUT_NAME, yes_id, no_id, size_holder)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
