"""
[규칙 위반 — 사용 금지]
SNU AI Challenge 규칙 3.3: "생성형 모델(Generative Model)을 이용한 데이터 생성/변형은 허용하지 않음"
이 스크립트는 Qwen2.5-VL (생성형 모델)로 학습 문장을 생성하는 코드로, 대회 규칙을 위반함.
절대 실행하지 말 것. 참고 목적으로만 보존.

로컬 Qwen2.5-VL (TPRU-7B)로 Type B 문장 생성
정답 순서로 정렬한 4장 이미지 → "First... then... after which... finally..." 포맷

torchrun --nproc_per_node=4 augment_sentences_local.py
"""

import os
import ast
import torch
import torch.distributed as dist
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/data/gyuyeonlim/models/TPRU-7B"
DATA_DIR   = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
OUT_DIR    = Path("/data/gyuyeonlim/snu_ai_challenge")

MAX_SIZE = 336  # augmentation에서는 더 작게 (생성만 하면 됨)

PROMPT = (
    "The 4 frames below are shown in CORRECT chronological order.\n"
    "Original description: \"{sentence}\"\n\n"
    "Rewrite as ONE sentence using this exact format:\n"
    "\"First [event1], then [event2], after which [event3], and finally [event4].\"\n"
    "Keep it 20-50 words. Focus on the visible actions or changes between frames."
)

SYSTEM = "You are a helpful assistant that describes video sequences."


def load_image(path: str):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_msg(images, sentence: str):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content}]


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"World size: {world_size}  |  Model: {MODEL_PATH}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    real_df  = train_df[train_df["No_ordering"] == False].reset_index(drop=True)
    shard    = real_df.iloc[rank::world_size].reset_index(drop=True)

    if rank == 0:
        print(f"총 {len(real_df)}개 → GPU당 {len(shard)}개")

    results = []

    for _, row in tqdm(shard.iterrows(), total=len(shard), desc=f"[rank{rank}]", position=rank):
        answer  = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / row["Id"]
        files   = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")

        # 정답 순서로 이미지 정렬 (answer[i] = temporal position of Input_(i+1))
        inv = [0] * 4
        for i, pos in enumerate(answer):
            inv[pos - 1] = i
        imgs = [load_image(str(img_dir / files[inv[t]])) for t in range(4)]

        msg  = build_msg(imgs, row["Sentence"])
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

        try:
            with torch.no_grad():
                inp = processor(text=[text], images=[imgs], return_tensors="pt").to(device)
                out = model.generate(
                    **inp,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            generated = processor.decode(
                out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            generated = None
            if rank == 0:
                print(f"  Error on {row['Id']}: {e}")

        results.append({
            "Id":          row["Id"],
            "Sentence":    row["Sentence"],
            "Augmented":   generated,
            "Answer":      row["Answer"],
            "No_ordering": row["No_ordering"],
        })

    partial = OUT_DIR / f"aug_partial_{rank}.csv"
    pd.DataFrame(results).to_csv(partial, index=False)
    if rank == 0:
        print(f"[rank0] partial 저장: {partial}")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dfs = [pd.read_csv(OUT_DIR / f"aug_partial_{i}.csv") for i in range(world_size)]
        out = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        n_ok = out["Augmented"].notna().sum()
        out.to_csv(OUT_DIR / "augmented_sentences.csv", index=False)
        print(f"\n완료: {n_ok}/{len(out)}개 생성 성공")
        print(out[["Sentence", "Augmented"]].head(3).to_string())
        for i in range(world_size):
            (OUT_DIR / f"aug_partial_{i}.csv").unlink(missing_ok=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
