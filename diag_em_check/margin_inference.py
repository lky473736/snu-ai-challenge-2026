"""
EM vs 부분점수(쌍순서) 채점 판별용 마진 계산 — DDP, 4-GPU.

myunhh 팀(Ver8/scripts/diag_swap_submission.py)의 방법론을 그대로 가져오되, 코드는 베끼지 않고
우리 v14 컨벤션(1-based Answer 포맷, reorder 기반 24-permutation 전수조사)으로 새로 구현.

원리: 재추론 없이 기존 제출 파일만 건드리는 게 아니라, 여기서는 마진(1등-2등 점수차, =확신도)이
아직 저장돼 있지 않아서 test 819개에 대해 best_v14로 24-permutation 스코어링을 다시 돌려
(id, answer, margin)만 뽑는다. inference.py와 동일한 스코어링 로직 재사용 — 예측 자체는
기존 submission_v14_best.csv와 100% 동일해야 함(같은 체크포인트, 같은 로직).
"""

import os
import ast
from itertools import permutations
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig

MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"
CKPT_PATH = Path("/data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax/checkpoints/best_v14")
DATA_DIR = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

MAX_SIZE = 448
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


def build_messages(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_4F.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def reorder(order, base_imgs):
    inv = [0] * 4
    for inp_idx, t_pos in enumerate(order):
        inv[t_pos - 1] = inp_idx
    return [base_imgs[inv[t]] for t in range(4)]


def get_yes_no_token_ids(processor):
    tok = processor.tokenizer
    return tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1], tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]


def forward_logit(model, inputs, yes_id, no_id):
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id] - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)


def _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, device, chunk_size):
    while True:
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(text=texts[bi:bi + chunk_size], images=imgs_list[bi:bi + chunk_size],
                                 return_tensors="pt", padding=True).to(device)
                parts.append(forward_logit(model, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)


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
    model = PeftModel.from_pretrained(base_model, str(CKPT_PATH)).eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    shard = test_df.iloc[rank::world_size].reset_index(drop=True)
    if rank == 0:
        print(f"World size: {world_size}  총 {len(test_df)}건 -> GPU당 {len(shard)}건", flush=True)

    rows = []
    for _, row in tqdm(shard.iterrows(), total=len(shard), desc=f"rank{rank}", position=rank):
        sample_id, sentence = row["Id"], row["Sentence"]
        img_dir = DATA_DIR / "test" / sample_id
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        texts, imgs_list = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm), base_imgs)
            msg = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            imgs_list.append(imgs)

        scores = _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, device, INFER_BATCH)
        ranked = sorted(
            [(scores[i].item(), list(ALL_PERMS[i])) for i in range(len(ALL_PERMS))],
            key=lambda x: -x[0],
        )
        top1_score, top1_perm = ranked[0]
        top2_score, _ = ranked[1]
        margin = top1_score - top2_score
        rows.append({"Id": sample_id, "Answer": str(top1_perm), "margin": margin})

    partial_path = OUT_DIR / f"partial_margins_{rank}.csv"
    pd.DataFrame(rows).to_csv(partial_path, index=False)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        dfs = [pd.read_csv(OUT_DIR / f"partial_margins_{i}.csv") for i in range(world_size)]
        final_df = pd.concat(dfs).sort_values("Id").reset_index(drop=True)
        out_path = OUT_DIR / "margins.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\n저장: {out_path}  ({len(final_df)}행)")
        for i in range(world_size):
            (OUT_DIR / f"partial_margins_{i}.csv").unlink(missing_ok=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
