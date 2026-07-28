"""
v21 — LoRA rank(r) 한계 탐색. Qwen3-VL-32B QLoRA(4bit), v14/v20과 동일 레시피(hard negative
K=7, ListwiseSoftmaxLoss, group_size=8, TRAIN_MINIBATCH=8), --lora_r만 바꿔가며 실제 학습
루프(optimizer.step() 2사이클, §5-11 교훈)를 4-GPU DDP로 검증한다.

idea.md §5-11의 "같은 프로세스/객체를 재사용해서 연속 테스트하면 상태 오염 위험" 교훈에 따라,
이 스크립트는 rank 1개만 검증하고 끝난다(exit code로 성공/OOM 판정) — 여러 rank를 이어서
테스트할 땐 반드시 별도 프로세스(accelerate launch를 매번 새로)로 호출할 것 (run_rank_sweep.sh 참고).

exit code: 0=성공, 1=OOM, 2=기타 에러
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
import argparse
import time
from datetime import timedelta
from itertools import permutations
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

DATA_DIR = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-32B-Instruct"

MAX_IMAGE_SIZE = 448
SEED = 42
LORA_ALPHA_RATIO = 2.0  # v14/v20과 동일 비율(alpha = r * 2.0) 유지
LORA_DROPOUT = 0.05
LR = 5e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8
TRAIN_MINIBATCH = 8
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

ALL_PERMS = list(permutations([1, 2, 3, 4]))

PROMPT = (
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


def kendall_dist(p, q):
    rank = {v: i for i, v in enumerate(q)}
    arr = [rank[v] for v in p]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def sample_group(gt):
    by_dist = {}
    for p in ALL_PERMS:
        if p == gt:
            continue
        d = kendall_dist(p, gt)
        by_dist.setdefault(d, []).append(p)
    import random
    samples = [(list(gt), 0)]
    for dist, n_take in SAMPLE_COUNTS.items():
        pool = by_dist.get(dist, [])
        chosen = random.sample(pool, min(n_take, len(pool)))
        for p in chosen:
            samples.append((list(p), dist))
    return samples


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_messages(images, sentence):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


class GroupedTemporalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["Id"]
        gt = tuple(ast.literal_eval(row["Answer"]))
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        images_list, sentences, dists = [], [], []
        for perm, dist in sample_group(gt):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(perm):
                inv[t_pos - 1] = inp_idx
            imgs = [base_imgs[inv[t]] for t in range(4)]
            images_list.append(imgs)
            sentences.append(row["Sentence"])
            dists.append(int(dist))
        return {"images": images_list, "sentences": sentences, "dists": dists, "group_size": len(images_list)}


def collate_fn(batch):
    return {
        "images": [b["images"] for b in batch],
        "sentences": [b["sentences"] for b in batch],
        "dists": [b["dists"] for b in batch],
        "group_sizes": [b["group_size"] for b in batch],
    }


class ListwiseSoftmaxLoss:
    def __call__(self, logits, dists, group_sizes):
        offset = 0
        losses = []
        for gs in group_sizes:
            g_logits = logits[offset: offset + gs]
            g_dists = dists[offset: offset + gs]
            offset += gs
            pos_idxs = [i for i, d in enumerate(g_dists) if d == 0]
            if not pos_idxs:
                continue
            log_probs = torch.log_softmax(g_logits, dim=0)
            losses.append(-log_probs[pos_idxs[0]])
        return torch.stack(losses).mean()


def get_model_class(model_path):
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


def load_model_and_processor(lora_r: int):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = get_model_class(MODEL_PATH)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual"],
    )
    model = ModelClass.from_pretrained(MODEL_PATH, quantization_config=bnb_config,
                                        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_alpha = int(round(lora_r * LORA_ALPHA_RATIO))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, processor


def get_yes_no_token_ids(processor):
    tok = processor.tokenizer
    return tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1], tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]


def forward_logit(model, inputs, yes_id, no_id):
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id] - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_r", type=int, required=True)
    args = ap.parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    torch.manual_seed(SEED)

    lora_alpha = int(round(args.lora_r * LORA_ALPHA_RATIO))
    if is_main:
        print(f"[v21 rank-sweep] r={args.lora_r}  alpha={lora_alpha}  processes={accelerator.num_processes}", flush=True)

    try:
        model, processor = load_model_and_processor(args.lora_r)
        yes_id, no_id = get_yes_no_token_ids(processor)
        criterion = ListwiseSoftmaxLoss()
        if is_main:
            model.print_trainable_parameters()

        n_groups_needed = GRAD_ACCUM * 2 * accelerator.num_processes
        train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(frac=1, random_state=SEED).reset_index(drop=True)
        smoke_df = train_csv.iloc[:n_groups_needed].reset_index(drop=True)
        ds = GroupedTemporalDataset(smoke_df)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        model.train()

        torch.cuda.reset_peak_memory_stats()
        opt_steps = 0

        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)
                texts, imgs_list, group_offsets = [], [], []
                for grp_imgs, grp_sents, grp_dists in zip(batch["images"], batch["sentences"], batch["dists"]):
                    start = len(texts)
                    for imgs, sent in zip(grp_imgs, grp_sents):
                        msg = build_messages(imgs, sent)
                        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                        texts.append(text)
                        imgs_list.append(imgs)
                    group_offsets.append((start, len(grp_imgs), grp_dists))

                parts = []
                for bi in range(0, len(texts), TRAIN_MINIBATCH):
                    inp = processor(text=texts[bi:bi + TRAIN_MINIBATCH], images=imgs_list[bi:bi + TRAIN_MINIBATCH],
                                     return_tensors="pt", padding=True).to(device)
                    parts.append(forward_logit(unwrapped, inp, yes_id, no_id))
                logits = torch.cat(parts)

                ord_dists, ord_sizes = [], []
                for start, size, grp_dists in group_offsets:
                    ord_dists.extend(grp_dists)
                    ord_sizes.append(size)

                loss = criterion(logits, ord_dists, ord_sizes)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt_steps += 1
                    if is_main:
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        print(f"[r={args.lora_r}] optimizer.step() #{opt_steps}  peak={peak:.2f}GB", flush=True)
                optimizer.step()
                optimizer.zero_grad()

            if opt_steps >= 2:
                break

        accelerator.wait_for_everyone()
        peak_final = torch.cuda.max_memory_allocated() / 1e9
        if is_main:
            print(f"RESULT r={args.lora_r} OK peakVRAM={peak_final:.2f}GB", flush=True)
        sys.exit(0)

    except torch.cuda.OutOfMemoryError as e:
        if accelerator.is_main_process:
            print(f"RESULT r={args.lora_r} OOM ({e})", flush=True)
        sys.exit(1)
    except Exception as e:
        if accelerator.is_main_process:
            import traceback
            traceback.print_exc()
            print(f"RESULT r={args.lora_r} ERROR ({e})", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
