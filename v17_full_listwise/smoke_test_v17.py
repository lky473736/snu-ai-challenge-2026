"""
v17 스모크 — K=23(group_size=24) 실제 학습 루프를 4-GPU DDP(accelerate launch)로 재현해서
진짜 VRAM 한계를 검증한다.

idea.md §5-11 교훈 두 가지를 그대로 적용:
  1) forward+backward 단발 테스트는 AdamW 옵티마이저 상태(momentum+variance) 할당을 포함하지
     않아 실제 학습 중 여러 GRAD_ACCUM 사이클을 거치며 서서히 OOM 나는 걸 못 잡는다
     -> optimizer.step()을 진짜로 2사이클(=GRAD_ACCUM*2=16그룹) 돌려서 확인한다.
  2) "실제 학습 코드와 동일한 accelerate 패턴 + 독립 프로세스"로 검증해야 신뢰할 수 있다
     -> notebook의 vram_selfcheck()(단일 GPU, cuda:0 고정, 단발 forward+backward)를 쓰지 않고
        train_fn()과 100% 동일한 accelerator.accumulate 패턴을, 실제 4-GPU DDP로 돌린다.

v17_full_listwise.ipynb의 §2 Config / §6 Hard Negative(K=23) / §9 train_fn 로직을 그대로
가져왔다(노트북은 건드리지 않음, 이 파일은 스모크 전용 독립 스크립트).

실행: accelerate launch --num_processes=4 --mixed_precision=bf16 smoke_test_v17.py
"""

import ast
import time
from datetime import timedelta
from itertools import permutations
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

# ── 이 서버에 이미 있는 데이터/모델을 그대로 씀 (노트북의 ./data, ./models 상대경로 대신
#    절대경로로 재다운로드 없이 재사용) ──────────────────────────────────────────
DATA_DIR   = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"

MAX_IMAGE_SIZE = 448
SEED = 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 128, 256, 0.05
LR = 5e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8
# v17 노트북 §2의 시작값 그대로 — 이 스모크의 목적이 바로 "이 값이 이 서버에서 버티는지" 확인.
# OOM 나면 이 값을 4/2로 줄여서 재실행.
TRAIN_MINIBATCH = 8

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


def sample_group_full(gt):
    """v17 핵심: 샘플링 없이 23개 오답 전부 + 정답 1개 = 24개."""
    samples = [(list(gt), 0)]
    for p in ALL_PERMS:
        if p == gt:
            continue
        samples.append((list(p), kendall_dist(p, gt)))
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


class GroupedTemporalDatasetFull(Dataset):
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
        for perm, dist in sample_group_full(gt):
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
    """v14/v17과 동일 — group_size=24(K=23 전체)면 근사가 아니라 완전한 Plackett-Luce top-1."""
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


def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ModelClass = get_model_class(MODEL_PATH)
    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("No attn_implementation worked")
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
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
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    torch.manual_seed(SEED)

    if is_main:
        print(f"[v17 smoke] processes={accelerator.num_processes}  TRAIN_MINIBATCH={TRAIN_MINIBATCH}  "
              f"group_size=24(K=23 전체)  목표: optimizer.step() 2사이클(=16그룹/process) 통과", flush=True)

    model, processor = load_model_and_processor()
    yes_id, no_id = get_yes_no_token_ids(processor)
    criterion = ListwiseSoftmaxLoss()

    # 프로세스마다 GRAD_ACCUM*2(=16)그룹씩 돌아가도록 전체 num_processes*16개만 사용
    n_groups_needed = GRAD_ACCUM * 2 * accelerator.num_processes
    train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(frac=1, random_state=SEED).reset_index(drop=True)
    smoke_df = train_csv.iloc[:n_groups_needed].reset_index(drop=True)
    ds = GroupedTemporalDatasetFull(smoke_df)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    opt_steps = 0
    t0 = time.time()

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

            logit_parts = []
            for bi in range(0, len(texts), TRAIN_MINIBATCH):
                inp = processor(text=texts[bi:bi + TRAIN_MINIBATCH], images=imgs_list[bi:bi + TRAIN_MINIBATCH],
                                 return_tensors="pt", padding=True).to(device)
                logit_parts.append(forward_logit(unwrapped, inp, yes_id, no_id))
            logits = torch.cat(logit_parts)

            ord_dists, ord_sizes = [], []
            for start, size, grp_dists in group_offsets:
                ord_dists.extend(grp_dists)
                ord_sizes.append(size)

            loss = criterion(logits, ord_dists, ord_sizes)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt_steps += 1
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[rank{accelerator.process_index}] optimizer.step() #{opt_steps}  "
                      f"loss={loss.item():.4f}  peak VRAM so far: {peak:.2f}GB", flush=True)
            optimizer.step()
            optimizer.zero_grad()

        if opt_steps >= 2:
            break

    accelerator.wait_for_everyone()
    peak_final = torch.cuda.max_memory_allocated() / 1e9
    elapsed = (time.time() - t0) / 60
    print(f"[rank{accelerator.process_index}] 스모크 통과 — TRAIN_MINIBATCH={TRAIN_MINIBATCH}  "
          f"optimizer.step() {opt_steps}사이클 완료  최종 peak VRAM: {peak_final:.2f}GB  소요: {elapsed:.1f}분", flush=True)


if __name__ == "__main__":
    main()
