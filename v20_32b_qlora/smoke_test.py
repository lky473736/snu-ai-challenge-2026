"""
v20 스모크 — Qwen3-VL-32B-Instruct QLoRA(4bit)가 이 하드웨어(H100)에서 실제로 돌아가는지 검증.
전례(v17/v18) 교훈 그대로 적용:
  1) 32B는 이 프로젝트에서 완전히 새로운 모델 크기라 VRAM을 아예 모름 -> 반드시 먼저 확인.
  2) idea.md §5-11 교훈: forward+backward 단발 테스트는 AdamW 옵티마이저 상태 할당을 못 잡음
     -> optimizer.step()을 실제로 2사이클(=GRAD_ACCUM*2그룹) 돌려서 확인.
  3) 실제 train.py와 동일한 accelerate 패턴 + 4-GPU DDP(accelerate launch)로 검증.

실행: accelerate launch --num_processes=4 --mixed_precision=bf16 smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import timedelta

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from config import DATA_DIR, LORA_R, LORA_ALPHA, SEED, GRAD_ACCUM, LR, TRAIN_MINIBATCH
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import ListwiseSoftmaxLoss


def _forward_chunk(model, processor, texts, imgs_list, yes_id, no_id, device, chunk_size):
    parts = []
    for bi in range(0, len(texts), chunk_size):
        inp = processor(text=texts[bi:bi + chunk_size], images=imgs_list[bi:bi + chunk_size],
                         return_tensors="pt", padding=True).to(device)
        parts.append(forward_logit(model, processor, inp, yes_id, no_id))
    return torch.cat(parts)


def main():
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    torch.manual_seed(SEED)

    if is_main:
        print(f"[v20 smoke] processes={accelerator.num_processes}  TRAIN_MINIBATCH={TRAIN_MINIBATCH}  "
              f"목표: 32B QLoRA가 H100에서 optimizer.step() 2사이클(=GRAD_ACCUM*2그룹/process) 통과하는지 확인", flush=True)

    t_load = time.time()
    model, processor = load_model_and_processor(lora_r=LORA_R, lora_alpha=LORA_ALPHA)
    yes_id, no_id = get_yes_no_token_ids(processor)
    criterion = ListwiseSoftmaxLoss()
    if is_main:
        print(f"모델 로드 소요: {(time.time()-t_load)/60:.1f}분  peak VRAM(로드 직후): "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    n_groups_needed = GRAD_ACCUM * 2 * accelerator.num_processes
    train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(frac=1, random_state=SEED).reset_index(drop=True)
    smoke_df = train_csv.iloc[:n_groups_needed].reset_index(drop=True)
    ds = GroupedTemporalDataset(smoke_df)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = 2 * accelerator.num_processes  # 스모크용 최소치, 스케줄러 형태만 필요
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, max(n_steps, 1))
    model, optimizer, dl, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)
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

            logits = _forward_chunk(unwrapped, processor, texts, imgs_list, yes_id, no_id, device, TRAIN_MINIBATCH)

            ord_dists, ord_sizes = [], []
            for start, size, grp_dists in group_offsets:
                ord_dists.extend(grp_dists)
                ord_sizes.append(size)

            loss, _, _ = criterion(logits, ord_dists, ord_sizes)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt_steps += 1
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[rank{accelerator.process_index}] optimizer.step() #{opt_steps}  "
                      f"loss={loss.item():.4f}  peak VRAM so far: {peak:.2f}GB", flush=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if opt_steps >= 2:
            break

    accelerator.wait_for_everyone()
    peak_final = torch.cuda.max_memory_allocated() / 1e9
    elapsed = (time.time() - t0) / 60
    print(f"[rank{accelerator.process_index}] 스모크 통과 — optimizer.step() {opt_steps}사이클 완료  "
          f"최종 peak VRAM: {peak_final:.2f}GB  루프 소요: {elapsed:.1f}분", flush=True)


if __name__ == "__main__":
    main()
