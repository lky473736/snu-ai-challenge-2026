"""
진짜 accelerate의 accelerator.accumulate() 메커니즘을 그대로 써서 K=7(n_extra=0)과 K=8(n_extra=1)을
검증한다. train.py와 동일하게 optimizer.step()/zero_grad()를 매 micro-step마다 "호출"하되,
accelerate가 GRAD_ACCUM 경계 전까지 내부적으로 no-op 처리하게 둔다(수동 시뮬레이션에서 이 부분이
실제와 달랐을 가능성이 있어 이번엔 train.py와 100% 동일한 패턴으로 검증).
3번의 GRAD_ACCUM 사이클(=진짜 optimizer 업데이트 3회)까지 버티는지 확인.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import ast
import random
import torch
import pandas as pd
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from config import DATA_DIR, SAMPLE_COUNTS, LR, GRAD_ACCUM, LORA_R, LORA_ALPHA
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.dataset import load_image, build_messages
from src.hard_negative import kendall_dist, _perms_by_dist

parser = argparse.ArgumentParser()
parser.add_argument("--n_extra", type=int, required=True)
parser.add_argument("--n_cycles", type=int, default=3)  # 진짜 optimizer.step() 몇 번 볼지
args = parser.parse_args()

pg_kwargs   = InitProcessGroupKwargs(timeout=timedelta(days=2))
accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, kwargs_handlers=[pg_kwargs])
device = accelerator.device
print(f"[verify] n_extra={args.n_extra}  GRAD_ACCUM={GRAD_ACCUM}  n_cycles={args.n_cycles}  device={device}")

model, processor = load_model_and_processor(lora_r=LORA_R, lora_alpha=LORA_ALPHA)
yes_id, no_id = get_yes_no_token_ids(processor)

n_groups = GRAD_ACCUM * args.n_cycles
train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(n_groups, random_state=3).reset_index(drop=True)
prepared = []
for _, row in train_csv.iterrows():
    gt = tuple(ast.literal_eval(row["Answer"]))
    img_dir = DATA_DIR / "train" / row["Id"]
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    base_imgs = [load_image(str(img_dir / f)) for f in files]
    prepared.append((gt, row["Sentence"], base_imgs))


def sample_group_k(gt, n_extra):
    by_dist = _perms_by_dist(gt)
    samples = [(list(gt), 0)]
    used = {gt}
    for dist, n_take in SAMPLE_COUNTS.items():
        pool = by_dist.get(dist, [])
        chosen = random.sample(pool, min(n_take, len(pool)))
        for p in chosen:
            samples.append((list(p), dist)); used.add(p)
    if n_extra > 0:
        all_negs = [p for pl in by_dist.values() for p in pl]
        pool = [p for p in all_negs if p not in used]
        for p in random.sample(pool, min(n_extra, len(pool))):
            samples.append((list(p), kendall_dist(p, gt))); used.add(p)
    return samples


from src.loss import ListwiseSoftmaxLoss
criterion = ListwiseSoftmaxLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, 1, n_groups)

model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

torch.cuda.reset_peak_memory_stats()
opt_update_count = 0
try:
    for gi, (gt, sentence, base_imgs) in enumerate(prepared):
        with accelerator.accumulate(model):
            unwrapped = accelerator.unwrap_model(model)
            texts, imgs_list, dists = [], [], []
            for perm, dist in sample_group_k(gt, args.n_extra):
                inv = [0]*4
                for ii, tp in enumerate(perm): inv[tp-1] = ii
                imgs = [base_imgs[inv[t]] for t in range(4)]
                msg = build_messages(imgs, sentence)
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                texts.append(text); imgs_list.append(imgs); dists.append(dist)
            group_size = len(texts)

            parts = []
            for bi in range(0, group_size, 8):
                inp = processor(text=texts[bi:bi+8], images=imgs_list[bi:bi+8],
                                 return_tensors="pt", padding=True).to(device)
                parts.append(forward_logit(unwrapped, processor, inp, yes_id, no_id))
            logits = torch.cat(parts)
            loss, _, _ = criterion(logits, dists, [group_size])

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt_update_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            peak_now = torch.cuda.max_memory_allocated() / 1e9
            tag = "  <- 진짜 optimizer 업데이트" if accelerator.sync_gradients else ""
            print(f"  step {gi+1}/{n_groups}  group_size={group_size}  "
                  f"누적peak={peak_now:.2f}GB{tag}", flush=True)

    final_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n결과: n_extra={args.n_extra}  OK  진짜 optimizer 업데이트 {opt_update_count}회 완료  "
          f"최종 peak={final_peak:.2f}GB", flush=True)
except torch.cuda.OutOfMemoryError as e:
    print(f"\n결과: n_extra={args.n_extra}  OOM  (optimizer 업데이트 {opt_update_count}회까지는 성공)", flush=True)
