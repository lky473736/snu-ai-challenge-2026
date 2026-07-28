"""
n_extra=0 vs n_extra=1을 optimizer.step()까지 포함해서 실측 (그리드서치는 forward+backward까지만
봐서 AdamW 옵티마이저 상태(momentum+variance) 메모리가 빠져있었음 — 이 차이를 실제로 확인).
GRAD_ACCUM=8이라 실제 학습에서도 8 스텝 연속 backward 후 optimizer.step() 1회이므로, 그 패턴을
그대로 재현해서 8 그룹 연속 처리 후 optimizer.step()까지 실행.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import pandas as pd
import ast
import gc

from config import DATA_DIR, SAMPLE_COUNTS, LR
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.dataset import load_image, build_messages
from src.hard_negative import kendall_dist, _perms_by_dist
from src.loss import ListwiseSoftmaxLoss
from torch.optim import AdamW
import random

parser = argparse.ArgumentParser()
parser.add_argument("--n_extra", type=int, required=True)
args = parser.parse_args()

device = torch.device("cuda:0")

print("모델 로딩...")
model, processor = load_model_and_processor()
model = model.to(device)
model.gradient_checkpointing_enable()
yes_id, no_id = get_yes_no_token_ids(processor)
criterion = ListwiseSoftmaxLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(8, random_state=2).reset_index(drop=True)
prepared = []
for _, row in train_csv.iterrows():
    gt = tuple(ast.literal_eval(row["Answer"]))
    img_dir = DATA_DIR / "train" / row["Id"]
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    base_imgs = [load_image(str(img_dir / f)) for f in files]
    prepared.append((row["Id"], gt, row["Sentence"], base_imgs))


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


def run_grad_accum_cycle(n_extra, minibatch=8):
    """GRAD_ACCUM=8 그룹을 연속 backward 후 optimizer.step() 1회 — 실제 학습 1 optimizer step 재현.
    각 단계(micro-step, optimizer.step())마다 peak를 찍어서 OOM 발생 지점을 정확히 짚는다."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for gi in range(8):  # GRAD_ACCUM
            sid, gt, sentence, base_imgs = prepared[gi % len(prepared)]
            texts, imgs_list, dists = [], [], []
            for perm, dist in sample_group_k(gt, n_extra):
                inv = [0]*4
                for ii, tp in enumerate(perm): inv[tp-1] = ii
                imgs = [base_imgs[inv[t]] for t in range(4)]
                msg = build_messages(imgs, sentence)
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                texts.append(text); imgs_list.append(imgs); dists.append(dist)
            group_size = len(texts)

            parts = []
            for bi in range(0, group_size, minibatch):
                inp = processor(text=texts[bi:bi+minibatch], images=imgs_list[bi:bi+minibatch],
                                 return_tensors="pt", padding=True).to(device)
                parts.append(forward_logit(model, processor, inp, yes_id, no_id))
            logits = torch.cat(parts)
            loss, _, _ = criterion(logits, dists, [group_size])
            (loss / 8).backward()  # GRAD_ACCUM 평균
            mid_peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"    micro-step {gi+1}/8 완료, 누적 peak={mid_peak:.2f}GB", flush=True)

        pre_opt_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"    optimizer.step() 직전 peak={pre_opt_peak:.2f}GB (첫 호출이면 여기서 momentum/variance 최초 할당)", flush=True)
        optimizer.step()
        optimizer.zero_grad()
        peak = torch.cuda.max_memory_allocated() / 1e9
        return "OK", peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); optimizer.zero_grad()
        return "OOM", None


status, peak = run_grad_accum_cycle(args.n_extra)
print(f"n_extra={args.n_extra}  (GRAD_ACCUM=8 사이클 + optimizer.step() 포함)  "
      f"{status}  peak={peak if peak is None else f'{peak:.2f}GB'}", flush=True)
