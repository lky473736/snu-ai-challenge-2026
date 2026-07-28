"""
v15 (N_EXTRA, TRAIN_MINIBATCH) VRAM 그리드서치.

실제 train.py의 흐름을 그대로 재현: 그룹 크기(1+7+N_EXTRA)를 TRAIN_MINIBATCH 단위로 청크 나눠
forward(각 청크의 계산 그래프가 backward 전까지 누적) -> 전체 logit 모아서 ListwiseSoftmaxLoss ->
backward 1회. 청크가 쌓이는 실제 구조를 그대로 써야 VRAM을 과소평가 안 함(5-4/6절 교훈:
반복 이미지로 테스트하면 실제보다 메모리 적게 나옴 -> 여기서도 매번 다른 실제 train 이미지 사용).

각 (n_extra, minibatch) 조합에 대해 OK/OOM과 peak VRAM을 기록, 실행 가능한 조합 중 총 그룹 크기가
제일 큰(=한 스텝에 가장 많은 negative를 보는) 조합을 추천.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import torch
import pandas as pd

from config import DATA_DIR, MODEL_PATH, SAMPLE_COUNTS
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.dataset import load_image, build_messages
from src.hard_negative import kendall_dist, _perms_by_dist
from src.loss import ListwiseSoftmaxLoss
import ast
import random


def sample_group_variable_k(gt, n_extra):
    """그리드서치 전용 — hard_negative.py는 이미 K=7 고정으로 확정됐으므로(replace 방식),
    n_extra를 실제로 변화시켜 VRAM 한계를 세밀하게(1단위) 재기 위한 별도 함수.
    기본 7(d1..6 전구간, v8/v14와 동일) + distribution 풀에서 추가 n_extra개 무작위."""
    by_dist = _perms_by_dist(gt)
    samples = [(list(gt), 0)]
    used = {gt}
    for dist, n_take in SAMPLE_COUNTS.items():
        pool = by_dist.get(dist, [])
        chosen = random.sample(pool, min(n_take, len(pool)))
        for p in chosen:
            samples.append((list(p), dist))
            used.add(p)
    if n_extra > 0:
        all_negs = [p for plist in by_dist.values() for p in plist]
        pool = [p for p in all_negs if p not in used]
        extra = random.sample(pool, min(n_extra, len(pool)))
        for p in extra:
            samples.append((list(p), kendall_dist(p, gt)))
            used.add(p)
    return samples

device = torch.device("cuda:0")

print("모델 로딩 (LoRA r=128, v15 config)...")
model, processor = load_model_and_processor()
model = model.to(device)
yes_id, no_id = get_yes_no_token_ids(processor)
criterion = ListwiseSoftmaxLoss()

# 실제 train 샘플 여러 개(이미지 다양성 확보)
train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(8, random_state=1).reset_index(drop=True)
prepared = []
for _, row in train_csv.iterrows():
    sid = row["Id"]
    gt = tuple(ast.literal_eval(row["Answer"]))
    img_dir = DATA_DIR / "train" / sid
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    base_imgs = [load_image(str(img_dir / f)) for f in files]
    prepared.append((sid, gt, row["Sentence"], base_imgs))


def build_group(sid, gt, sentence, base_imgs, n_extra):
    texts, imgs_list, dists = [], [], []
    for perm, dist in sample_group_variable_k(gt, n_extra):
        inv = [0] * 4
        for inp_idx, t_pos in enumerate(perm):
            inv[t_pos - 1] = inp_idx
        imgs = [base_imgs[inv[t]] for t in range(4)]
        msg = build_messages(imgs, sentence)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
        dists.append(dist)
    return texts, imgs_list, dists


def try_combo_once(n_extra, minibatch, sample_idx):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sid, gt, sentence, base_imgs = prepared[sample_idx % len(prepared)]
        texts, imgs_list, dists = build_group(sid, gt, sentence, base_imgs, n_extra)
        group_size = len(texts)

        logit_parts = []
        for bi in range(0, group_size, minibatch):
            inp = processor(
                text=texts[bi: bi + minibatch], images=imgs_list[bi: bi + minibatch],
                return_tensors="pt", padding=True,
            ).to(device)
            logit_parts.append(forward_logit(model, processor, inp, yes_id, no_id))
        logits = torch.cat(logit_parts)

        loss, _, _ = criterion(logits, dists, [group_size])
        loss.backward()
        model.zero_grad()

        peak = torch.cuda.max_memory_allocated() / 1e9
        return "OK", peak, group_size
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.zero_grad()
        return "OOM", None, None


def try_combo(n_extra, minibatch, n_repeat=2):
    """서로 다른 실제 샘플로 n_repeat번 반복해서 더 안 좋은(=peak 더 높은/OOM) 쪽을 채택 —
    단편화·샘플별 편차로 인한 노이즈를 줄여 더 보수적이고 신뢰도 높은 판정을 얻기 위함."""
    worst_status, worst_peak, gs = "OK", -1.0, None
    for i in range(n_repeat):
        status, peak, group_size = try_combo_once(n_extra, minibatch, sample_idx=i)
        gs = group_size if group_size is not None else gs
        if status == "OOM":
            return "OOM", None, gs
        if peak > worst_peak:
            worst_peak = peak
    return worst_status, worst_peak, gs


N_EXTRA_GRID    = list(range(0, 13))   # 0~12, 1단위
MINIBATCH_GRID  = list(range(1, 33))   # 1~32, 1단위

results = []
for n_extra in N_EXTRA_GRID:
    for minibatch in MINIBATCH_GRID:
        status, peak, group_size = try_combo(n_extra, minibatch)
        gs = group_size if group_size is not None else (1 + 7 + n_extra)
        row = {"n_extra": n_extra, "minibatch": minibatch, "group_size": gs,
               "status": status, "peak_gb": round(peak, 2) if peak else None}
        results.append(row)
        print(f"n_extra={n_extra:3d}  minibatch={minibatch:3d}  group_size={gs:3d}  "
              f"{status:4s}  peak={peak if peak is None else f'{peak:.2f}GB'}", flush=True)

df = pd.DataFrame(results)
df.to_csv("/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/grid_search_results.csv", index=False)

ok = df[df.status == "OK"].copy()
if len(ok):
    # "빵빵하게" = 한 스텝에 가장 많은 candidate를 보는(=group_size가 큰) 조합 중,
    # peak_gb에 안전마진(H100 80GB의 90%=72GB) 남기고 제일 큰 것 추천
    safe = ok[ok.peak_gb <= 72.0]
    best = (safe if len(safe) else ok).sort_values(
        ["group_size", "minibatch"], ascending=[False, False]
    ).iloc[0]
    print(f"\n추천 조합: n_extra={int(best.n_extra)}  minibatch={int(best.minibatch)}  "
          f"group_size={int(best.group_size)}  peak={best.peak_gb}GB")
else:
    print("\n실행 가능한 조합이 없음 — 그리드 범위를 낮춰야 함")

print("\n저장: grid_search_results.csv")
