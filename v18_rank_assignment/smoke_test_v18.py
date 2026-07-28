import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd

from config import DATA_DIR, LORA_R, LORA_ALPHA, SEED, GRAD_ACCUM, LR
from src.dataset import PairwiseTemporalDataset, load_image, PAIRS, collate_fn
from src.model import load_model_and_processor, get_yes_no_token_ids, pairwise_logits_fast
from src.loss import pairwise_bt_loss
from src.aggregate import score_pairs_fast, score_pairs_slow, aggregate_ranks

device = torch.device("cuda:0")
print(f"LORA_R={LORA_R}  LORA_ALPHA={LORA_ALPHA}  (alpha/r={LORA_ALPHA/LORA_R:.2f})")

model, processor = load_model_and_processor()
model = model.to(device)
yes_id, no_id = get_yes_no_token_ids(processor)

# ── 1) 고속 경로 vs 느린 경로(참조) 일치 확인 — 반드시 여기서 통과해야 아래로 진행 ──
print("\n=== 1) 고속 경로 검증 (slow vs fast) ===")
train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(3, random_state=SEED).reset_index(drop=True)
max_diff = 0.0
for _, row in train_csv.iterrows():
    sid, sentence = row["Id"], row["Sentence"]
    img_dir = DATA_DIR / "train" / sid
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    base_imgs = [load_image(str(img_dir / f)) for f in files]

    slow = score_pairs_slow(model, processor, base_imgs, sentence, yes_id, no_id, device)
    fast = score_pairs_fast(model, processor, base_imgs, sentence, device)
    for pair in PAIRS:
        diff = abs(slow[pair] - fast[pair])
        max_diff = max(max_diff, diff)
        print(f"  {sid} {pair}  slow={slow[pair]:.4f}  fast={fast[pair]:.4f}  diff={diff:.5f}")

print(f"\n최대 오차: {max_diff:.5f}")
assert max_diff <= 1e-2, (
    f"고속 경로 결과가 느린 경로와 {max_diff:.5f}만큼 어긋남 — 학습 전에 원인부터 잡을 것."
)
print("검증 통과 — 고속 경로 사용 가능")

# ── 2) loss / backward / VRAM 스모크 ────────────────────────────
print("\n=== 2) loss / backward / VRAM 스모크 ===")
torch.cuda.reset_peak_memory_stats()
ds = PairwiseTemporalDataset(train_csv)
batch = collate_fn([ds[0]])
print(f"그룹 크기(4C2): {batch['group_sizes']}")

sid0, base_imgs0, sentence0 = batch["sids"][0], batch["images"][0][0], batch["sentences"][0][0]
z = pairwise_logits_fast(model, processor, base_imgs0, sentence0, device)
labels0, adj0 = batch["labels"][0], batch["adj_flags"][0]
loss = pairwise_bt_loss(z, labels0, adj0)
print("loss:", loss.item(), "finite:", torch.isfinite(loss).item())

loss.backward()
print("backward 성공")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM: {peak:.2f} GB")

# ── 3) 집계(Hungarian 없이 행 평균) 스모크 ───────────────────────
print("\n=== 3) predict_permutation 스모크 ===")
pair_scores = score_pairs_fast(model, processor, base_imgs0, sentence0, device)
pred = aggregate_ranks(pair_scores)
gt = train_csv.iloc[0]["Answer"]
print(f"pred={pred}  gt={gt}")

# ── 4) 실제 학습 루프 재현: optimizer.step() 2사이클 통과 확인 ─────
# idea.md §5-11 교훈: forward+backward 단발 테스트는 AdamW 옵티마이저 상태(momentum+variance)
# 할당을 포함하지 않아, 실제 학습 중 GRAD_ACCUM 사이클을 여러 번 거치며 서서히 OOM 나는 걸
# 못 잡는다(K=8 때 1번째 사이클은 통과하고 2번째 사이클 도중 OOM났던 전례 재현). 여기서는
# train.py와 동일한 accelerator.accumulate 패턴으로 진짜 optimizer.step()을 2회 일으켜서
# 그 함정을 미리 잡는다. 단일 GPU로 실행하지만 DDP에서도 rank당 옵티마이저 상태 크기는
# world_size와 무관하므로 peak VRAM 근사치로 유효함 — 그래도 최종 4-GPU 잡 로그의 peak VRAM은
# 별도로 다시 확인할 것.
print("\n=== 4) 실제 학습 루프 재현 (optimizer.step() 2사이클, GRAD_ACCUM=%d) ===" % GRAD_ACCUM)
from torch.optim import AdamW
from accelerate import Accelerator

model.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()

accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
model, optimizer = accelerator.prepare(model, optimizer)
model.train()

n_groups = GRAD_ACCUM * 2  # optimizer.step() 2회치 분량
cycle_df = pd.read_csv(DATA_DIR / "train.csv").sample(n_groups, random_state=SEED + 1).reset_index(drop=True)
cycle_ds = PairwiseTemporalDataset(cycle_df)

opt_steps = 0
for idx in range(n_groups):
    batch = collate_fn([cycle_ds[idx]])
    base_imgs, sentence = batch["images"][0][0], batch["sentences"][0][0]
    labels, adj = batch["labels"][0], batch["adj_flags"][0]
    with accelerator.accumulate(model):
        unwrapped = accelerator.unwrap_model(model)
        z = pairwise_logits_fast(unwrapped, processor, base_imgs, sentence, device)
        loss = pairwise_bt_loss(z, labels, adj)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if accelerator.sync_gradients:
            opt_steps += 1
            print(f"  optimizer.step() #{opt_steps} 완료 — 누적 peak VRAM: "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

assert opt_steps == 2, f"예상한 optimizer.step() 2회와 다름(실제 {opt_steps}회) — n_groups/GRAD_ACCUM 확인 필요"
print(f"GRAD_ACCUM={GRAD_ACCUM} 기준 optimizer.step() 2사이클 통과. "
      f"최종 peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\n스모크 테스트 전체 통과")
