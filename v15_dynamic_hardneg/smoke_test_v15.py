import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd

from config import DATA_DIR, LORA_R, LORA_ALPHA
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn
from src.model import load_model_and_processor, get_yes_no_token_ids
from src.loss import ListwiseSoftmaxLoss
from src.train import forward_batch, val_exact_match, sync_bank

device = torch.device("cuda:0")
print(f"LORA_R={LORA_R}  LORA_ALPHA={LORA_ALPHA}")

train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(6, random_state=0).reset_index(drop=True)

# n_extra는 이제 hard_negative.sample_group에서 무시됨(K=7 고정, replace 방식) — 인자로 줘도 무해.
ds = GroupedTemporalDataset(train_csv)
model, processor = load_model_and_processor()
model = model.to(device)
yes_id, no_id = get_yes_no_token_ids(processor)
criterion = ListwiseSoftmaxLoss()

print("\n=== epoch1 시뮬레이션 (뱅크 비어있음) ===")
local_updates = {}
for i in range(len(ds)):
    item = ds[i]
    batch = collate_fn([item])
    texts, imgs_list, group_offsets = [], [], []
    for grp_imgs, grp_sents, grp_dists, grp_perms, sid in zip(
        batch["images"], batch["sentences"], batch["dists"], batch["perms"], batch["sample_ids"]
    ):
        start = len(texts)
        for imgs, sent in zip(grp_imgs, grp_sents):
            msg = build_messages(imgs, sent)
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text); imgs_list.append(imgs)
        group_offsets.append((start, len(grp_imgs), grp_dists, grp_perms, sid))

    logits = forward_batch(model, processor, texts, imgs_list, yes_id, no_id, device)
    ord_logits, ord_dists, ord_sizes = [], [], []
    for start, size, grp_dists, grp_perms, sid in group_offsets:
        g_logits = logits[start:start+size]
        ord_logits.append(g_logits); ord_dists.extend(grp_dists); ord_sizes.append(size)
        neg_idxs = [j for j, d in enumerate(grp_dists) if d != 0]
        scores = g_logits.detach()
        best_j = max(neg_idxs, key=lambda j: scores[j].item())
        local_updates[sid] = (grp_perms[best_j], scores[best_j].item())

    loss, _, _ = criterion(torch.cat(ord_logits), ord_dists, ord_sizes)
    loss.backward()
    model.zero_grad()
    print(f"  sample {i}: group_size={sum(1 for _ in item['perms'])}, loss={loss.item():.4f}")

global_bank = sync_bank(local_updates, world_size=1)
print(f"\nepoch1 끝 — 뱅크 크기: {len(global_bank)} (기대: {len(ds)})")
ds.bank = global_bank

print("\n=== epoch2 시뮬레이션 (뱅크 반영됨) — d1 보너스 슬롯이 뱅크로 교체되는지 확인 ===")
sample0 = ds.df.iloc[0]
sid0 = sample0["Id"]
if sid0 in ds.bank:
    hard_perm = tuple(ds.bank[sid0][0])
    print(f"  {sid0}의 뱅크 하드 네거티브: {hard_perm}")
    item2 = ds[0]
    perms_in_group = item2["perms"]
    print(f"  epoch2 그룹에 포함됐는가: {hard_perm in perms_in_group}")
    print(f"  dist 분포: {sorted(item2['dists'])}  (d1~d6 전 구간 커버 여부: "
          f"{set(d for d in item2['dists'] if d>0) == {1,2,3,4,5,6}})")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPeak VRAM: {peak:.2f} GB")

print("\n=== val_exact_match 스모크 (n=3) ===")
val_df = train_csv[train_csv["No_ordering"] == False].head(3)
if len(val_df):
    acc = val_exact_match(model, processor, val_df, yes_id, no_id, device)
    print("val smoke acc:", acc)

print("\n스모크 테스트 전체 통과")
