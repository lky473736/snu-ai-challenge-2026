import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd

from config import DATA_DIR, WEIGHT_TEMP, EMA_ALPHA
from src.hard_negative import generate_hard_negative_samples
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn
from src.model import load_model_and_processor, get_yes_no_token_ids
from src.loss import AdaptiveDistanceLoss
from src.train import forward_batch, val_exact_match

device = torch.device("cuda:0")

train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(20, random_state=0).reset_index(drop=True)
print("샘플 20개 중 No_ordering 분포:\n", train_csv["No_ordering"].value_counts())

df_train = generate_hard_negative_samples(train_csv)
print(f"생성된 총 row: {len(df_train)}  그룹 수: {df_train['group_id'].nunique()}")
print(df_train.head(10))

ds = GroupedTemporalDataset(df_train)
# 실제 train.py는 BATCH_SIZE=1이라 매 스텝 그룹 1개(8샘플)만 forward+즉시backward함.
# 여러 그룹을 forward만 몰아서 하고 backward를 나중에 한 번에 하면 activation이 안 풀려서
# 실제보다 메모리를 더 많이 씀(이전 시도의 OOM 원인) -> 그룹 1개로 정확히 재현.
batch = collate_fn([ds[0]])

model, processor = load_model_and_processor()
model = model.to(device)
yes_id, no_id = get_yes_no_token_ids(processor)
criterion = AdaptiveDistanceLoss(ema_alpha=EMA_ALPHA, temperature=WEIGHT_TEMP)

texts, imgs_list, group_offsets = [], [], []
for grp_imgs, grp_sents, grp_dists in zip(batch["images"], batch["sentences"], batch["dists"]):
    start = len(texts)
    for imgs, sent in zip(grp_imgs, grp_sents):
        msg = build_messages(imgs, sent)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
    group_offsets.append((start, len(grp_imgs), grp_dists))

logits = forward_batch(model, processor, texts, imgs_list, yes_id, no_id, device)
print("logits shape:", logits.shape)

ord_logits_parts, ord_dists, ord_sizes = [], [], []
for start, size, grp_dists in group_offsets:
    ord_logits_parts.append(logits[start:start+size])
    ord_dists.extend(grp_dists)
    ord_sizes.append(size)

loss, per_dist_loss, weights = criterion(torch.cat(ord_logits_parts), ord_dists, ord_sizes)
print("loss:", loss.item(), "finite:", torch.isfinite(loss).item())
print("weights:", weights.tolist())

loss.backward()
print("backward 성공")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM: {peak:.2f} GB")

print("\n=== val_exact_match 스모크 (n=3) ===")
val_df = train_csv[train_csv["No_ordering"] == False].head(3)
if len(val_df) > 0:
    acc = val_exact_match(model, processor, val_df, yes_id, no_id, device)
    print("val smoke acc:", acc)
else:
    print("(No_ordering=False 샘플 없음, val 스모크 스킵)")

print("\n스모크 테스트 전체 통과")
