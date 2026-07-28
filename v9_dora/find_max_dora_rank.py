"""
DoRA(use_dora=True)가 OOM 없이 돌아가는 최대 rank를 한 job 안에서 자동 탐색.
base model은 한 번만 로드하고, 매 rank 후보마다 LoRA/DoRA 어댑터만 씌웠다 벗겼다 하며 시도.
r=128(OOM, 94MB 부족), r=120(OOM, 2MB 부족)까지 확인됨 -> 112부터 8씩 계속 낮춰가며 탐색.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType

from config import DATA_DIR, WEIGHT_TEMP, EMA_ALPHA, MODEL_PATH, LORA_DROPOUT
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn
from src.model import get_yes_no_token_ids, _get_model_class
from src.loss import AdaptiveDistanceLoss
from src.train import forward_batch

device = torch.device("cuda:0")
CANDIDATES = [112, 104, 96, 88, 80, 72, 64]

print("Loading processor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
ModelClass = _get_model_class(MODEL_PATH)

print("Loading base model (한 번만 로드, 이후 어댑터만 교체)...")
base_model = None
for attn_impl in ("flash_attention_2", "sdpa", "eager"):
    try:
        base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
        base_model = base_model.to(device)
        print(f"  -> {attn_impl} OK")
        break
    except Exception as e:
        print(f"  -> {attn_impl} skipped: {e}")
base_model.gradient_checkpointing_enable()
yes_id, no_id = get_yes_no_token_ids(processor)

train_csv = pd.read_csv(DATA_DIR / "train.csv").sample(20, random_state=0).reset_index(drop=True)
ds = GroupedTemporalDataset(train_csv)
batch = collate_fn([ds[0]])
print(f"그룹 크기(1 pos+7 neg): {batch['group_sizes']}")

texts, imgs_list, group_offsets = [], [], []
for grp_imgs, grp_sents, grp_dists in zip(batch["images"], batch["sentences"], batch["dists"]):
    start = len(texts)
    for imgs, sent in zip(grp_imgs, grp_sents):
        msg = build_messages(imgs, sent)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(imgs)
    group_offsets.append((start, len(grp_imgs), grp_dists))


import gc


def try_rank(r):
    alpha = 2 * r
    pre_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"\n{'='*50}\n시도: r={r}  alpha={alpha}  (시작 전 잔여 할당량: {pre_alloc:.2f}GB)\n{'='*50}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_dora=True,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    logits, loss = None, None
    try:
        logits = forward_batch(model, processor, texts, imgs_list, yes_id, no_id, device)
        ord_logits_parts, ord_dists, ord_sizes = [], [], []
        for start, size, grp_dists in group_offsets:
            ord_logits_parts.append(logits[start:start + size])
            ord_dists.extend(grp_dists)
            ord_sizes.append(size)
        criterion = AdaptiveDistanceLoss(ema_alpha=EMA_ALPHA, temperature=WEIGHT_TEMP)
        loss, _, _ = criterion(torch.cat(ord_logits_parts), ord_dists, ord_sizes)
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"성공! r={r}  loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}  Peak VRAM={peak:.2f}GB")
        model.zero_grad(set_to_none=True)
        model = model.unload()
        return True, peak
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] r={r} 실패 — 상세: {e}")
        del e
        logits, loss = None, None
        model.zero_grad(set_to_none=True)
        model = model.unload()
        gc.collect()
        torch.cuda.empty_cache()
        post_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  정리 후 잔여 할당량: {post_alloc:.2f}GB")
        return False, None


found = None
for r in CANDIDATES:
    ok, peak = try_rank(r)
    if ok:
        found = (r, peak)
        break

print(f"\n{'='*50}")
if found:
    print(f"최대 feasible rank: r={found[0]}  Peak VRAM={found[1]:.2f}GB")
else:
    print("모든 후보에서 OOM — DoRA는 이 VRAM으로 불가능해 보임")
