"""
HNTV v7 Training — DDP via Accelerate
v7: v6.5(Qwen3-VL-8B + 448px + live hard-negative resampling, best val=0.6029) 위에
AdaptiveDistanceNounLoss(n_nouns 축 이중 적응 가중치)를 추가.
EDA.md §10-2: sent_len/temporal_words/max_dep_depth/n_verbs는 전부 n_nouns와의 다중공선성에서
온 부산물이었고, 로지스틱회귀+bootstrap CI+split-half 재현성까지 통과한 유일한 독립 신호는
n_nouns(문장에 언급된 명사 개수, 구간별 정확도 29.4%->83.3%)였음이 확정됨.
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import csv
import time
import argparse
import torch
import pandas as pd
from itertools import permutations
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm

from config import (
    DATA_DIR, CKPT_DIR, LOG_DIR,
    EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR, WARMUP_RATIO,
    LOGGING_STEPS, VAL_RATIO, SEED,
    LORA_R, LORA_ALPHA, MODEL_PATH,
    TRAIN_MINIBATCH, INFER_BATCH_SIZE,
    EMA_ALPHA, WEIGHT_TEMP, NOUN_WEIGHT_TEMP,
)
from src.dataset import GroupedTemporalDataset, build_messages, collate_fn, load_image
from src.model import load_model_and_processor, get_yes_no_token_ids, forward_logit
from src.loss import AdaptiveDistanceNounLoss, N_NOUN_BUCKETS

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def val_exact_match(model, processor, val_raw_df, yes_id, no_id, device, n_samples=None):
    model.eval()
    df = val_raw_df if n_samples is None else val_raw_df.sample(min(n_samples, len(val_raw_df)), random_state=SEED)
    correct, wrong_cases = 0, []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="val", leave=False):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        gt        = ast.literal_eval(row["Answer"])
        img_dir   = DATA_DIR / "train" / sample_id
        files     = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / files[i])) for i in range(4)]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        all_texts, all_imgs = [], []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_imgs.append(imgs)
        with torch.no_grad():
            s = _forward_chunk(model, processor, all_texts, all_imgs, yes_id, no_id, device, INFER_BATCH_SIZE)
        scores = [(s[i].item(), list(perm)) for i, perm in enumerate(ALL_PERMS)]

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    acc = correct / len(df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[Val] exact_match={acc:.4f} ({correct}/{len(df)})  "
          f"adj_swap_fail={adj}  other_fail={len(wrong_cases) - adj}")
    model.train()
    return acc


def _forward_chunk(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, chunk_size):
    """OOM 나면 chunk_size를 절반으로 줄여 재시도 (512px, 여유 3.2GB로 빡빡해서 안전장치 추가)"""
    while True:
        try:
            parts = []
            for bi in range(0, len(texts), chunk_size):
                inp = processor(
                    text=texts[bi: bi + chunk_size],
                    images=imgs_list[bi: bi + chunk_size],
                    return_tensors="pt", padding=True,
                ).to(device)
                parts.append(forward_logit(model_unwrapped, processor, inp, yes_id, no_id))
            return torch.cat(parts)
        except torch.cuda.OutOfMemoryError:
            if chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            print(f"[OOM] forward chunk_size -> {chunk_size}", flush=True)


def forward_batch(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device):
    return _forward_chunk(model_unwrapped, processor, texts, imgs_list, yes_id, no_id, device, TRAIN_MINIBATCH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r",     type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha", type=int,   default=LORA_ALPHA)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--ckpt_name",  type=str,   default="best_v7")
    parser.add_argument("--resume_from", type=str,  default=None,
                         help="job이 중간에 끊겼을 때 대비: 저장된 LoRA 체크포인트 경로(예: checkpoints/best_v7_last)"
                              "에서 가중치만 이어받아 재출발. 옵티마이저/스케줄러는 새로 초기화됨 (true resume 아님).")
    parser.add_argument("--start_epoch", type=int,  default=1,
                         help="--resume_from과 함께 사용: 몇 번째 epoch부터 다시 돌릴지 (예: epoch4까지 끝났으면 5)")
    parser.add_argument("--prev_best_val", type=float, default=0.0,
                         help="--resume_from과 함께 사용: 이전 run에서 기록된 best val_acc "
                              "(안 채우면 첫 epoch에 무조건 덮어써버림)")
    args = parser.parse_args()

    pg_kwargs   = InitProcessGroupKwargs(timeout=timedelta(days=2))
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM,
                               kwargs_handlers=[pg_kwargs])
    device  = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[v7] Processes={accelerator.num_processes}  Device={device}")
        print(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  lr={args.lr}")
        print(f"  Loss: AdaptiveDistanceNounLoss (temp_dist={WEIGHT_TEMP}, temp_noun={NOUN_WEIGHT_TEMP}, "
              f"all d=1..6 covered, n_nouns {N_NOUN_BUCKETS}-bucket 축 추가)")
        print(f"  Hard negatives: LIVE resampled every __getitem__ (v6 was fixed-once)")
        if args.resume_from:
            print(f"  [resume] {args.resume_from} 에서 재출발, start_epoch={args.start_epoch}")

    torch.manual_seed(SEED)

    # ── 데이터 ─────────────────────────────────────────────────────
    # v6.5: hard negative를 학습 전 CSV로 고정하지 않고, GroupedTemporalDataset이
    # __getitem__마다 sample_group()으로 새로 뽑는다 (매 epoch, 사실상 매 접근마다 조합이 달라짐).
    # v7: n_nouns 룩업(precompute_n_nouns.py로 미리 계산)을 Id 기준으로 병합.
    n_nouns_lookup = pd.read_csv(CKPT_DIR / "_n_nouns_lookup.csv")

    if is_main:
        train_csv = pd.read_csv(DATA_DIR / "train.csv")
        train_csv = train_csv.merge(n_nouns_lookup, on="Id", how="left")
        assert train_csv["n_nouns"].notna().all(), "n_nouns 룩업에 없는 Id 존재 — precompute_n_nouns.py 재실행 필요"
        train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val     = int(len(train_csv) * VAL_RATIO)
        val_raw   = train_csv[:n_val]
        trn_raw   = train_csv[n_val:]

        val_raw.to_csv(CKPT_DIR / "_val_raw.csv", index=False)
        print(f"Train samples (raw, pre-expansion): {len(trn_raw)}")

    accelerator.wait_for_everyone()

    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    train_csv = train_csv.merge(n_nouns_lookup, on="Id", how="left")
    train_csv = train_csv.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val     = int(len(train_csv) * VAL_RATIO)
    trn_raw   = train_csv[n_val:]
    val_raw   = pd.read_csv(CKPT_DIR / "_val_raw.csv")

    def _worker_init_fn(worker_id):
        # sample_group()은 전역 random 모듈을 씀. fork된 워커가 동일 시드를 물려받아
        # 서로 똑같은 negative를 뽑는 걸 막기 위해 워커별로 다시 시드를 섞는다.
        import random as _random
        seed = (SEED + accelerator.process_index * 1000 + worker_id
                + int(time.time() * 1000) % 100000)
        _random.seed(seed)

    train_ds = GroupedTemporalDataset(trn_raw)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2,
        worker_init_fn=_worker_init_fn,
    )

    # ── 모델 ────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                                  resume_from=args.resume_from)
    yes_id, no_id    = get_yes_no_token_ids(processor)
    criterion        = AdaptiveDistanceNounLoss(ema_alpha=EMA_ALPHA, temperature=WEIGHT_TEMP,
                                                  noun_temperature=NOUN_WEIGHT_TEMP)

    remaining_epochs = EPOCHS - args.start_epoch + 1
    n_steps   = (len(train_dl) // GRAD_ACCUM) * remaining_epochs
    n_warmup  = int(n_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )

    best_val_acc = args.prev_best_val
    global_step  = 0

    weight_log_path = LOG_DIR / "dist_weights.csv"
    if is_main and not (args.resume_from and weight_log_path.exists()):
        with open(weight_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "w1","w2","w3","w4","w5","w6",
                "ema1","ema2","ema3","ema4","ema5","ema6",
                "L1","L2","L3","L4","L5","L6",
                "wn1","wn2","wn3","wn4","wn5",
                "eman1","eman2","eman3","eman4","eman5",
            ])

    for epoch in range(args.start_epoch, EPOCHS + 1):
        t0 = time.time()
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{EPOCHS}  시작: {time.strftime('%H:%M:%S')}")
            print(f"  Effective batch = {BATCH_SIZE}×{GRAD_ACCUM}×{accelerator.num_processes}")
            print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dl, desc=f"Ep{epoch}", disable=not is_main)):
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                texts, imgs_list = [], []
                group_offsets   = []  # (start, size, grp_dists, noun_bucket)

                for grp_imgs, grp_sents, grp_dists, grp_nb in zip(
                    batch["images"], batch["sentences"], batch["dists"], batch["noun_buckets"]
                ):
                    start = len(texts)
                    for imgs, sent in zip(grp_imgs, grp_sents):
                        msg  = build_messages(imgs, sent)
                        text = processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                        texts.append(text)
                        imgs_list.append(imgs)
                    group_offsets.append((start, len(grp_imgs), grp_dists, grp_nb))

                logits = forward_batch(unwrapped, processor, texts, imgs_list, yes_id, no_id, device)

                ord_logits_parts, ord_dists, ord_sizes, ord_noun_buckets = [], [], [], []

                for start, size, grp_dists, grp_nb in group_offsets:
                    g_logits = logits[start: start + size]
                    ord_logits_parts.append(g_logits)
                    ord_dists.extend(grp_dists)
                    ord_sizes.append(size)
                    ord_noun_buckets.append(grp_nb)

                loss, per_dist_loss, weights, noun_weights = criterion(
                    torch.cat(ord_logits_parts), ord_dists, ord_sizes, ord_noun_buckets
                )

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss  += loss.item()

                if is_main and global_step % LOGGING_STEPS == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg    = epoch_loss / (step + 1)
                    w  = weights.tolist()
                    e  = criterion.ema.tolist()
                    wn = noun_weights.tolist()
                    en = criterion.ema_noun.tolist()
                    L = [per_dist_loss.get(d, float("nan")) for d in range(1, 7)]
                    print(
                        f"  step={global_step:5d}  loss={avg:.4f}  lr={lr_now:.2e}"
                        f"  w=[{','.join(f'{x:.3f}' for x in w)}]"
                        f"  wn=[{','.join(f'{x:.3f}' for x in wn)}]"
                        f"  L=[{','.join(f'{x:.3f}' if not (x!=x) else 'nan' for x in L)}]"
                    )
                    with open(weight_log_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [global_step] + w + e + L + wn + en
                        )

        accelerator.wait_for_everyone()

        if is_main:
            elapsed = (time.time() - t0) / 60
            print(f"\n[Epoch {epoch}/{EPOCHS} 완료]  소요: {elapsed:.1f}분  "
                  f"avg_loss: {epoch_loss / len(train_dl):.4f}")

            unwrapped = accelerator.unwrap_model(model)
            val_acc = val_exact_match(unwrapped, processor, val_raw, yes_id, no_id, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = CKPT_DIR / args.ckpt_name
                unwrapped.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"  ★ Best saved (val_acc={val_acc:.4f})")

            last_path = CKPT_DIR / (args.ckpt_name + "_last")
            unwrapped.save_pretrained(last_path)
            processor.save_pretrained(last_path)
            print(f"  → Last saved to {last_path.name}")

        accelerator.wait_for_everyone()

    if is_main:
        print(f"\n학습 완료. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
