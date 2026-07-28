"""
Qwen2.5-VL-7B-Instruct vs Qwen3-VL-8B-Instruct — zero-shot(LoRA 없음) val 비교
같은 job 안에서 순차 평가 (모델 하나 끝나면 GPU 비우고 다음 모델 로드)

비교 기준 (idea.md 11절):
  Qwen2-VL-7B base zero-shot : 5.5%
  TPRU-7B zero-shot          : 26.3% (105/399)
  Video-R1-7B zero-shot      : 24.3% (97/399)
  Qwen2-VL v1 fine-tuned     : 50.1% (200/399)

사용법:
  python eval_both.py --n_samples 399
"""
import sys, os, gc, argparse, ast
sys.path.insert(0, "/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair")

import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig

from src.dataset import build_messages, load_image, DATA_DIR
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))

MODELS = [
    ("Qwen2.5-VL-7B-Instruct", "/data/gyuyeonlim/models/Qwen2.5-VL-7B-Instruct"),
    ("Qwen3-VL-8B-Instruct",   "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"),
]


def get_model_class(model_path: str):
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


def load_val(val_csv: str, n_samples: int, seed: int = 42):
    val_df = pd.read_csv(val_csv)
    if "No_ordering" in val_df.columns:
        val_df = val_df[val_df["No_ordering"] != True].reset_index(drop=True)
    val_df = val_df.sample(min(n_samples, len(val_df)), random_state=seed)
    return val_df


def evaluate(name, model_path, val_df, device):
    print(f"\n{'='*60}\nLoading {name} ({model_path})\n{'='*60}", flush=True)
    ModelClass = get_model_class(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = ModelClass.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            ).to(device)
            print(f"  -> {attn_impl} OK", flush=True)
            break
        except Exception as e:
            print(f"  -> {attn_impl} skipped: {e}", flush=True)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    correct, wrong_cases = 0, []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=name):
        sample_id = row["Id"]
        sentence  = row["Sentence"]
        gt        = ast.literal_eval(row["Answer"])
        img_dir   = DATA_DIR / "train" / sample_id
        files     = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        scores = []
        for perm in ALL_PERMS:
            perm_list = list(perm)
            imgs = reorder(perm_list)
            msgs = build_messages(imgs, sentence)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp  = processor(text=[text], images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                s = forward_logit(model, processor, inp, yes_id, no_id)
            scores.append((s.item(), perm_list))

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    acc = correct / len(val_df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[{name}] exact_match={acc:.4f} ({correct}/{len(val_df)})  "
          f"adj_swap_fail={adj}  other_fail={len(wrong_cases) - adj}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acc, correct, len(val_df), adj, len(wrong_cases) - adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=399)
    parser.add_argument("--val_csv", type=str,
                         default="/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/checkpoints/_val_raw.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    val_df = load_val(args.val_csv, args.n_samples)
    print(f"Val samples (No_ordering excluded): {len(val_df)}", flush=True)

    results = {}
    for name, path in MODELS:
        results[name] = evaluate(name, path, val_df, device)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print("비교 기준 (idea.md):")
    print("  Qwen2-VL-7B base zero-shot : 0.0550")
    print("  TPRU-7B zero-shot          : 0.2632 (105/399)")
    print("  Video-R1-7B zero-shot      : 0.2431 (97/399)")
    print("  Qwen2-VL v1 fine-tuned     : 0.5013 (200/399)")
    print("  v4 TPRU fine-tuned(epoch3) : 0.5336 (254/476, no_ord 포함 val)")
    print("-" * 60)
    for name, (acc, c, n, adj, other) in results.items():
        print(f"  {name:28s}: {acc:.4f} ({c}/{n})  adj_swap_fail={adj}  other_fail={other}")


if __name__ == "__main__":
    main()
