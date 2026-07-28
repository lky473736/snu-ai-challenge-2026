"""
Qwen3-VL-30B-A3B-Instruct (MoE) vs Qwen3-VL-32B-Instruct (Dense) — 4bit 양자화 zero-shot 비교.
목적: (1) zero-shot 정확도가 8B 대비 확연히 좋은지, (2) 4bit 양자화 시 VRAM이 24GB(3090) 안에
들어가는지, (3) 실제 배치=1, 24-순열 시나리오에서 속도(s/sample)가 어느 정도인지 실측.

비교 기준 (동일 val, 동일 24-permutation 방식):
  Qwen3-VL-8B-Instruct zero-shot : 32.83% (131/399)
"""
import sys, os, gc, argparse, ast, time
sys.path.insert(0, "/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair")

import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig

from src.dataset import build_messages, load_image, DATA_DIR
from src.model import get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))

MODELS = [
    ("Qwen3-VL-30B-A3B-Instruct(MoE,4bit)", "/data/gyuyeonlim/models/Qwen3-VL-30B-A3B-Instruct"),
    ("Qwen3-VL-32B-Instruct(Dense,4bit)",   "/data/gyuyeonlim/models/Qwen3-VL-32B-Instruct"),
]


def get_model_class(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path)
    mt = getattr(cfg, "model_type", "")
    if mt == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
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
    print(f"\n{'='*60}\nLoading {name}\n{'='*60}", flush=True)
    ModelClass = get_model_class(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    t_load0 = time.time()
    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = ModelClass.from_pretrained(
                model_path, quantization_config=bnb_config,
                torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
                device_map={"": device},
            )
            print(f"  -> {attn_impl} OK (load {time.time()-t_load0:.1f}s)", flush=True)
            break
        except Exception as e:
            print(f"  -> {attn_impl} skipped: {e}", flush=True)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    torch.cuda.reset_peak_memory_stats(device)
    weight_vram = torch.cuda.memory_allocated(device) / 1e9
    print(f"  가중치 로드 후 VRAM: {weight_vram:.2f} GB", flush=True)

    correct, wrong_cases = 0, []
    t0 = time.time()
    n_forward_calls = 0
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
            n_forward_calls += 1

        best = max(scores, key=lambda x: x[0])[1]
        if best == gt:
            correct += 1
        else:
            diffs = [i for i in range(4) if best[i] != gt[i]]
            t = "adj_swap" if (len(diffs) == 2 and abs(diffs[0] - diffs[1]) == 1) else "other"
            wrong_cases.append({"id": sample_id, "gt": gt, "pred": best, "type": t})

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
    acc = correct / len(val_df)
    adj = sum(1 for w in wrong_cases if w["type"] == "adj_swap")
    print(f"\n[{name}] exact_match={acc:.4f} ({correct}/{len(val_df)})  "
          f"adj_swap_fail={adj}  other_fail={len(wrong_cases) - adj}", flush=True)
    print(f"  속도: {elapsed:.1f}s 총, {elapsed/len(val_df):.3f}s/sample, "
          f"{elapsed/n_forward_calls*1000:.1f}ms/forward(24개 중 1개)", flush=True)
    print(f"  Peak VRAM: {peak_vram:.2f} GB  (3090 24GB 기준: "
          f"{'✅ 들어감' if peak_vram < 22 else '⚠️ 빡빡함/초과 위험'})", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return dict(acc=acc, correct=correct, n=len(val_df), adj=adj, other=len(wrong_cases)-adj,
                sec_per_sample=elapsed/len(val_df), peak_vram=peak_vram, weight_vram=weight_vram)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--val_csv", type=str,
                         default="/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/checkpoints/_val_raw.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    val_df = load_val(args.val_csv, args.n_samples)
    print(f"Val samples (No_ordering excluded): {len(val_df)}", flush=True)

    results = {}
    for name, path in MODELS:
        if not os.path.exists(os.path.join(path, "config.json")):
            print(f"[skip] {name}: 아직 다운로드 안 됨 ({path})")
            continue
        results[name] = evaluate(name, path, val_df, device)

    print(f"\n{'='*60}\nSUMMARY (비교 기준: Qwen3-VL-8B-Instruct zero-shot 32.83%, bf16, VRAM~18GB)\n{'='*60}")
    for name, r in results.items():
        print(f"  {name}")
        print(f"    acc={r['acc']:.4f} ({r['correct']}/{r['n']})  adj_swap_fail={r['adj']}  other_fail={r['other']}")
        print(f"    {r['sec_per_sample']:.3f}s/sample  peak_vram={r['peak_vram']:.2f}GB  weight_vram={r['weight_vram']:.2f}GB")


if __name__ == "__main__":
    main()
