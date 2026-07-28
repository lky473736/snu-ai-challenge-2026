"""CoT 테스트와 정확히 같은 10개 샘플(random_state=42)에 원래 방식(CoT 없음)을 돌려서 공정 비교."""
import sys, os, ast, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel

from config import DATA_DIR, MODEL_PATH, CKPT_DIR
from src.dataset import build_messages, load_image
from src.model import _get_model_class, get_yes_no_token_ids, forward_logit

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    ckpt = str(CKPT_DIR / "best_v4")
    processor = AutoProcessor.from_pretrained(ckpt)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_model, ckpt).to(device)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    val_df = pd.read_csv(CKPT_DIR / "_val_raw.csv").sample(args.n_samples, random_state=42).reset_index(drop=True)

    correct = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        sid = row["Id"]
        gt = ast.literal_eval(row["Answer"])
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        def reorder(order):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(order):
                inv[t_pos - 1] = inp_idx
            return [base_imgs[inv[t]] for t in range(4)]

        scores = []
        for perm in ALL_PERMS:
            imgs = reorder(list(perm))
            msgs = build_messages(imgs, row["Sentence"])
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            with torch.no_grad():
                inp = processor(text=[text], images=[imgs], return_tensors="pt").to(device)
                s = forward_logit(model, processor, inp, yes_id, no_id)
            scores.append((s.item(), list(perm)))

        best = max(scores, key=lambda x: x[0])[1]
        ok = best == gt
        correct += ok
        print(f"[sample {idx}] gt={gt} pred={best} ok={ok}")

    acc = correct / len(val_df)
    print(f"\n[원래 방식(CoT 없음), n={len(val_df)}] exact_match={acc:.4f} ({correct}/{len(val_df)})")


if __name__ == "__main__":
    main()
