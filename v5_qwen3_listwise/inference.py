"""
v5 최종 추론: test.csv 전체에 대해 제약 디코딩으로 순열 직접 생성.
샘플당 forward 1회(+짧은 생성 7토큰)뿐이라 24-permutation 방식보다 훨씬 빠름.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel

from config import DATA_DIR, MODEL_PATH
from src.dataset import build_messages, load_image
from src.model import _get_model_class, get_digit_comma_ids, generate_permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="LoRA adapter dir (e.g. checkpoints/best_v5)")
    parser.add_argument("--out", type=str, default="submission_v5.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.ckpt)
    ModelClass = _get_model_class(MODEL_PATH)
    base_model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base_model, args.ckpt).to(device)
    model.eval()
    digit_ids, comma_id = get_digit_comma_ids(processor)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sid = row["Id"]
        img_dir = DATA_DIR / "test" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        imgs = [load_image(str(img_dir / f)) for f in files]
        try:
            pred = generate_permutation(model, processor, imgs, row["Sentence"], digit_ids, comma_id, device, build_messages)
            if len(pred) != 4 or sorted(pred) != [1, 2, 3, 4]:
                pred = [1, 2, 3, 4]  # 방어적 fallback (이론상 발생 안 해야 함)
        except Exception as e:
            print(f"[warn] {sid}: {e}")
            pred = [1, 2, 3, 4]
        rows.append({"Id": sid, "Answer": str(pred)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"저장: {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
