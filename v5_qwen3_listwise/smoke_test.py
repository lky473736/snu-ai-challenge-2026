import sys, os, argparse, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=672)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()

from config import DATA_DIR
from src.dataset import build_messages, load_image, answer_to_target
from src.model import load_model_and_processor, get_digit_comma_ids, generate_permutation
from src.train import build_batch_inputs

model, processor = load_model_and_processor()
digit_ids, comma_id = get_digit_comma_ids(processor)
device = torch.device("cuda:0")
model = model.to(device)

train_csv = pd.read_csv(DATA_DIR / "train.csv")
rows = train_csv[train_csv["No_ordering"] == False].sample(args.batch, random_state=0)

batch = {"images": [], "sentences": [], "targets": []}
for _, row in rows.iterrows():
    img_dir = DATA_DIR / "train" / row["Id"]
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    imgs = [load_image(str(img_dir / f)) for f in files]
    batch["images"].append(imgs)
    batch["sentences"].append(row["Sentence"])
    batch["targets"].append(answer_to_target(ast.literal_eval(row["Answer"])))

torch.cuda.reset_peak_memory_stats()
inputs = build_batch_inputs(processor, batch, device)
print("input_ids shape:", inputs["input_ids"].shape)
print("labels sample (row0):", inputs["labels"][0][-15:].tolist())
print("pixel_values shape:", inputs["pixel_values"].shape)

out = model(**inputs)
print("loss:", out.loss.item(), "finite:", torch.isfinite(out.loss).item())
out.loss.backward()

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n=== Peak VRAM (batch={args.batch} samples, {args.size}px) === {peak:.2f} GB")
print(f"H100 80GB 여유: {80-peak:.2f} GB")

# 제약 디코딩 스모크 테스트 (val 진단용 함수 그대로 사용)
model.eval()
row0 = rows.iloc[0]
img_dir = DATA_DIR / "train" / row0["Id"]
files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
imgs0 = [load_image(str(img_dir / f)) for f in files]
pred = generate_permutation(model, processor, imgs0, row0["Sentence"], digit_ids, comma_id, device, build_messages)
print(f"\n제약디코딩 결과: pred={pred}  gt={ast.literal_eval(row0['Answer'])}  유효한 순열={sorted(pred)==[1,2,3,4]}")
