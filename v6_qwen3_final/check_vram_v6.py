"""
v6 VRAM 체크: Qwen3-VL-8B + v4 hard-negative 구조(그룹당 8샘플=32장 이미지)
해상도 x TRAIN_MINIBATCH 조합별로 forward+backward peak VRAM 측정.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=448)
parser.add_argument("--minibatch", type=int, default=8)  # 한 forward에 넣을 샘플 수(=이미지 4*minibatch장)
parser.add_argument("--attn", type=str, default="flash_attention_2")
args = parser.parse_args()

MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"
DATA_DIR = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")

def load_image(path, size=args.size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

from src.model import _get_model_class

processor = AutoProcessor.from_pretrained(MODEL_PATH)
ModelClass = _get_model_class(MODEL_PATH)
model = ModelClass.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=args.attn).to("cuda:0")
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_csv = pd.read_csv(DATA_DIR / "train.csv")
row = train_csv[train_csv["No_ordering"] == False].iloc[0]
sid = row["Id"]
img_dir = DATA_DIR / "train" / sid
files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
imgs = [load_image(str(img_dir / f)) for f in files]

PROMPT = ("Sentence: {s}\n\nThese 4 frames are presented in this exact order.\n"
          "Please carefully examine the changes between consecutive frames.\n"
          "Is this the correct chronological order of events?\nAnswer only with \"Yes\" or \"No\".")
SYSTEM = "You are a temporal ordering assistant."

def build_msg(imgs, sent):
    content = []
    for i, img in enumerate(imgs, 1):
        content.append({"type":"text","text":f"Frame {i}:"})
        content.append({"type":"image","image":img})
    content.append({"type":"text","text":PROMPT.format(s=sent)})
    return [{"role":"system","content":SYSTEM}, {"role":"user","content":content}]

# v4처럼 그룹당 TRAIN_MINIBATCH개 샘플(각 4장) 한 forward에
texts, imgs_list = [], []
for _ in range(args.minibatch):
    msg = build_msg(imgs, row["Sentence"])
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    texts.append(text)
    imgs_list.append(imgs)

torch.cuda.reset_peak_memory_stats()
inp = processor(text=texts, images=imgs_list, return_tensors="pt", padding=True).to("cuda:0")
print(f"input_ids shape: {inp['input_ids'].shape}")
if "pixel_values" in inp:
    print(f"pixel_values shape: {inp['pixel_values'].shape}")

out = model(**inp)
logits = out.logits[:, -1, :].float()
loss = logits.sum()
loss.backward()

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\n=== Peak VRAM (minibatch={args.minibatch}, {args.size}px, attn={args.attn}) === {peak:.2f} GB")
print(f"H100 80GB 여유: {80-peak:.2f} GB")
