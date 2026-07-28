import ast
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_DIR, MAX_IMAGE_SIZE

SYSTEM = (
    "You are a temporal ordering assistant. You are given 4 frames from a video, "
    "shown in an arbitrary (file-order) sequence labeled Frame 1..4, and a caption "
    "describing the events across the frames in chronological order. "
    "Determine the chronological rank (1=earliest, 4=latest) of each shown frame."
)

PROMPT = (
    "Sentence: {sentence}\n\n"
    "Frame 1, Frame 2, Frame 3, Frame 4 are shown above in file order (not necessarily "
    "chronological order). For each of Frame 1..4, output its chronological rank (1-4).\n"
    "Answer with exactly 4 digits separated by commas, e.g. \"3,1,2,4\" means "
    "Frame 1 is 3rd, Frame 2 is 1st, Frame 3 is 2nd, Frame 4 is 4th chronologically.\n"
    "Answer:"
)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_messages(images, sentence: str):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": content}]


def answer_to_target(answer_parsed):
    """[3,1,2,4] (Input_i의 순위) -> 그대로 텍스트 타겟 '3,1,2,4'"""
    return ",".join(str(x) for x in answer_parsed)


class ListwiseDataset(Dataset):
    """그룹/hard-negative 없이 train.csv 각 행이 곧 학습 샘플 1개."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["Id"]
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        imgs = [load_image(str(img_dir / f)) for f in files]
        answer = ast.literal_eval(row["Answer"])
        target = answer_to_target(answer)
        return {
            "images": imgs,
            "sentence": row["Sentence"],
            "target": target,
        }


def collate_fn(batch):
    return {
        "images":   [b["images"]   for b in batch],
        "sentences":[b["sentence"] for b in batch],
        "targets":  [b["target"]   for b in batch],
    }
