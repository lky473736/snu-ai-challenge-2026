import ast
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_DIR, MAX_IMAGE_SIZE

PROMPT = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Please carefully examine the changes between consecutive frames.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)

SYSTEM = (
    "You are a temporal ordering assistant. "
    "Given video frames in a specific order and a caption, "
    "determine if the frames are in the correct chronological order."
)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_messages(images, sentence: str, n_frames: int = 4):
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT.format(sentence=sentence)})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content}]


class GroupedTemporalDataset(Dataset):
    def __init__(self, df):
        self.groups = [grp.reset_index(drop=True) for _, grp in df.groupby("group_id")]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        grp    = self.groups[idx]
        row0   = grp.iloc[0]
        sid    = row0["Id"]

        img_dir   = DATA_DIR / "train" / sid
        files     = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        images_list, sentences, labels, dists = [], [], [], []
        no_ordering = bool(row0.get("no_ordering", False))

        for _, row in grp.iterrows():
            perm = ast.literal_eval(row["frame_order"])
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(perm):
                inv[t_pos - 1] = inp_idx
            imgs = [base_imgs[inv[t]] for t in range(4)]

            images_list.append(imgs)
            sentences.append(row["Sentence"])
            labels.append(float(row["label"]))
            dists.append(int(row["dist"]))

        return {
            "images":      images_list,
            "sentences":   sentences,
            "labels":      labels,
            "dists":       dists,
            "group_size":  len(grp),
            "no_ordering": no_ordering,
        }


def collate_fn(batch):
    return {
        "images":       [b["images"]      for b in batch],
        "sentences":    [b["sentences"]   for b in batch],
        "labels":       [b["labels"]      for b in batch],
        "dists":        [b["dists"]       for b in batch],
        "group_sizes":  [b["group_size"]  for b in batch],
        "no_orderings": [b["no_ordering"] for b in batch],
    }
