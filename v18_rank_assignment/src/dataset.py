import ast
from itertools import combinations
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_DIR, MAX_IMAGE_SIZE

PAIRS = list(combinations(range(1, 5), 2))  # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

PROMPT_PAIR = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are shown in their original file order, which may NOT be the chronological order.\n"
    "Carefully examine the visual changes across all 4 frames together with the sentence.\n"
    "Does the event shown in Frame {i} happen before the event shown in Frame {j}, "
    "in the correct chronological order?\n"
    "Answer only with \"Yes\" or \"No\"."
)
SYSTEM = (
    "You are a temporal ordering assistant. "
    "Given 4 video frames shown in an arbitrary fixed order and a caption describing the full event, "
    "determine the relative chronological order between two specified frames."
)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_IMAGE_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def build_messages_pair(base_imgs, sentence: str, i: int, j: int):
    content = []
    for k, img in enumerate(base_imgs, 1):
        content.append({"type": "text", "text": f"Frame {k}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_PAIR.format(sentence=sentence, i=i, j=j)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


class PairwiseTemporalDataset(Dataset):
    """샘플 1개 -> 쌍별 질문 6개(4C2). 항상 원본 파일 순서로 4프레임 전부를 보여주고,
    두 프레임 중 어느 게 먼저인지만 Yes/No로 묻는다."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["Id"]
        ranks = ast.literal_eval(row["Answer"])  # ranks[k-1] = 프레임 k의 정답 시간위치(1~4)
        img_dir = DATA_DIR / "train" / sid
        files = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        images_list, sentences, pair_ijs, labels, adj_flags = [], [], [], [], []
        for (i, j) in PAIRS:
            images_list.append(base_imgs)
            sentences.append(row["Sentence"])
            pair_ijs.append((i, j))
            labels.append(1.0 if ranks[i - 1] < ranks[j - 1] else 0.0)
            adj_flags.append(1 if abs(ranks[i - 1] - ranks[j - 1]) == 1 else 0)

        return {"sid": sid, "images": images_list, "sentences": sentences, "pair_ijs": pair_ijs,
                "labels": labels, "adj_flags": adj_flags, "group_size": len(PAIRS)}


def collate_fn(batch):
    return {
        "sids": [b["sid"] for b in batch],
        "images": [b["images"] for b in batch],
        "sentences": [b["sentences"] for b in batch],
        "pair_ijs": [b["pair_ijs"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "adj_flags": [b["adj_flags"] for b in batch],
        "group_sizes": [b["group_size"] for b in batch],
    }
