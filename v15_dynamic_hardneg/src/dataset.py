import ast
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import DATA_DIR, MAX_IMAGE_SIZE
from src.hard_negative import sample_group

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
    """v15: v6.5부터 이어진 live hard-negative 재샘플링 + 동적 하드 네거티브 뱅크.

    self.bank: {sample_id: (perm_tuple, score)} — train.py가 매 epoch 끝에서 갱신하는
    in-memory dict. DataLoader worker는 num_workers>0 + persistent_workers=False(기본값)이라
    매 epoch iterator 생성 시 fork로 새로 뜨는데, 그 시점의 self.bank를 그대로 복사해가므로
    "메인 프로세스가 epoch 사이에 self.bank를 갱신 → 다음 epoch의 fork된 worker가 최신 값을
    물려받는다"는 흐름이 별도 파일 IO 없이 성립한다.
    """

    def __init__(self, df, n_extra: int = None):
        self.df = df.reset_index(drop=True)
        self.bank = {}
        self.n_extra = n_extra

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["Id"]
        gt  = tuple(ast.literal_eval(row["Answer"]))

        img_dir   = DATA_DIR / "train" / sid
        files     = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
        base_imgs = [load_image(str(img_dir / f)) for f in files]

        images_list, sentences, labels, dists, perms = [], [], [], [], []

        for perm, dist in sample_group(gt, sample_id=sid, bank=self.bank, n_extra=self.n_extra):
            inv = [0] * 4
            for inp_idx, t_pos in enumerate(perm):
                inv[t_pos - 1] = inp_idx
            imgs = [base_imgs[inv[t]] for t in range(4)]

            images_list.append(imgs)
            sentences.append(row["Sentence"])
            labels.append(1.0 if dist == 0 else 0.0)
            dists.append(int(dist))
            perms.append(tuple(perm))

        return {
            "sample_id":   sid,
            "images":      images_list,
            "sentences":   sentences,
            "labels":      labels,
            "dists":       dists,
            "perms":       perms,
            "group_size":  len(images_list),
        }


def collate_fn(batch):
    return {
        "sample_ids":   [b["sample_id"]   for b in batch],
        "images":       [b["images"]      for b in batch],
        "sentences":    [b["sentences"]   for b in batch],
        "labels":       [b["labels"]      for b in batch],
        "dists":        [b["dists"]       for b in batch],
        "perms":        [b["perms"]       for b in batch],
        "group_sizes":  [b["group_size"]  for b in batch],
    }
