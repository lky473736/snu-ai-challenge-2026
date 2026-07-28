"""
GroupedTemporalDataset
한 그룹 = 같은 Id의 positive 1개 + hard negative들
Qwen2-VL 네이티브 멀티이미지 입력 사용 (2x2 grid X)
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Any

from config import DATA_DIR, MAX_IMAGE_SIZE


PROMPT_TEMPLATE = (
    "Sentence: {sentence}\n\n"
    "These 4 frames are presented in this exact order.\n"
    "Please carefully examine the changes between consecutive frames.\n"
    "Is this the correct chronological order of events?\n"
    "Answer only with \"Yes\" or \"No\"."
)


def load_image(path: str, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


class GroupedTemporalDataset(Dataset):
    """
    __getitem__ 반환: 한 그룹 (같은 Id의 pos + hard_neg 묶음)
    {
        "groups": [
            {
                "images": [PIL, PIL, PIL, PIL],   # frame_order대로 정렬된 이미지
                "sentence": str,
                "label": int (1 or 0),
                "is_hard_negative": bool,
            },
            ...
        ],
        "id": str
    }
    """

    def __init__(self, df: pd.DataFrame, split: str = "train"):
        self.split = split
        # Id 기준으로 그룹화
        self.groups = []
        for sample_id, group_df in df.groupby("Id", sort=False):
            self.groups.append((sample_id, group_df.reset_index(drop=True)))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id, group_df = self.groups[idx]
        img_dir = DATA_DIR / self.split / sample_id

        # 원본 이미지 파일명 목록 (Input_1 ~ Input_4 순서)
        # frame_order[i] = Input_{i+1}의 temporal position
        # → temporal position 순서로 정렬: temporal_pos i → Input_j
        # frame_order = [2, 4, 1, 3] 이면 Input_1이 2번째, Input_2가 4번째 ...
        # "이 순서로 배열" = frame_order의 inverse mapping
        #   inverse[temporal_pos-1] = input_idx
        input_files = _get_input_files(img_dir, sample_id)

        items = []
        for _, row in group_df.iterrows():
            frame_order = row["frame_order"]
            if isinstance(frame_order, str):
                import ast
                frame_order = ast.literal_eval(frame_order)

            # frame_order[i] = Input_{i+1}의 temporal position (1-based)
            # 우리가 원하는 건: temporal 1번째 → 2번째 → 3번째 → 4번째 순서로 이미지 나열
            # inverse: temporal position t에 해당하는 input index 찾기
            inverse = [0] * 4
            for input_idx, temporal_pos in enumerate(frame_order):
                inverse[temporal_pos - 1] = input_idx  # 0-based

            # temporal 순서대로 이미지 로드
            try:
                images = [
                    load_image(str(img_dir / input_files[inverse[t]]))
                    for t in range(4)
                ]
            except Exception as e:
                print(f"[dataset] 이미지 로드 실패 {sample_id}: {e}")
                images = [Image.new("RGB", (224, 224)) for _ in range(4)]

            items.append({
                "images":           images,
                "sentence":         row["Sentence"],
                "label":            int(row["label"]),
                "is_hard_negative": bool(row["is_hard_negative"]),
                "frame_order":      frame_order,
            })

        return {"id": sample_id, "items": items}


def _get_input_files(img_dir: Path, sample_id: str) -> List[str]:
    """img_dir에서 Input_1~4에 해당하는 파일명 순서대로 반환 (알파벳 정렬)"""
    files = sorted([f.name for f in img_dir.iterdir() if f.suffix == ".jpg"])
    assert len(files) == 4, f"{sample_id}: 이미지 수 != 4 ({files})"
    return files


def build_messages(images: List[Image.Image], sentence: str) -> List[Dict]:
    """Qwen2-VL chat template용 메시지 구성 (4장 별도 입력)"""
    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "text",  "text": f"Frame {i}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_TEMPLATE.format(sentence=sentence)})

    return [
        {
            "role": "system",
            "content": (
                "You are a temporal ordering assistant. "
                "Given 4 video frames in a specific order and a caption, "
                "determine if the frames are in the correct chronological order."
            ),
        },
        {"role": "user", "content": content},
    ]


def collate_fn(batch: List[Dict]) -> Dict:
    """DataLoader용 collate: 배치 내 모든 items를 flat하게 모음"""
    all_images    = []
    all_sentences = []
    all_labels    = []
    all_is_hard   = []
    all_ids       = []
    group_sizes   = []

    for group in batch:
        items = group["items"]
        group_sizes.append(len(items))
        for item in items:
            all_images.append(item["images"])
            all_sentences.append(item["sentence"])
            all_labels.append(item["label"])
            all_is_hard.append(item["is_hard_negative"])
            all_ids.append(group["id"])

    return {
        "images":      all_images,      # List[List[PIL]] (N × 4장)
        "sentences":   all_sentences,   # List[str]
        "labels":      torch.tensor(all_labels, dtype=torch.float),
        "is_hard":     all_is_hard,
        "ids":         all_ids,
        "group_sizes": group_sizes,
    }
