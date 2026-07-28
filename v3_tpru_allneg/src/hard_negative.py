"""
Hard negative 생성: 정답 순서를 제외한 23개 모든 wrong permutation을 negative로 사용
"""

import ast
import pandas as pd
from itertools import permutations
from typing import List

ALL_PERMS = list(permutations([1, 2, 3, 4]))  # 24가지


def generate_hard_negative_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    train.csv → positive 1개 + 모든 wrong permutation 23개 = 24개/group
    반환 컬럼: Id, Sentence, frame_order (list), label (1/0), is_hard_negative (bool)
    """
    rows = []
    for _, row in df.iterrows():
        pos_order = ast.literal_eval(row["Answer"])
        pos_tuple = tuple(pos_order)

        rows.append({
            "Id":               row["Id"],
            "Sentence":         row["Sentence"],
            "frame_order":      pos_order,
            "label":            1,
            "is_hard_negative": False,
        })

        for perm in ALL_PERMS:
            if perm == pos_tuple:
                continue
            rows.append({
                "Id":               row["Id"],
                "Sentence":         row["Sentence"],
                "frame_order":      list(perm),
                "label":            0,
                "is_hard_negative": True,
            })

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[hard_negative] positive: {(result['label']==1).sum()}, "
          f"hard_neg: {(result['label']==0).sum()}, "
          f"total: {len(result)}")
    return result
