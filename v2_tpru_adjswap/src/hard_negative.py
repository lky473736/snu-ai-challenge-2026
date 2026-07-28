"""
Hard negative 생성: 정답 순서에서 인접 위치(0-1, 1-2, 2-3)를 swap한 3개 negative 사용
(v1과 동일한 adjacent-swap 로직 + No_ordering 샘플도 포함하여 학습)
"""

import ast
import pandas as pd
from typing import List


def _adjacent_swaps(order: List[int]) -> List[List[int]]:
    """[0-1], [1-2], [2-3] 위치를 각각 swap한 3가지 반환"""
    swaps = []
    for i in range(len(order) - 1):
        neg = order.copy()
        neg[i], neg[i + 1] = neg[i + 1], neg[i]
        swaps.append(neg)
    return swaps  # 길이 3


def generate_hard_negative_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    train.csv → positive 1개 + adjacent swap negative 3개 = 4개/group
    No_ordering=True 그룹도 positive=[1,2,3,4]로 포함 (skip하지 않음)
    반환 컬럼: Id, Sentence, frame_order (list), label (1/0), is_hard_negative (bool)
    """
    rows = []
    for _, row in df.iterrows():
        pos_order = ast.literal_eval(row["Answer"])

        rows.append({
            "Id":               row["Id"],
            "Sentence":         row["Sentence"],
            "frame_order":      pos_order,
            "label":            1,
            "is_hard_negative": False,
        })

        for neg_order in _adjacent_swaps(pos_order):
            rows.append({
                "Id":               row["Id"],
                "Sentence":         row["Sentence"],
                "frame_order":      neg_order,
                "label":            0,
                "is_hard_negative": True,
            })

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[hard_negative] positive: {(result['label']==1).sum()}, "
          f"hard_neg: {(result['label']==0).sum()}, "
          f"total: {len(result)}")
    return result
