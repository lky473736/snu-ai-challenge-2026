"""
Hard negative 생성: 정답 순서에서 인접 위치를 swap한 샘플 생성
"""

import ast
import pandas as pd
from itertools import combinations
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
    train.csv → positive + hard negative 확장 DataFrame 반환
    반환 컬럼: Id, Sentence, frame_order (list), label (1/0), is_hard_negative (bool)
    """
    rows = []
    for _, row in df.iterrows():
        if row["No_ordering"]:
            continue  # [1,2,3,4] 고정이라 hard neg 의미 없음

        pos_order = ast.literal_eval(row["Answer"])

        # Positive
        rows.append({
            "Id":              row["Id"],
            "Sentence":        row["Sentence"],
            "frame_order":     pos_order,
            "label":           1,
            "is_hard_negative": False,
        })

        # Hard Negatives (adjacent swaps)
        for neg_order in _adjacent_swaps(pos_order):
            rows.append({
                "Id":              row["Id"],
                "Sentence":        row["Sentence"],
                "frame_order":     neg_order,
                "label":           0,
                "is_hard_negative": True,
            })

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[hard_negative] positive: {(result['label']==1).sum()}, "
          f"hard_neg: {(result['label']==0).sum()}, "
          f"total: {len(result)}")
    return result
