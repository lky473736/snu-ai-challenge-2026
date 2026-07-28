"""
v6 Hard Negative Generator — distribution-informed sampling

모든 샘플(Ordered + No_ordering)을 동일하게 취급: 1 positive(d=0) + SAMPLE_COUNTS negatives.
  {d=1:2, d=2:1, d=3:1, d=4:1, d=5:1, d=6:1} = 7 neg -> 8 total = 1 forward pass
  Distribution based on val error rates: d=1 1.70x, d=5 1.27x, d=3 1.16x over-represented

No_ordering 샘플도 정답 라벨이 [1,2,3,4]로 고정되어 있고 채점이 exact-match라서,
"다른 순서도 의미상 맞다"는 가정(구버전 x5 증강)보다 Ordered와 동일하게 hard negative로
대조 학습시키는 게 채점 기준과 더 일치함. 특수 분기/증강 로직 불필요 + 그룹 수 대폭 감소
(no_ordering 그룹 7005개 -> 1401개, 80% 감소).
"""

import ast
import random
from itertools import permutations

import pandas as pd

from config import SEED, SAMPLE_COUNTS

ALL_PERMS = list(permutations([1, 2, 3, 4]))


def kendall_dist(p, q):
    rank = {v: i for i, v in enumerate(q)}
    arr  = [rank[v] for v in p]
    inv  = 0
    n    = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def _perms_by_dist(gt: tuple) -> dict:
    by_dist = {}
    for p in ALL_PERMS:
        if p == gt:
            continue
        d = kendall_dist(p, gt)
        by_dist.setdefault(d, []).append(p)
    return by_dist


def sample_group(gt: tuple) -> list:
    """Returns list of (perm_list, dist) for one group."""
    by_dist = _perms_by_dist(gt)
    samples = [(list(gt), 0)]  # positive

    for dist, n_take in SAMPLE_COUNTS.items():
        pool = by_dist.get(dist, [])
        chosen = random.sample(pool, min(n_take, len(pool)))
        for p in chosen:
            samples.append((list(p), dist))

    return samples


def generate_hard_negative_samples(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Returns single DataFrame with columns:
      Id, Sentence, frame_order (str), label (float), dist (int), group_id (int)

    모든 행(Ordered + No_ordering) 동일하게 1 pos + SAMPLE_COUNTS neg = 8 rows/group.
    No_ordering도 Answer가 항상 [1,2,3,4]이므로 gt로 그대로 사용.
    """
    random.seed(SEED)
    records  = []
    group_id = 0

    for _, row in df_train.iterrows():
        gt       = tuple(ast.literal_eval(row["Answer"]))
        sid      = row["Id"]
        sentence = row["Sentence"]

        for perm, dist in sample_group(gt):
            records.append({
                "Id":          sid,
                "Sentence":    sentence,
                "frame_order": str(perm),
                "label":       1.0 if dist == 0 else 0.0,
                "dist":        dist,
                "group_id":    group_id,
            })
        group_id += 1

    df_out = pd.DataFrame(records)
    n_per_group = sum(SAMPLE_COUNTS.values()) + 1
    print(f"  [data] groups: {df_train.shape[0]} ({n_per_group} samples each)  "
          f"total rows: {len(df_out)}")
    return df_out
