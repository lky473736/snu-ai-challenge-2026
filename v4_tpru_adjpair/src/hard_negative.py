"""
v4 Hard Negative Generator — distribution-informed sampling + No_ordering x5 aug

Ordered: 1 positive (d=0) + SAMPLE_COUNTS negatives per group
  {d=1:2, d=2:1, d=3:1, d=4:1, d=5:1, d=6:1} = 7 neg → 8 total = 1 forward pass
  Distribution based on val error rates: d=1 1.70x, d=5 1.27x, d=3 1.16x over-represented
No_ordering: positive × 5 with diverse base permutations
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


# No_ordering augmentation pool: d >= 3 from identity (13 candidates)
_IDENTITY = (1, 2, 3, 4)
NO_ORD_AUG_POOL = [
    p for p in ALL_PERMS
    if p != _IDENTITY and kendall_dist(p, _IDENTITY) >= 3
]
NO_ORD_AUG_COUNT = 4  # original 1 + aug 4 = x5


def generate_hard_negative_samples(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Returns single DataFrame with columns:
      Id, Sentence, frame_order (str), label (float), dist (int),
      group_id (int), no_ordering (bool)

    Ordered: 1 pos + SAMPLE_COUNTS neg = 8 rows per group (1 forward pass)
    No_ordering: positive only × 5 (diverse base permutations, singleton groups)
    """
    random.seed(SEED)

    df_ord    = df_train[df_train["No_ordering"] == False].reset_index(drop=True)
    df_no_ord = df_train[df_train["No_ordering"] == True].reset_index(drop=True)

    records  = []
    group_id = 0

    # ── Ordered samples ──────────────────────────────────────
    for _, row in df_ord.iterrows():
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
                "no_ordering": False,
            })
        group_id += 1

    # ── No_ordering samples: positive only × 5 ───────────────
    for _, row in df_no_ord.iterrows():
        sid      = row["Id"]
        sentence = row["Sentence"]

        rng       = random.Random(hash(sid) % (2 ** 32))
        aug_perms = rng.sample(NO_ORD_AUG_POOL, NO_ORD_AUG_COUNT)
        base_perms = [_IDENTITY] + aug_perms

        for base_gt in base_perms:
            records.append({
                "Id":          sid,
                "Sentence":    sentence,
                "frame_order": str(list(base_gt)),
                "label":       1.0,
                "dist":        0,
                "group_id":    group_id,
                "no_ordering": True,
            })
            group_id += 1

    df_out = pd.DataFrame(records)
    n_per_group = sum(SAMPLE_COUNTS.values()) + 1
    print(f"  [data] ordered groups: {df_ord.shape[0]} ({n_per_group} samples each)  "
          f"no_ordering groups: {df_no_ord.shape[0]}×5={df_no_ord.shape[0]*5}  "
          f"total rows: {len(df_out)}")
    return df_out
