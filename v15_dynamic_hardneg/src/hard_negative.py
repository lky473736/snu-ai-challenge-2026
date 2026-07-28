"""
v15 Hard Negative Generator — K는 v8/v14와 동일하게 7로 고정(그리드서치 결과 H100×4 + LoRA r=128 +
448px 조합에서는 K를 늘릴 VRAM 여유가 전혀 없음이 확인됨, grid_search_results.csv 참고. n_extra>0은
전부 OOM).

그래서 "추가"가 아니라 "교체" 방식으로 동적 하드 네거티브를 반영한다: SAMPLE_COUNTS의 d=1 슬롯 2개 중
**보너스 1개만** 뱅크의 하드 네거티브로 교체하고, 나머지(d1 최소 1개 + d2~d6 각 1개)는 항상 그대로
유지 — 매 그룹에서 d1~d6 전 구간 최소 커버리지가 100% 보장된다(v2 맹점 재발 방지 원칙 유지).
K 총합은 v14와 완전히 동일(7 negative + 1 positive = 8) → 추가 VRAM 불필요.
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


def sample_group(gt: tuple, sample_id: str = None, bank: dict = None, n_extra: int = None) -> list:
    """Returns list of (perm_list, dist) for one group. 총 K=7(v14와 동일) 고정.

    bank: {sample_id: (perm_tuple, score)} — train.py가 매 epoch 끝에 갱신하는 동적 하드 네거티브.
    n_extra: 사용 안 함(하위 호환용 인자, 무시). K 확장은 그리드서치로 VRAM 불가 확인되어 폐기.
    """
    by_dist = _perms_by_dist(gt)
    samples = [(list(gt), 0)]  # positive
    used = {gt}

    # d=1은 "필수 1개 + 보너스 1개"로 분리 — 보너스 슬롯만 뱅크로 교체 대상
    d1_pool = by_dist.get(1, [])
    d1_required = random.sample(d1_pool, min(1, len(d1_pool)))
    for p in d1_required:
        samples.append((list(p), 1))
        used.add(p)

    # 뱅크에 이 샘플의 하드 네거티브가 있으면 d1 보너스 슬롯을 그걸로 교체, 없으면 기존처럼 무작위
    bonus_perm = None
    if bank and sample_id in bank:
        hard_perm, _score = bank[sample_id]
        hard_perm = tuple(hard_perm)
        if hard_perm != gt and hard_perm not in used:
            bonus_perm = hard_perm
    if bonus_perm is None:
        remaining_d1 = [p for p in d1_pool if p not in used]
        if remaining_d1:
            bonus_perm = random.choice(remaining_d1)
    if bonus_perm is not None:
        d = kendall_dist(bonus_perm, gt)
        samples.append((list(bonus_perm), d))
        used.add(bonus_perm)

    # d2~d6은 v8/v14와 완전히 동일하게 각 1개씩 (최소 커버리지, 항상 보장)
    for dist in (2, 3, 4, 5, 6):
        n_take = SAMPLE_COUNTS.get(dist, 1)
        pool = [p for p in by_dist.get(dist, []) if p not in used]
        chosen = random.sample(pool, min(n_take, len(pool)))
        for p in chosen:
            samples.append((list(p), dist))
            used.add(p)

    return samples


def generate_hard_negative_samples(df_train: pd.DataFrame) -> pd.DataFrame:
    """참고용 정적 생성(스모크 테스트 등에서 사용) — 실제 학습은 GroupedTemporalDataset이
    __getitem__마다 sample_group()을 live 호출한다(v6.5 이후 방식과 동일)."""
    random.seed(SEED)
    records  = []
    group_id = 0

    for _, row in df_train.iterrows():
        gt       = tuple(ast.literal_eval(row["Answer"]))
        sid      = row["Id"]
        sentence = row["Sentence"]

        for perm, dist in sample_group(gt, sample_id=sid):
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
    print(f"  [data] groups: {df_train.shape[0]}  total rows: {len(df_out)}")
    return df_out
