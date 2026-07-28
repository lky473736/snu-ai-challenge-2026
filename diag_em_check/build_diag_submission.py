"""
EM vs 부분점수(쌍순서) 채점 판별 — 진단 제출 생성 (GPU 불필요).

myunhh 팀 방법론과 동일한 원리, 코드는 우리 컨벤션(1-based Answer)으로 새로 구현:
margins.csv(margin_inference.py 출력)에서 가장 확신도(margin) 높은 K개만 골라 인접 스왑
1개를 적용하고, 나머지는 원래 예측 그대로 둔 제출 파일을 만든다.

  EM 채점이면    실측 ΔLB ≈ -K/N * 100   (그 K건이 전부 정답->오답으로 뒤집힌다고 가정)
  쌍순서 채점이면 실측 ΔLB ≈ -K/(6N) * 100 (스와프 1개당 쌍순서 손실은 정확히 1/6)

기존 챔피언 LB(v14_listwise_softmax, 0.90052)에서 이 파일 제출 후 실측 LB를 빼서
두 예측값과 비교하면 된다. Kaggle 제출은 이 스크립트 밖에서 사용자가 직접 진행할 것
(제출 슬롯 1일 2회 제한 — 자동 제출 안 함).
"""

import argparse
import ast

import pandas as pd


def rank_to_order(rank):
    """rank[k-1] = Input_k의 정답 위치(1-based) -> order[p-1] = p번째로 오는 Input(1-based)."""
    order = [0] * 4
    for inp_idx0, pos in enumerate(rank):
        order[pos - 1] = inp_idx0 + 1
    return order


def order_to_rank(order):
    rank = [0] * 4
    for pos0, inp_idx in enumerate(order):
        rank[inp_idx - 1] = pos0 + 1
    return rank


def adjacent_swap(rank, swap_index=0):
    """order(시간순) 공간에서 인접한 두 위치(swap_index, swap_index+1)를 교환."""
    order = rank_to_order(rank)
    order[swap_index], order[swap_index + 1] = order[swap_index + 1], order[swap_index]
    return order_to_rank(order)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margins", default="margins.csv")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--swap-index", type=int, default=0, choices=[0, 1, 2],
                     help="order 공간에서 교환할 인접 위치 쌍 (0=1~2번째, 1=2~3번째, 2=3~4번째)")
    ap.add_argument("--out", default="submission_diag_swap60.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.margins)
    n = len(df)
    if not (0 < args.k <= n):
        raise SystemExit(f"--k={args.k}는 (0, {n}] 범위여야 함")

    df = df.sort_values("margin", ascending=False).reset_index(drop=True)
    swap_ids = set(df["Id"].iloc[: args.k])

    rows = []
    for _, r in df.iterrows():
        rank = ast.literal_eval(r["Answer"])
        if r["Id"] in swap_ids:
            rank = adjacent_swap(rank, swap_index=args.swap_index)
        rows.append({"Id": r["Id"], "Answer": str(rank)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    em_delta = -args.k / n * 100
    pair_delta = -args.k / (6 * n) * 100
    print(f"저장: {args.out}  ({len(out_df)}행, swap 적용 {len(swap_ids)}건)")
    print(f"\n현재 챔피언 LB(v14, 0.90052)에서 이 파일 제출 후:")
    print(f"  EM 채점 가설이면   실측 ΔLB ≈ {em_delta:+.2f}pp  (예상 LB ≈ {0.90052 + em_delta/100:.5f})")
    print(f"  쌍순서 채점 가설이면 실측 ΔLB ≈ {pair_delta:+.2f}pp  (예상 LB ≈ {0.90052 + pair_delta/100:.5f})")
    print(f"\n실제 제출 후 나온 LB를 위 두 값과 비교해서 어느 쪽에 가까운지 확인할 것.")


if __name__ == "__main__":
    main()
