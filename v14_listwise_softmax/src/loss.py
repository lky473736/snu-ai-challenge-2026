"""
ListwiseSoftmaxLoss — 8-way joint softmax cross-entropy (v14, v8 대비 loss만 교체)

배경: v4~v10이 쓴 AdaptiveDistanceLoss는 distance(d=1..6)별로 독립적인 pairwise margin
(softplus(neg-pos))을 구한 뒤, EMA로 추적한 난이도를 softmax 가중치로 사후 결합하는 방식이었다.
이 버전은 그룹(1 positive + 7 hard negative, hard_negative.py의 SAMPLE_COUNTS로 생성된 구성 그대로)
전체를 하나의 categorical cross-entropy로 다룬다:

  L_g = -log( exp(s(y*)) / sum_{y in group} exp(s(y)) )

이는 Plackett-Luce top-1 모델(ListMLE의 top-1 특수 케이스)과 동치이며, gradient가
  dL/ds(y) = softmax(y) - 1[y=y*]
로 정확히 계산되어, "지금 모델이 헷갈려서 높은 점수를 준 negative일수록 더 강하게 눌리는"
self-hardening 효과를 EMA 없이 매 스텝 즉시 낸다.

v14는 distance 가중치를 의도적으로 넣지 않는다(균등) — AdaptiveDistanceLoss와의 차이를
"joint softmax 정규화 방식" 하나로만 격리하기 위함. distance 가중치를 다시 넣는 실험은
이 결과를 본 뒤 별도 버전(v15)으로 분리한다.

`per_dist_diag`는 loss 계산에 관여하지 않는 순수 로깅용 진단값 — distance d의 negative들이
평균적으로 받은 softmax 확률(모델이 실제로 얼마나 헷갈려하는지, EMA 없이 즉석 관측).
train.py의 dist_weights.csv 로깅 코드와의 호환을 위해 AdaptiveDistanceLoss와 동일한
(loss, per_dist_dict, weights_tensor) 튜플 형태와 `.ema` 속성을 유지한다(값 자체는 미사용 placeholder).
"""

import torch


class ListwiseSoftmaxLoss:
    def __init__(self, *args, **kwargs):
        # AdaptiveDistanceLoss(ema_alpha=, temperature=) 호출부와 시그니처 호환을 위해 **kwargs 허용,
        # 실제로는 아무 파라미터도 쓰지 않음.
        self.ema  = torch.zeros(6)   # 로깅 호환용 placeholder (dist_weights.csv 컬럼 유지)
        self.step = 0

    def __call__(
        self,
        logits:      torch.Tensor,  # (N,) flat, N = sum(group_sizes)
        dists:       list,          # flat list[int], len=N, d=0(positive)..6
        group_sizes: list,          # list[int], 보통 전부 8
    ):
        offset = 0
        losses = []
        neg_prob_by_dist = {d: [] for d in range(1, 7)}

        for gs in group_sizes:
            g_logits = logits[offset: offset + gs]
            g_dists  = dists[offset: offset + gs]
            offset  += gs

            pos_idxs = [i for i, d in enumerate(g_dists) if d == 0]
            if not pos_idxs:
                continue
            pos_idx = pos_idxs[0]

            log_probs = torch.log_softmax(g_logits, dim=0)
            losses.append(-log_probs[pos_idx])

            probs = log_probs.exp().detach()
            for i, d in enumerate(g_dists):
                if d > 0:
                    neg_prob_by_dist[d].append(probs[i].item())

        total = torch.stack(losses).mean()

        per_dist_diag = {
            d: (sum(v) / len(v) if v else float("nan"))
            for d, v in neg_prob_by_dist.items()
        }
        # 균등 placeholder (실제 가중치 없음 — 로깅 코드 호환용)
        uniform_weights = torch.full((6,), 1.0 / 6.0)

        self.step += 1
        return total, per_dist_diag, uniform_weights
