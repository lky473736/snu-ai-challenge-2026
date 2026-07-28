"""
AdaptiveDistanceNounLoss — per-Kendall-distance (기존) + per-n_nouns-bucket (신규) 이중 적응 가중치.

배경 (EDA.md §10-2): sent_len/max_dep_depth/n_verbs/temporal_words는 전부 n_nouns와의
다중공선성(r=0.36~0.89)에서 온 부산물이었고, 로지스틱 회귀·bootstrap 95% CI·split-half
재현성 검증까지 통과한 **유일하게 독립적인 정확도 예측 신호는 n_nouns**였다
(구간별 정확도 29.4%→83.3%, 거의 3배 차이).

── 거리(distance) 축 (기존, 그룹 내부 — 어떤 negative를 더 세게 벌줄지) ──
  L_d = mean softplus(neg_score_d - pos_score)
  ema_d      ← alpha * ema_d + (1-alpha) * L_d
  w_d        = softmax(ema_d * temp_dist)
  group_loss = Σ_d w_d * L_d   (그룹 안에 존재하는 거리들의 가중 평균)

── n_nouns 축 (신규, 그룹 전체 단위 — 어떤 그룹에 더 집중할지) ──
  ★ 1차 설계(폐기) `w_k = K·softmax(ema_k·temp)`의 버그: "5개 버킷의 **단순** 평균이 1.0"만
  보장했지, 실제 학습 데이터에서 버킷마다 등장 빈도(p_k)가 다르다는 걸 놓쳤다. 버킷0(명사 적음,
  실측 28.5%로 가장 흔함)이 계속 업웨이트되면서 **빈도가중 평균이 1.0을 초과**, loss 스케일이
  실제로 계속 떠올랐음을 v7 1차 시도 로그에서 직접 확인함 (v6.5 대비 step300~450 loss가
  0.44~0.45 → 0.50~0.53로 눈에 띄게 상승).

  ★ 2차 설계(현재, 수학적으로 정정): 실제 버킷 빈도 p_k(train 9,535개 n_nouns 분포에서 계산,
  고정 상수)를 명시적으로 반영해 "빈도가중 평균이 항상 정확히 1.0"이 되도록 재정규화한다.

    ema_noun_k ← alpha * ema_noun_k + (1-alpha) * group_loss   (그룹이 속한 버킷 k에 대해)
    ema_bar    = Σ_k p_k * ema_noun_k                           (빈도가중 평균 난이도)
    raw_k      = exp(temp_noun * (ema_noun_k - ema_bar))        (평균 대비 상대적 난이도, 균등시 raw_k=1)
    Z          = Σ_k p_k * raw_k                                (빈도가중 정규화 상수)
    w_noun_k   = raw_k / Z

  증명:
    (a) 모든 버킷이 동일 난이도(ema_noun_k = ema_bar 전부)면 raw_k=exp(0)=1 → Z=Σp_k·1=1 → w_k=1 전부.
        (재가중치가 전혀 없는 원래 상태로 정확히 환원됨)
    (b) 임의의 ema 분포에서도 Σ_k p_k·w_k = Σ_k p_k·raw_k/Z = Z/Z = 1 이 **항상 정확히** 성립.
        (버킷 빈도가 아무리 불균등해도 실제 학습 샘플 기준 평균 가중치는 절대 1에서 벗어나지 않음
         → loss 스케일이 재가중치 도입 이전과 동일하게 유지됨, 1차 설계의 버그 원천 차단)

  final_group_loss = w_noun_k(detached) * group_loss

두 축 모두 torch.no_grad()로 계산되는 EMA 통계치이지 gradient로 학습되는 파라미터가 아니다.
(idea.md §5-5: 진짜 gradient-learnable 가중치는 ∂L/∂γ ≥ 0이 항상 성립해 "가중치를 꺼서 loss를
속이는" 방향으로 최적화되는 수학적 버그가 있음이 확인됨 — EMA는 gradient를 안 받으므로 이 문제가
원천적으로 발생하지 않음.)

두 축은 서로 직교(orthogonal)한다: n_noun 가중치는 그룹 전체에 곱해지는 상수라 그룹 내부의
거리별 상대 가중치 배분에는 영향을 주지 않고, 그룹 간 loss 스케일만 조절한다.
"""

import torch
import torch.nn.functional as F

N_NOUN_BUCKETS = 5
# EDA.md §10-2 근거 (n_nouns 분포: mean=7.2, std=3.5, quartile 4/7/10) — quintile 근사 경계값
NOUN_BUCKET_EDGES = [4, 6, 8, 10]  # bucket0:<=4, bucket1:5-6, bucket2:7-8, bucket3:9-10, bucket4:11+

# train.csv 9,535개 전체(precompute_n_nouns.py) 기준 실측 버킷 빈도 — 고정 상수.
# 빈도가중 정규화(Z)에 반드시 필요 (버킷별 등장 확률이 균등하지 않음: bucket0이 28.5%로 최다).
NOUN_BUCKET_FREQ = torch.tensor([0.285160, 0.159413, 0.165705, 0.189198, 0.200524])
assert abs(NOUN_BUCKET_FREQ.sum().item() - 1.0) < 1e-4


def noun_bucket(n_nouns: int) -> int:
    for i, edge in enumerate(NOUN_BUCKET_EDGES):
        if n_nouns <= edge:
            return i
    return len(NOUN_BUCKET_EDGES)


class AdaptiveDistanceNounLoss:
    def __init__(self, ema_alpha: float = 0.99, temperature: float = 1.0, noun_temperature: float = 1.0):
        self.ema_alpha        = ema_alpha
        self.temperature      = temperature
        self.noun_temperature = noun_temperature
        self.ema       = torch.ones(6)               # distance축, d=1..6 -> idx 0..5
        self.ema_noun  = torch.ones(N_NOUN_BUCKETS)   # n_nouns축
        self.noun_freq = NOUN_BUCKET_FREQ.clone()
        self.step = 0

    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.ema * self.temperature, dim=0)  # (6,), sum=1

    def get_noun_weights(self) -> torch.Tensor:
        """빈도가중 평균이 항상 정확히 1.0이 되도록 정규화된 버킷별 가중치. (수식 유도는 모듈 docstring 참고)"""
        freq = self.noun_freq.to(self.ema_noun.device)
        ema_bar = (freq * self.ema_noun).sum()
        raw = torch.exp(self.noun_temperature * (self.ema_noun - ema_bar))
        Z = (freq * raw).sum()
        return raw / Z  # (5,), Σ freq_k * w_k == 1 항상 성립

    def __call__(
        self,
        logits:       torch.Tensor,  # (N,) 배치 내 전체 그룹의 flat logits
        dists:        list,          # flat list[int], len=N, d=0..6
        group_sizes:  list,          # list[int], sums to N
        noun_buckets: list,          # list[int], len=len(group_sizes) — 그룹별 n_noun 버킷
    ):
        dist_weights = self.get_weights().to(logits.device).detach()
        noun_weights = self.get_noun_weights().to(logits.device).detach()

        offset = 0
        per_group_final = []          # (final_loss_tensor, noun_bucket, raw_group_loss_float)
        per_dist_accum  = {d: [] for d in range(1, 7)}

        for gs, nb in zip(group_sizes, noun_buckets):
            g_logits = logits[offset: offset + gs]
            g_dists  = dists[offset: offset + gs]
            offset  += gs

            pos_idxs = [i for i, d in enumerate(g_dists) if d == 0]
            if not pos_idxs:
                continue
            pos_score = g_logits[pos_idxs[0]]

            g_total  = torch.tensor(0.0, device=logits.device)
            has_any  = False
            for d in range(1, 7):
                neg_idxs = [i for i, di in enumerate(g_dists) if di == d]
                if not neg_idxs:
                    continue
                neg_scores = g_logits[neg_idxs]
                ld = F.softplus(neg_scores - pos_score).mean()
                per_dist_accum[d].append(ld.item())
                g_total = g_total + dist_weights[d - 1] * ld
                has_any = True

            if not has_any:
                continue

            g_final = noun_weights[nb] * g_total
            per_group_final.append((g_final, nb, g_total.item()))

        per_dist_loss = {d: sum(v) / len(v) for d, v in per_dist_accum.items() if v}

        if not per_group_final:
            zero = torch.zeros((), device=logits.device, requires_grad=True)
            return zero, per_dist_loss, dist_weights.cpu(), noun_weights.cpu()

        total = torch.stack([g for g, _, _ in per_group_final]).mean()

        with torch.no_grad():
            for d in range(1, 7):
                if d in per_dist_loss:
                    self.ema[d - 1] = (
                        self.ema_alpha * self.ema[d - 1] + (1 - self.ema_alpha) * per_dist_loss[d]
                    )
            for _, nb, g_raw in per_group_final:
                self.ema_noun[nb] = (
                    self.ema_alpha * self.ema_noun[nb] + (1 - self.ema_alpha) * g_raw
                )

        self.step += 1
        return total, per_dist_loss, dist_weights.cpu(), noun_weights.cpu()
