"""
AdaptiveDistanceLoss — per-Kendall-distance adaptive weighting

For each group:
  L_d = mean softplus(neg_score_d - pos_score)  [pairwise ranking loss, d=1..6]

Adaptive weights via EMA of per-distance losses:
  ema_d  ← alpha * ema_d + (1-alpha) * L_d
  w_d    = softmax(ema_d * temperature)          [high loss → high weight]

Total = Σ_d  w_d * mean_over_groups(L_d)
Gradient flows only through model params — w_d is detached (no bilevel needed).
"""

import torch
import torch.nn.functional as F


class AdaptiveDistanceLoss:
    def __init__(self, ema_alpha: float = 0.99, temperature: float = 1.0):
        self.ema_alpha   = ema_alpha
        self.temperature = temperature
        self.ema  = torch.ones(6)   # d=1..6 → indices 0..5, start uniform
        self.step = 0

    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.ema * self.temperature, dim=0)  # (6,)

    def __call__(
        self,
        logits:      torch.Tensor,  # (N,)
        dists:       list,          # flat list[int], len=N, d=0..6
        group_sizes: list,          # list[int], sums to N
    ):
        weights = self.get_weights().to(logits.device).detach()

        per_dist_accum = {d: [] for d in range(1, 7)}
        offset = 0

        for gs in group_sizes:
            g_logits = logits[offset: offset + gs]
            g_dists  = dists[offset: offset + gs]
            offset  += gs

            pos_idxs = [i for i, d in enumerate(g_dists) if d == 0]
            if not pos_idxs:
                continue
            pos_score = g_logits[pos_idxs[0]]

            for d in range(1, 7):
                neg_idxs = [i for i, di in enumerate(g_dists) if di == d]
                if not neg_idxs:
                    continue
                neg_scores = g_logits[neg_idxs]
                per_dist_accum[d].append(F.softplus(neg_scores - pos_score).mean())

        per_dist_loss = {}
        total = torch.tensor(0.0, device=logits.device)

        for d in range(1, 7):
            if not per_dist_accum[d]:
                continue
            ld = torch.stack(per_dist_accum[d]).mean()
            per_dist_loss[d] = ld.item()
            total = total + weights[d - 1] * ld

        with torch.no_grad():
            for d in range(1, 7):
                if d in per_dist_loss:
                    self.ema[d - 1] = (
                        self.ema_alpha * self.ema[d - 1]
                        + (1 - self.ema_alpha) * per_dist_loss[d]
                    )

        self.step += 1
        return total, per_dist_loss, weights.cpu()
