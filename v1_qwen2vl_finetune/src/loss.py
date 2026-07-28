"""
VerificationRankingLoss
= BCE(positive+negatives) + MarginRankingLoss(pos vs each neg)
"""

import torch
import torch.nn as nn
from config import BCE_WEIGHT, RANKING_WEIGHT, MARGIN


class VerificationRankingLoss(nn.Module):
    def __init__(
        self,
        margin: float = MARGIN,
        bce_weight: float = BCE_WEIGHT,
        ranking_weight: float = RANKING_WEIGHT,
    ):
        super().__init__()
        self.margin       = margin
        self.bce_weight   = bce_weight
        self.ranking_weight = ranking_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ranking = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        logits: torch.Tensor,      # (N,) — 배치 내 모든 샘플 logit
        labels: torch.Tensor,      # (N,) — 1=positive, 0=negative
        group_sizes: list,         # 각 그룹의 크기 (pos+neg 수)
    ) -> torch.Tensor:

        # ── BCE loss (전체) ────────────────────────────────
        bce_loss = self.bce(logits, labels)

        # ── Margin Ranking loss (그룹 내 pos vs neg) ───────
        ranking_losses = []
        offset = 0
        for size in group_sizes:
            group_logits = logits[offset: offset + size]
            group_labels = labels[offset: offset + size]
            offset += size

            pos_mask = group_labels == 1
            neg_mask = group_labels == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_logit = group_logits[pos_mask].mean()   # positive 평균
            neg_logits_g = group_logits[neg_mask]       # hard negatives

            # positive가 각 negative보다 margin 이상 높아야
            target = torch.ones(len(neg_logits_g), device=logits.device)
            ranking_losses.append(
                self.ranking(
                    pos_logit.expand_as(neg_logits_g),
                    neg_logits_g,
                    target,
                )
            )

        if ranking_losses:
            ranking_loss = torch.stack(ranking_losses).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=logits.device)

        total = self.bce_weight * bce_loss + self.ranking_weight * ranking_loss
        return total, bce_loss.detach(), ranking_loss.detach()
