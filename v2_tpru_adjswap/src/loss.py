"""
VerificationRankingLoss
= BCE + ListNet (all 24 permutations ranked jointly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BCE_WEIGHT, RANKING_WEIGHT, MARGIN


class VerificationRankingLoss(nn.Module):
    def __init__(
        self,
        margin: float = MARGIN,
        bce_weight: float = BCE_WEIGHT,
        ranking_weight: float = RANKING_WEIGHT,
    ):
        super().__init__()
        self.bce_weight     = bce_weight
        self.ranking_weight = ranking_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,   # (N,)
        labels: torch.Tensor,   # (N,) — 1=positive, 0=negative
        group_sizes: list,
    ):
        bce_loss = self.bce(logits, labels)

        listnet_losses = []
        offset = 0
        for size in group_sizes:
            g_logits = logits[offset: offset + size]
            g_labels = labels[offset: offset + size]
            offset += size

            n_pos = g_labels.sum()
            if n_pos == 0:
                continue

            log_probs = F.log_softmax(g_logits, dim=0)
            target    = g_labels / n_pos          # uniform over positives
            listnet_losses.append(-(target * log_probs).sum())

        if listnet_losses:
            rank_loss = torch.stack(listnet_losses).mean()
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)

        total = self.bce_weight * bce_loss + self.ranking_weight * rank_loss
        return total, bce_loss.detach(), rank_loss.detach()
