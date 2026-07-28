import torch
import torch.nn.functional as F

from config import ADJACENT_PAIR_WEIGHT


def pairwise_bt_loss(logits, labels, adj_flags):
    """Bradley-Terry negative log-likelihood — BCE(z_ij, y_ij)의 평균. d=1(인접) 쌍은 가중치 상향.
    reduction='mean'이라 그룹 크기(6)에 무관하게 손실 스케일이 v14의 단일 8-way softmax 항과
    비슷한 크기로 유지됨 -> LR=5e-5를 그대로 재사용해도 무리가 없음."""
    labels_t = torch.tensor(labels, device=logits.device, dtype=logits.dtype)
    weights = torch.tensor([ADJACENT_PAIR_WEIGHT if a else 1.0 for a in adj_flags],
                            device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels_t, weight=weights, reduction="mean")
