"""
쌍별 로그오즈 -> 전역 순위 집계.
6개 쌍의 로그오즈를 반대칭 행렬로 모으고 행 평균을 내면, K4(완전그래프·균형 관측)에서는
이게 정확히 최소제곱(Hodge 분해) 해와 같다 — 별도 solver 없이 mean만으로 충분.
"""

import torch

from src.dataset import PAIRS, build_messages_pair
from src.model import pairwise_logits_fast, forward_logit


def score_pairs_fast(model, processor, base_imgs, sentence, device):
    """고속 경로(vision 중복 인코딩 제거). {(i,j): z_ij} 반환."""
    with torch.no_grad():
        z = pairwise_logits_fast(model, processor, base_imgs, sentence, device)
    return dict(zip(PAIRS, z.cpu().tolist()))


def score_pairs_slow(model, processor, base_imgs, sentence, yes_id, no_id, device, minibatch=16):
    """느린 참조 경로 — 검증 스크립트 전용(smoke_test_v18.py)."""
    texts, imgs_list = [], []
    for (i, j) in PAIRS:
        msg = build_messages_pair(base_imgs, sentence, i, j)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        imgs_list.append(base_imgs)
    zs = []
    with torch.no_grad():
        for bi in range(0, len(texts), minibatch):
            inp = processor(text=texts[bi:bi+minibatch], images=imgs_list[bi:bi+minibatch],
                             return_tensors="pt", padding=True).to(device)
            s = forward_logit(model, inp, yes_id, no_id)
            zs.extend(s.cpu().tolist())
    return dict(zip(PAIRS, zs))


def aggregate_ranks(pair_scores):
    """raw 로그오즈 dict -> 닫힌 형태 집계(행 평균) -> Answer 포맷 순위. 이산 최적화 없음."""
    Z = [[0.0] * 5 for _ in range(5)]  # 1-indexed, 반대칭
    for (i, j), z in pair_scores.items():
        Z[i][j] = z
        Z[j][i] = -z
    s = [sum(Z[i][j] for j in range(1, 5) if j != i) / 3.0 for i in range(1, 5)]
    order = sorted(range(1, 5), key=lambda i: -s[i - 1])  # s 높을수록(더 자주 '먼저') 이른 프레임
    pred_ranks = [0] * 4
    for rank_pos, frame_i in enumerate(order, 1):
        pred_ranks[frame_i - 1] = rank_pos
    return pred_ranks


def predict_permutation(model, processor, base_imgs, sentence, device):
    return aggregate_ranks(score_pairs_fast(model, processor, base_imgs, sentence, device))
