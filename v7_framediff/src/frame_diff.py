"""
v7: 인접 프레임 diff 이미지 계산 — dataset.py(학습)와 inference.py(추론)에서 동일하게 import해서
쓴다 (학습/추론 불일치 버그 재발 방지, v6의 448/512px 사고 교훈).

d=1(인접 스왑) 오류가 해상도를 올려도 안 줄어든다는 EDA 결과(Cohen's d=0.303~0.306, 32~448px 전 구간
동일)에 근거: 문제는 "디테일 부족"이 아니라 "어디가 바뀌었는지 모델이 못 찾는 것"이라는 가설.
순수 픽셀 연산(|frame_t - frame_{t+1}|)만 사용 — 생성형 모델 아님, 제공 데이터만 가공.
"""
import numpy as np
from PIL import Image

DIFF_SIZE = 112  # 원본(448px) 대비 훨씬 작게 — diff는 정밀 디테일이 아니라 대략적 위치만 필요


def compute_diff_images(images: list, diff_size: int = DIFF_SIZE) -> list:
    """4장의 순서가 매겨진 PIL 이미지 -> 인접 3쌍의 diff 이미지 리스트 [diff(0,1), diff(1,2), diff(2,3)]"""
    diffs = []
    for i in range(len(images) - 1):
        a = np.asarray(images[i].convert("RGB"), dtype=np.int16)
        b = np.asarray(images[i + 1].convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            # 원본 이미지 종횡비가 다를 극히 드문 경우 대비: b를 a 크기로 맞춤
            b = np.asarray(
                images[i + 1].convert("RGB").resize((a.shape[1], a.shape[0]), Image.LANCZOS),
                dtype=np.int16,
            )
        diff = np.abs(a - b).astype(np.uint8)
        diff_img = Image.fromarray(diff, mode="RGB").resize((diff_size, diff_size), Image.LANCZOS)
        diffs.append(diff_img)
    return diffs
