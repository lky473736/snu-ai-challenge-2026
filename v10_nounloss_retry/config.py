from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
# v6: base를 TPRU-7B -> Qwen3-VL-8B로 교체 (zero-shot 32.83% > TPRU 26.32%, 같은 태스크 형식으로 검증됨)
# 512px는 실제 데이터로 스모크 테스트 시 OOM 재현(chunk_size=1까지 줄여도 실패) -> 448px로 복귀.
# 추가 EDA(deep_eda_v2.py): gap1 vs gap3 시각적 유사도 Cohen's d가 32px~448px 전 구간에서
# 0.303~0.306으로 거의 동일 -> 해상도를 올려도 이 신호 자체는 더 잘 안 보임. 448이면 충분.
# (배치도 못 늘림: 448px+minibatch=16도 OOM) -> train.py에 OOM 자동 축소 안전장치는 유지.
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
# 모든 샘플(Ordered + No_ordering) 동일: 1 pos + SAMPLE_COUNTS negatives = 8 total/group
# (1 forward pass at TRAIN_MINIBATCH=8). No_ordering도 Answer=[1,2,3,4] 고정 라벨이라
# Ordered와 동일하게 hard negative 대조 학습 (기존 x5 positive-only 증강 방식 폐기 -> 그룹 수
# 7005->1401로 감소, 학습 신호도 saturate 안 되고 지속적으로 유의미해짐).
# Distribution-informed: d=1 gets 2 (most error-prone, 1.70x), others 1 each
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg + 1 pos = 8 total

# ── LoRA ───────────────────────────────────────────────────
# v9: r=64->128 (v8과 동일, alpha/r=2.0 유지)
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
# v9: LoRA Learns Less and Forgets Less(Biderman+ 2024) 근거로 LR 상향 (v8과 동일 5e-5).
LR            = 5e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Adaptive distance weighting (AdaptiveDistanceLoss) ─────
EMA_ALPHA          = 0.99   # EMA smoothing
# v6: dist_weights.csv 실측 결과 temp=2.0은 d=1이 80% 확률로 제일 어렵다고 진단되는데도
# 실제 가중치엔 거의 반영 안 됨(w1=0.188, 균등 0.167과 별차이 없음). temp=6이면 d3~6 비중 61.5%
# 유지하면서 d1에 의미있게 더 집중(w1=0.236) -> v2 blind spot 재현 없이 안전한 강화.
WEIGHT_TEMP        = 6.0

# ── Adaptive n_nouns weighting (AdaptiveDistanceNounLoss 신규 축) ──────
# EDA.md §10-2: n_nouns가 유일하게 독립적인 정확도 예측 신호(로지스틱회귀+bootstrap CI+split-half
# 재현성 검증 통과, 구간별 정확도 29.4%->83.3%). 이 정보를 학습에 직접 반영.
# v7_nounloss(temp=6.0)는 epoch1(-0.42pp)/epoch2(-3.57pp, 격차 확대)로 v6.5 대비 계속 뒤처져
# 중단됨 — WEIGHT_TEMP(distance축)와 달리 이 축은 처음부터 6.0이 과했을 가능성.
# v9: temp를 2.5로 대폭 낮춰 재시도 (2.0~3.0 권장 범위의 중간값).
NOUN_WEIGHT_TEMP   = 2.5

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
TRAIN_MINIBATCH  = 8  # VRAM 체크 후 갱신 (배치 키워서 속도 개선 목표)
