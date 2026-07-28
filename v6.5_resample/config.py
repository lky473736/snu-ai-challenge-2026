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
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 1e-5
# v6.5: v6(EPOCHS=5)는 epoch4(0.5861)에서 정점 찍고 epoch5(0.5735)부터 과적합.
# live resampling이 "고정 negative 세트 암기"로 인한 과적합을 늦출 수 있다는 가설 검증 위해 10으로 확장.
# cosine LR 스케줄이 EPOCHS 전체 기준으로 계산되므로 자동으로 10epoch에 맞게 재조정됨.
EPOCHS        = 10
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

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
TRAIN_MINIBATCH  = 8  # VRAM 체크 후 갱신 (배치 키워서 속도 개선 목표)
