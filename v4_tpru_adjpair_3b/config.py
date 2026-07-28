import os
from pathlib import Path

# ── 경로 ── Colab 등 로컬 환경에서는 환경변수로 오버라이드
#   HNTV_DATA_DIR : train.csv/test.csv가 들어있는 snuaichallenge_data 폴더
#   HNTV_MODEL_PATH : 로컬 경로 또는 HuggingFace repo id (from_pretrained가 자동 다운로드)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path(os.environ.get("HNTV_DATA_DIR", str(PROJECT_ROOT / "data" / "snuaichallenge_data")))
MODEL_PATH   = os.environ.get("HNTV_MODEL_PATH", "Stephengzk/TPRU-3B")
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
# Ordered: 1 pos + SAMPLE_COUNTS negatives = 8 total per group (1 forward pass at TRAIN_MINIBATCH=8)
# Distribution-informed: d=1 gets 2 (most error-prone, 1.70x), others 1 each
# All d=1..6 covered; AdaptiveDistanceLoss up-weights struggling distances dynamically
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg + 1 pos = 8 total
# No_ordering: positive only × 5 (diverse base perms)

# ── LoRA ───────────────────────────────────────────────────
# 7B에서 쓰던 값 그대로 유지 (LoRA rank는 base 모델 크기와 독립적)
# Colab GPU 메모리가 빠듯하면 --lora_r 32 --lora_alpha 64로 낮춰서 실행 가능
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 1e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Adaptive distance weighting (AdaptiveDistanceLoss) ─────
EMA_ALPHA          = 0.99   # EMA smoothing
WEIGHT_TEMP        = 2.0    # softmax temperature for w_d
NO_ORDERING_WEIGHT = 0.5    # BCE positive loss weight for No_ordering samples

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
# 3B는 7B보다 activation이 작지만 Colab GPU(T4 16GB 등)는 H100보다 훨씬 작으므로
# OOM 시 HNTV_TRAIN_MINIBATCH=4 (또는 2)로 낮춰서 실행
TRAIN_MINIBATCH  = int(os.environ.get("HNTV_TRAIN_MINIBATCH", "8"))
