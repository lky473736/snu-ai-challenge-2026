from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/TPRU-7B"))
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
TRAIN_MINIBATCH  = 8
