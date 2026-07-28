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
N_HARD_NEG = 3

# ── LoRA ── conservative: TPRU가 이미 temporal 이해 → 작은 r로 domain 적응만
LORA_R      = 32
LORA_ALPHA  = 64
LORA_DROPOUT = 0.05

# ── 학습 ── LR 절반: TPRU의 temporal RL 지식 보존
LR            = 1e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Loss ── v3 설정 그대로 적용
MARGIN          = 1.2
BCE_WEIGHT      = 0.3
RANKING_WEIGHT  = 1.5

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24   # inference: 24 permutation 전부 한 번에
TRAIN_MINIBATCH  = 6    # training: forward pass 내 sub-batch 크기 (OOM 방지)
