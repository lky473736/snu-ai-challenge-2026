from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── LoRA ───────────────────────────────────────────────────
# v14와 100% 동일(검증된 값 재사용, 이 축은 안 건드림)
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 5e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 20

# d=1(인접) 쌍에 손실 가중치를 더 줘서 idea.md 5-1절의 가장 완고한 약점을 직접 타겟팅
# — 단, 확정적 해법이 아니라 실험 대상(과거 loss reweighting 시도들이 d=1 비율을 못 줄인 전례 있음)
ADJACENT_PAIR_WEIGHT = 1.5

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
CKPT_NAME = "best_v18"
