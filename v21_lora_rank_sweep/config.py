from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-32B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 / Hard Negative ───────────────────────────────────
# v14/v20과 100% 동일(단일변수 원칙 — LoRA rank만 격리해서 실험)
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42
SAMPLE_COUNTS  = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# ── LoRA ───────────────────────────────────────────────────
# rank는 run_rank_sweep.sh가 VRAM 한계까지 탐색해서 --lora_r로 넘겨준다.
# alpha는 항상 r*LORA_ALPHA_RATIO로 자동 계산(v14/v20의 r=128/alpha=256=비율 2.0을 그대로 유지).
LORA_R_DEFAULT    = 128
LORA_ALPHA_RATIO  = 2.0
LORA_DROPOUT      = 0.05

# ── QLoRA(4bit) ──────────────────────────────────────────────
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
LLM_INT8_SKIP_MODULES = ["visual"]

# ── 학습 ───────────────────────────────────────────────────
LR            = 5e-5
EPOCHS        = 5   # 마감 임박 + rank sweep 결과 반영해서 5epoch 본격 실행
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 20

TRAIN_MINIBATCH  = 8
INFER_BATCH_SIZE = 24
