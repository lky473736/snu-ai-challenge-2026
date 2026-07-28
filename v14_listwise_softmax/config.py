from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
# v14: v8과 100% 동일 레시피(LoRA r/alpha/LR/EPOCHS, hard negative 구성)를 유지하고
# loss.py만 AdaptiveDistanceLoss -> ListwiseSoftmaxLoss(8-way joint softmax, 가중치 없음)로 교체.
# 단일 변수 실험 원칙(idea.md 5-4절) 준수 위해 이 파일의 값들은 v8_lora128/config.py와 절대 다르면 안 됨.
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
# v8과 완전히 동일: 1 pos + SAMPLE_COUNTS negatives = 8 total/group, d=1..6 전 구간 커버.
# v14 실험은 이 구성을 절대 건드리지 않음 (K를 줄이거나 self-hard mining을 섞으면
# "loss 자체의 효과"와 "negative 커버리지 변화"가 뒤섞여 원인 분석이 불가능해짐).
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg + 1 pos = 8 total

# ── LoRA ───────────────────────────────────────────────────
# v8과 완전히 동일 (r=128, alpha=256, 비율 2.0)
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
# v8과 완전히 동일 (LR=5e-5, EPOCHS=5, batch 구성 동일)
LR            = 5e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Loss ────────────────────────────────────────────────────
# v14: ListwiseSoftmaxLoss는 파라미터가 없음(EMA_ALPHA/WEIGHT_TEMP 불필요).
# 아래 두 값은 오직 로깅 코드(train.py의 dist_weights.csv 헤더/스모크 테스트) 호환을 위해 남겨둠 — loss 계산엔 미사용.
EMA_ALPHA   = 0.99
WEIGHT_TEMP = 6.0

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
TRAIN_MINIBATCH  = 8
