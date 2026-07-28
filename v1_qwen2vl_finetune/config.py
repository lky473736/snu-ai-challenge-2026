from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT.parent / "data" / "snuaichallenge_data"
MODEL_PATH   = str(PROJECT_ROOT.parent / "models" / "Qwen2-VL-7B-Instruct")
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
MAX_IMAGE_SIZE = 448   # 각 이미지 longest edge
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
N_HARD_NEG = 3   # adjacent swap 3가지 (pos 0-1, 1-2, 2-3)

# ── 짧은 문장 보완 (val 진단: 문장 길이 ↔ 정확도 상관계수 0.376) ─────
SHORT_SENTENCE_WORD_THRESHOLD = 20   # 이 단어 수 이하면 "짧은 문장"으로 취급
SHORT_SENTENCE_OVERSAMPLE_WEIGHT = 1.5   # 짧은 문장 그룹 샘플링 가중치 (살짝 높임)

# ── LoRA ───────────────────────────────────────────────────
LORA_R      = 32
LORA_ALPHA  = 64
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 2e-5
EPOCHS        = 5
BATCH_SIZE    = 1    # 그룹 단위 (1그룹 = pos + hard_neg들)
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Loss ───────────────────────────────────────────────────
MARGIN          = 0.8
BCE_WEIGHT      = 0.3
RANKING_WEIGHT  = 1.0

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 8   # 24개 permutation을 묶어서 처리
