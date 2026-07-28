from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-32B-Instruct"))  # bf16 원본, on-the-fly 4bit 양자화
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
# v14와 100% 동일(단일변수 원칙 — 베이스 모델 32B QLoRA 교체만 격리)
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
# v14와 완전히 동일: 1 pos + SAMPLE_COUNTS negatives = 8 total/group, d=1..6 전 구간 커버.
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg + 1 pos = 8 total

# ── LoRA ───────────────────────────────────────────────────
# v14와 동일 값 유지(단일변수 원칙) — rank/LR을 32B에 맞게 다시 튜닝하는 건 이 실험 이후 과제.
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── QLoRA(4bit) ──────────────────────────────────────────────
# vision tower는 절대 양자화하지 않음 — Qwen3VLModel.visual 하위 전부 skip.
# (myunhh 팀이 겪은 "vision skip 이름 불일치로 vision까지 4bit화" 버그를 우리는 on-the-fly
#  양자화라 직접 skip 목록을 지정해서 원천 차단)
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
LLM_INT8_SKIP_MODULES = ["visual"]

# ── 학습 ───────────────────────────────────────────────────
# v14와 동일(LR/BATCH/GRAD_ACCUM). EPOCHS=3, val 0.6176(v14 5epoch 최고치 0.5987 이미 넘음).
# 이 설정으로 학습한 best_v20 체크포인트가 실제 제출본(submission_v20_best.csv,
# public 0.91099 / private 0.90650)을 만들었다 — 재현성을 위해 반드시 EPOCHS=3 유지.
LR            = 5e-5
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 20

# ── 학습 배치(그룹 내 청크) ───────────────────────────────────
# v14는 8(=group_size)이었음 — 32B는 훨씬 크므로 스모크로 재확인 후 조정할 것.
TRAIN_MINIBATCH  = 8
INFER_BATCH_SIZE = 24
