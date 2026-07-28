from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
# v5 재설계: hard-negative 없이 정답 순열을 "3,1,2,4" 형태 텍스트로 직접 생성(SFT) 학습.
# 그룹당 forward pass가 8(v4)->1로 줄어서 해상도를 크게 올릴 여유가 생김.
# smoke test 결과(job 226395): 448x4=58.93GB / 560x4=74.70GB(너무빡빡) / 672x2=51.14GB(가장 안전) / 560x8=OOM
# -> 672px + batch=2 채택 (v4의 448 대비 50% 상향, EDA 시각유사도 발견에 대한 직접 대응)
MAX_IMAGE_SIZE = 672
VAL_RATIO      = 0.05
SEED           = 42

# ── LoRA ───────────────────────────────────────────────────
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 1e-5
EPOCHS        = 5
BATCH_SIZE    = 2        # 672px에서 4는 OOM -> 2로 축소 (51.14GB, 여유 28.86GB)
GRAD_ACCUM    = 8        # BATCH_SIZE 절반이 된 만큼 늘려서 유효배치(16/GPU) 유지
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── 추론 ───────────────────────────────────────────────────
# 제약 디코딩: "3,1,2,4" 포맷에서 digit 위치(0,2,4,6)만 남은 후보로 제한, comma 위치(1,3,5)는 고정
INFER_BATCH_SIZE = 8
