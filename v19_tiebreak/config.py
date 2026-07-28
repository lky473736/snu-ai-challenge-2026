from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))

# v19는 재학습 없이 v14의 best 체크포인트를 그대로 불러와 zero-shot으로 타이브레이크만 테스트한다.
V14_CKPT    = Path("/data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax/checkpoints/best_v14")
V14_VAL_CSV = Path("/data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax/checkpoints/_val_raw.csv")

CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR  = PROJECT_ROOT / "logs"

MAX_IMAGE_SIZE    = 448  # v14와 동일 (학습 때 본 해상도와 일치시켜야 함)
INFER_BATCH_SIZE  = 24   # 24-permutation 전수조사, 그룹 전체를 한 배치로
