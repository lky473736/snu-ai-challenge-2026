from pathlib import Path

# v13: 재학습 없음 — 기존 best_v8 체크포인트(현재 최고 제출, LB 0.89528)에 프롬프트만 바꿔서
# EDA.md §10-2 핵심 발견(n_nouns가 정확도의 유일한 독립 신호, r=0.467)을 겨냥한 저비용 A/B.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))

V8_DIR       = Path("/data/gyuyeonlim/snu_ai_challenge/v8_lora128")
CKPT_PATH    = V8_DIR / "checkpoints" / "best_v8"
VAL_RAW_CSV  = Path("/data/gyuyeonlim/snu_ai_challenge/v10_nounloss_retry/checkpoints/_val_raw.csv")
# v10의 _val_raw.csv는 v8과 동일 SEED=42/VAL_RATIO=0.05 split(Id 476개 완전 일치 확인됨)에
# n_nouns 컬럼까지 이미 merge되어 있어 재사용 (spaCy 재계산 불필요).

MAX_IMAGE_SIZE   = 448   # v8 학습 해상도와 반드시 일치
INFER_BATCH_SIZE = 24
LOG_DIR          = PROJECT_ROOT / "logs"
