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

# ── Hard Negative ──────────────────────────────────────────
# K=7(v8/v14와 완전히 동일) 고정. grid_search.py 실측 결과(2026-07-07): H100×4 + LoRA r=128 +
# 448px 조합에서 n_extra>0(K 확장)은 전부 OOM — 이 하드웨어에서 K를 늘릴 VRAM 여유가 전혀 없음이
# 확인됨(grid_search_results.csv 참고). 그래서 K 확장 대신 "d1 보너스 슬롯 1개를 동적 하드 네거티브로
# 교체"하는 방식으로 설계 변경(src/hard_negative.py 참고) — 총 K는 그대로 7, 추가 VRAM 불필요.
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg (v8/v14와 동일, 절대 축소 안 함)
N_EXTRA = 0  # 사용 안 함 (하위 호환용, hard_negative.py는 이 값을 무시함)

# ── LoRA ───────────────────────────────────────────────────
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── 학습 ───────────────────────────────────────────────────
LR            = 5e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Loss (v14와 동일, ListwiseSoftmaxLoss는 파라미터 없음) ──
EMA_ALPHA   = 0.99   # 로깅 호환용 placeholder, 미사용
WEIGHT_TEMP = 6.0     # 로깅 호환용 placeholder, 미사용

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
# TRAIN_MINIBATCH: 정밀 그리드서치(2회 반복 평균, 2026-07-07) 결과 group_size=8(K=7 고정) 기준
# minibatch 4~32 전 구간이 76.42GB로 사실상 동일(그룹 크기 자체가 8이라 청크가 어차피 1개) —
# 첫 코-스 그리드서치에서 24가 더 낫다고 나온 건 노이즈였음. v14와 동일하게 8로 고정.
TRAIN_MINIBATCH  = 8

# ── 동적 하드 네거티브 뱅크 ───────────────────────────────
# epoch1은 뱅크가 비어있어 기본+추가 랜덤만 사용, epoch2부터 전 epoch에 기록된 하드 네거티브 반영.
# DDP(4-GPU) 환경에서는 rank별로 다른 샘플을 볼 수 있어 각 rank의 로컬 관찰을 all_gather_object로
# 모아 병합한 뒤 전 rank가 동일한 전역 뱅크를 갖도록 동기화한다(src/train.py 참고).
