from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
MODEL_PATH   = str(Path("/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"))
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── 데이터 ─────────────────────────────────────────────────
# v6: base를 TPRU-7B -> Qwen3-VL-8B로 교체 (zero-shot 32.83% > TPRU 26.32%, 같은 태스크 형식으로 검증됨)
# 512px는 실제 데이터로 스모크 테스트 시 OOM 재현(chunk_size=1까지 줄여도 실패) -> 448px로 복귀.
# 추가 EDA(deep_eda_v2.py): gap1 vs gap3 시각적 유사도 Cohen's d가 32px~448px 전 구간에서
# 0.303~0.306으로 거의 동일 -> 해상도를 올려도 이 신호 자체는 더 잘 안 보임. 448이면 충분.
# (배치도 못 늘림: 448px+minibatch=16도 OOM) -> train.py에 OOM 자동 축소 안전장치는 유지.
MAX_IMAGE_SIZE = 448
VAL_RATIO      = 0.05
SEED           = 42

# ── Hard Negative ──────────────────────────────────────────
# 모든 샘플(Ordered + No_ordering) 동일: 1 pos + SAMPLE_COUNTS negatives = 8 total/group
# (1 forward pass at TRAIN_MINIBATCH=8). No_ordering도 Answer=[1,2,3,4] 고정 라벨이라
# Ordered와 동일하게 hard negative 대조 학습 (기존 x5 positive-only 증강 방식 폐기 -> 그룹 수
# 7005->1401로 감소, 학습 신호도 saturate 안 되고 지속적으로 유의미해짐).
# Distribution-informed: d=1 gets 2 (most error-prone, 1.70x), others 1 each
SAMPLE_COUNTS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}  # 7 neg + 1 pos = 8 total

# ── LoRA ───────────────────────────────────────────────────
# v9(1차, 폐기): PiSSA+rsLoRA(alpha=256 유지)+LoRA+ 조합 -> epoch1 val=0.1618로 붕괴(참사).
# 원인 규명: rsLoRA 논문 저자 본인이 GitHub issue(peft#1387)에서 명시 - "rank를 올릴 때
# alpha를 rank에 비례해서 같이 올리면 안 되고, 기존에 쓰던 낮은 rank용 고정 alpha를 그대로
# 유지해야 한다." 우리는 alpha=256(=2×r, rank에 비례해서 잡은 값)을 그대로 rsLoRA 공식
# (alpha/sqrt(r))에 대입해서 effective scale이 256/sqrt(128)≈22.6까지 폭주함 -> 붕괴 원인.
# v9(2차): PiSSA + rsLoRA 조합 (LoRA+는 변수 격리 위해 이번엔 제외).
# alpha는 rank와 무관한 고정값 16(흔한 LoRA 기본값)으로 -> effective scale=16/sqrt(128)≈1.41
# (v8의 2.0과 비슷한 안전한 범위).
LORA_R       = 128
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_INIT    = "pissa_niter_16"  # 근사 SVD. DoRA와 달리 forward/backward 구조 동일해 메모리 오버헤드 없음.
USE_RSLORA   = True

# ── LoRA+ (v9 2차에서는 비활성 — 변수 격리) ──────────────────
LORA_PLUS_RATIO = 1.0  # 1.0 = A/B 동일 LR (LoRA+ 비활성과 동일 효과)

# ── 학습 ───────────────────────────────────────────────────
# v8: LoRA Learns Less and Forgets Less(Biderman+ 2024) 근거로 LR 상향.
# 논문 실측: LoRA 최적 LR은 full finetuning(우리가 지금까지 쓰던 1e-5)보다 한 자릿수 높아야 함
# (IFT 기준 5e-5~5e-4 권장, r=64에서 1e-4). 우리 태스크는 code/math보다 도메인 시프트가
# 작은 이진분류형이라 보수적으로 5e-5(현재의 5배)로 설정 — "약간 공격적" 선택.
LR            = 5e-5
EPOCHS        = 5
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 50

# ── Adaptive distance weighting (AdaptiveDistanceLoss) ─────
EMA_ALPHA          = 0.99   # EMA smoothing
# v6: dist_weights.csv 실측 결과 temp=2.0은 d=1이 80% 확률로 제일 어렵다고 진단되는데도
# 실제 가중치엔 거의 반영 안 됨(w1=0.188, 균등 0.167과 별차이 없음). temp=6이면 d3~6 비중 61.5%
# 유지하면서 d1에 의미있게 더 집중(w1=0.236) -> v2 blind spot 재현 없이 안전한 강화.
WEIGHT_TEMP        = 6.0

# ── Inference ──────────────────────────────────────────────
INFER_BATCH_SIZE = 24
TRAIN_MINIBATCH  = 8  # VRAM 체크 후 갱신 (배치 키워서 속도 개선 목표)
