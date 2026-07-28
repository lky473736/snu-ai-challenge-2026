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
# v8: r=64->128 실험. alpha/r 비율(2.0)을 v6.5와 동일하게 유지하려고 alpha도 128->256로 같이 올림
# (r만 올리고 alpha를 안 바꾸면 LoRA 업데이트 스케일(alpha/r)이 절반으로 줄어서
# "용량만 늘린 순수 비교"가 안 됨 — 반드시 비율 유지해서 비교해야 함)
LORA_R       = 128
LORA_ALPHA   = 256
LORA_DROPOUT = 0.05

# ── Vision encoder LoRA (v12: LLM+vision 동시 처음부터 학습) ──
# v11(freeze 전략)과 비교하기 위한 대조군 — 베이스 모델에서 LLM LoRA(r=128)와 vision
# LoRA를 동시에 처음부터 학습. v11 스모크(4블록, r=64)에서 VRAM이 79.81/80GB로 거의
# 한계였으므로, 동시학습은 학습 중 OOM 위험을 낮추기 위해 블록 수를 4->3으로 축소.
VISION_LORA_ENABLE = True
VISION_LAST_N_BLOCKS = 3
VISION_LORA_R        = 64
VISION_LORA_ALPHA    = 128  # 비율 2.0 유지

# ── 학습 ───────────────────────────────────────────────────
# v8: LoRA Learns Less and Forgets Less(Biderman+ 2024) 근거로 LR 상향.
# 논문 실측: LoRA 최적 LR은 full finetuning(우리가 지금까지 쓰던 1e-5)보다 한 자릿수 높아야 함
# (IFT 기준 5e-5~5e-4 권장, r=64에서 1e-4). 우리 태스크는 code/math보다 도메인 시프트가
# 작은 이진분류형이라 보수적으로 5e-5(현재의 5배)로 설정 — "약간 공격적" 선택.
# vision encoder LoRA는 LLM LoRA보다 낮은 LR이 안정적이라는 일반적 권장(2e-5~3e-5)에 따라
# LLM(5e-5)의 절반 수준(2.5e-5)으로 분리. LLM은 처음부터 학습하는데 vision까지 같은 LR로
# 동시에 크게 흔들면 두 파트가 서로 다른 속도로 수렴해야 하는데 불안정해질 수 있음.
LR            = 5e-5
VISION_LR     = 2.5e-5
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
