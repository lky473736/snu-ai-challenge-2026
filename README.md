# TEMPO

**Temporal Event Matching via Permutation-scored Ordering**
Hard-Negative Listwise Ranking for Chronological Frame Reconstruction with Vision-Language Models

SNU AI Challenge 2026 · Kaggle `snuaichallenge` · Public **0.91099** / Private **0.90650**

![overall architecture](figures/architecture.png)

4장의 셔플된 비디오 프레임과 그 사건을 설명하는 문장(caption)이 주어졌을 때, 원래 사건의 올바른
시간 순서(4! = 24가지 경우의 수 중 1개)를 맞히는 과제입니다.

- **T**emporal: 시간 순서를 다루는 과제 본질
- **E**vent Matching: 문장과 프레임 순서를 매칭
- **via** **P**ermutation-scored: 24개 순열을 전수조사해 Yes/No 검증 점수(`log P(Yes) - log P(No)`)로 채점
- **Ordering**: 최종적으로 올바른 시간 순서를 결정
- **Hard-Negative Listwise Ranking**: 켄달타우 거리로 뽑은 hard negative와 정답을 묶어 8-way joint
  softmax(Plackett-Luce top-1)로 학습하는 `ListwiseSoftmaxLoss` — 핵심 방법론

이 저장소는 예선 전 기간의 실험 히스토리(v1-v21)와 예선 최종 제출 모델(v20)의 학습·추론 전체 코드
및 가중치를 담고 있습니다.

최종 제출 모델은 `v20_32b_qlora/`이고, 모델은 Qwen3-VL-32B-Instruct + QLoRA(4bit NF4) LoRA
어댑터입니다. 체크포인트는 `v20_32b_qlora/checkpoints/best_v20` (epoch 3, val exact-match 0.6176)
이고, v1-v19와 v21은 이 최종 모델에 도달하기까지의 실험 히스토리입니다.

---

## Quick Start

```bash
git clone https://github.com/lky473736/snu-ai-challenge-2026.git
cd snu-ai-challenge-2026/v20_32b_qlora

conda create -n aichallenge python=3.10 -y && conda activate aichallenge
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r ../requirements.txt

python3 -c "import kagglehub; kagglehub.competition_download('snuaichallenge', output_dir='data')"

bash download_base_model.sh
bash download_weights.sh

python inference.py
```

결과는 `v20_32b_qlora/submission_v20_best.csv`에 생성됩니다. 아래 1-5번 섹션에 각 단계의 상세
설명이 있습니다.

## 목차

1. 실행 환경
2. 설치
3. 데이터 및 가중치
4. 실행 방법
5. 재현성에 대하여
6. 준수 사항
7. 버전 히스토리
8. 결과
9. 라이선스

---

## 1. 실행 환경

학습(4×H100)과 제출 모델 재현(대회 규정상 RTX 3090 1장)은 서로 다른 CUDA 환경을 씁니다.

| 항목 | 값 |
|---|---|
| OS / Python | Linux / 3.10 |
| GPU (학습) | NVIDIA H100 80GB × 4, CUDA 13.0, PyTorch 2.12.1+cu130 |
| GPU (재현 대상) | NVIDIA RTX 3090 24GB × 1, driver 550.54.15, CUDA 12.4 |

driver 550.54.15는 CUDA 12.4까지만 지원하므로, RTX 3090에서는 반드시 cu124 빌드 torch를 설치해야
합니다.

라이브러리 버전은 `requirements.txt`에 고정되어 있습니다 (`transformers==5.12.1`, `peft==0.19.1`,
`accelerate==1.14.0`, `bitsandbytes==0.49.2`). `torch`/`torchvision`은 CUDA 빌드가 갈려서 따로
설치합니다.

## 2. 설치

```bash
conda create -n aichallenge python=3.10 -y
conda activate aichallenge

# RTX 3090 / CUDA 12.4 (제출 모델 재현, 기본)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 4×H100 / CUDA 13.0 (학습 재현 시에는 위 대신 이걸로)
# pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

## 3. 데이터 및 가중치

대회 원본 데이터와 모델 가중치는 용량 문제로 저장소에 포함하지 않았습니다. `config.py`의
`DATA_DIR`/`MODEL_PATH` 기본값은 모두 이 파일 기준 상대 경로입니다.

```bash
cd v20_32b_qlora

# 데이터 -> data/train.csv, data/test.csv, data/train|test/<Id>/*.jpg
python3 -c "import kagglehub; kagglehub.competition_download('snuaichallenge', output_dir='data')"

# 베이스 모델 (Qwen/Qwen3-VL-32B-Instruct, 공개 오픈소스, 약 65GB)
bash download_base_model.sh

# LoRA 어댑터 (4.1GB, https://huggingface.co/lky473736/snuaichallenge-v20-qwen3vl32b-qlora)
bash download_weights.sh
```

위 두 다운로드 스크립트는 인터넷이 되는 동안 한 번만 실행하면 됩니다. 그 이후 `inference.py`
실행 자체는 로컬 캐시만 사용해 인터넷이 필요 없습니다 (대회 규정 3.1). 네트워크 호출을 코드
레벨에서 완전히 막으려면 실행 전 `export HF_HUB_OFFLINE=1`을 설정하세요.

## 4. 실행 방법

**학습 (재현용, 4×H100)**

```bash
cd v20_32b_qlora
sbatch run_train.sh
# 또는: accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py
```

핵심 하이퍼파라미터(전체는 `config.py` 참고): LoRA r=128/alpha=256/dropout=0.05, 4bit NF4 double
quant(vision tower는 bf16 유지), epochs=3(반드시 이 값이어야 재현됨), LR 5e-5, hard negative는
켄달타우 거리 d=1 2개 + d=2-6 각 1개.

**추론 — 4×GPU (학습 환경)**

```bash
cd v20_32b_qlora
torchrun --nproc_per_node=4 inference.py
```

**추론 — 단일 GPU (RTX 3090 24GB, 대회 규정 4번 실행 환경)**

```bash
cd v20_32b_qlora
python inference.py
```

`WORLD_SIZE`/`RANK`/`LOCAL_RANK`가 없으면 기본값 1/0/0으로 동작해 코드 수정 없이 GPU 1장에서 그대로
실행됩니다. 베이스 모델은 4bit NF4로 로드되고 LoRA만 bf16으로 얹히며, 24GB에 다 안 들어가 vision
tower와 lm_head(둘 다 bf16 유지, 정밀도 변화 없음)는 CPU로 오프로드됩니다. 819개 test set 처리에
총 약 5시간 20분 소요(24시간 제한 내).

**추론 — 외부 데이터셋 검증용 (본선 후보자)**

```bash
cd v20_32b_qlora
# inference_semifinal.py 상단 EXT_DATA_DIR을 실제 경로로 수정
sbatch run_infer_semifinal.sh
```

추가 학습 없이 예선 제출본과 100% 동일한 체크포인트(`best_v20`)만 재사용합니다.

**사전 VRAM 점검 (선택)**

```bash
cd v20_32b_qlora && bash run_smoke.sh
```

## 5. 재현성에 대하여

결정론적 실행을 위해 seed 고정, cuDNN deterministic 모드, TF32 비활성화, 고정 배치 크기를 모두
적용했습니다. 다만 GPU에서 bf16, 4bit 양자화, attention 연산을 수행할 때는 부동소수점 덧셈 순서가
완전히 고정되지 않아, 로짓 값이 소수점 아래에서 미세하게 흔들릴 수 있습니다.

이 과제는 24가지 순서 중 하나를 고르는 분류 문제이므로, 1등과 2등 점수 차이가 근소한 일부 샘플은
이런 미세한 흔들림만으로 최종 예측(argmax)이 뒤집힐 수 있습니다. 이는 저희 코드만의 문제가 아니라
GPU 딥러닝 추론 전반에 알려진 현상이며, 학습에 쓴 하드웨어(H100)와 재현 환경(RTX 3090)처럼 GPU
자체가 다르면 그 폭이 더 커질 수 있습니다.

그래서 동일한 설정으로 다시 실행해도 전체 행의 2-4% 정도는 달라질 수 있습니다. 이 정도 범위의
차이는 버그가 아니라 예상된 numerical discrepancy로 봐주시면 됩니다.

## 6. 준수 사항

- 학습 데이터는 대회 제공 데이터만 사용했으며, 테스트 데이터를 학습에 참조하지 않았습니다.
- 외부 상용 API(ChatGPT, Gemini 등)는 사용하지 않았습니다.
- 모델 앙상블은 사용하지 않았습니다 (단일 모델, 단일 체크포인트).
- 사용한 오픈소스 모델(Qwen3-VL-32B-Instruct)은 2026년 5월 31일 이전 공개 모델입니다.
- 경량화 기법으로 QLoRA(4bit NF4 + LoRA)를 사용했습니다 (대회 규정상 허용).
- 모든 경로는 `config.py` 기준 상대 경로로 관리됩니다.

## 7. 버전 히스토리 (v1-v21)

각 `vN_*` 폴더는 독립적인 실험 단위입니다. 최종 제출 모델 v20 외에는 용량 제한상 가중치를 포함하지
않았습니다 (코드로 재현 가능).

| 버전 | 베이스 모델 | 핵심 아이디어 | 결과 |
|---|---|---|---|
| v1 | Qwen2-VL-7B | LoRA r=32, adjacent-swap 3종 hard negative | LB 0.79057 |
| v2 | TPRU-7B | adj-swap 3 + reverse 1, ListNet | LB 0.81849 |
| v2.5 | TPRU-7B | +TTA flip +pairwise +560px(추론만) | LB 0.71553 (급락) |
| v3 | TPRU-7B | 23개 negative 전부 사용 | 취소(24h 한도 초과) |
| v4 | TPRU-7B | SAMPLE_COUNTS 전거리 커버 + AdaptiveDistanceLoss | LB 0.83944 |
| v5 | Qwen3-VL-8B | listwise 순열 직접 생성(SFT) | val 0.4475 (폐기) |
| v6 | Qwen3-VL-8B | v4 레시피 + base 교체 + temp 튜닝 | LB 0.87085 |
| v6.5 | Qwen3-VL-8B | hard negative 매 epoch 재샘플링 + 10epoch | LB 0.87260 |
| v7 | Qwen3-VL-8B | AdaptiveDistanceNounLoss / frame-diff 입력 | 폐기 |
| v8 | Qwen3-VL-8B | LoRA r=128, LR 5e-5, EPOCHS 5 | LB 0.89528 |
| v9 | Qwen3-VL-8B | DoRA / PiSSA+rsLoRA+LoRA+ | 폐기(OOM/학습 붕괴) |
| v10 | Qwen3-VL-8B | AdaptiveDistanceNounLoss 재보정 | val 0.6071(역대 최고), LB 0.88481 |
| v11 | Qwen3-VL-8B | LLM 병합 고정 + vision-only LoRA | LB 0.89528 |
| v12 | Qwen3-VL-8B | LLM+vision LoRA 동시학습 | LB 0.89354 |
| v13 | Qwen3-VL-8B | 프롬프트 nudge(재학습 없음) | 효과 없음 |
| v14 | Qwen3-VL-8B | loss를 ListwiseSoftmaxLoss로 교체 | LB 0.90052 |
| v15 | Qwen3-VL-8B | dynamic hard-negative(1슬롯 교체) | 제출 보류 |
| v16 | Qwen3-VL-8B | curriculum learning(n_nouns 난이도순) | (별도 서버 실행) |
| v17 | Qwen3-VL-8B | Full listwise(K=23, 샘플링 없음) | (별도 서버 실행) |
| v18 | Qwen3-VL-8B | Pairwise Bradley-Terry rank assignment | val 0.5042 (폐기) |
| v19 | Qwen3-VL-8B | diff-image adjacent-swap tiebreak | val 0.5105 (폐기) |
| **v20** | **Qwen3-VL-32B (QLoRA)** | v14 레시피 그대로, 베이스만 32B로 교체 | **LB public 0.91099 / private 0.90650, 최종 제출** |
| v21 | Qwen3-VL-32B (QLoRA) | LoRA rank 128→256 스윕 | LB public 0.89877 / private 0.90650 |

추가 참고 문서: `idea.md` (전체 작업 노트).

## 8. 결과

버전별(v1-v21) 리더보드 점수 추이. v2.5의 급락(TTA flip 실패)과 v20의 최고점 달성이 보입니다.

![LB score progression](figures/lb_score_progression.png)

최종 제출 모델 학습 곡선 (3 epoch, 6780 step).

| Loss | Learning rate | Peak GPU memory |
|---|---|---|
| ![loss curve](figures/loss_curve.png) | ![lr curve](figures/lr_curve.png) | ![vram curve](figures/vram_curve.png) |

켄달타우 거리별 loss. d=1(인접 스왑)이 다른 거리보다 일관되게 loss가 높습니다 — "d=1이 가장
어렵다"는 결론과 일치합니다.

| Per-distance loss (heatmap) | Mean per-distance loss |
|---|---|
| ![distance loss heatmap](figures/distance_loss_heatmap.png) | ![distance loss mean bar](figures/distance_loss_mean_bar.png) |

![distance loss heatmap animated](figures/distance_loss_heatmap.gif)

## 9. 라이선스

베이스 모델 Qwen3-VL-32B-Instruct는 Apache 2.0 라이선스를 따릅니다. 본 저장소의 학습/추론 코드는
대회 제출 및 재현성 검증 목적으로 공개합니다.
