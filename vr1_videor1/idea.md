# HNTV (Hard Negative Temporal Verification) — 작업 노트 / 아이디어

SNU AI Challenge 2026 출품 프로젝트. 4장의 비디오 프레임 + 설명 문장을 입력으로 받아
"이 프레임 순서가 사건의 올바른 시간 순서인가?"를 Qwen2-VL-7B(+LoRA)로 Yes/No 판별하는 모델.

## 1. 모델 구조 요약

- Base: Qwen2-VL-7B-Instruct + LoRA (r=32, alpha=64, dropout=0.05, target: q/k/v/o_proj, gate/up/down_proj)
- 입력: 4장 프레임(temporal 순서대로 배열) + Sentence → "Yes"/"No" 답변
- 점수: log P(Yes) − log P(No) (log-odds ratio)
- Hard Negative: 정답 순서에서 **인접 위치 swap** 3가지(0-1, 1-2, 2-3)만 사용
- Loss: BCE(전체) + Margin Ranking Loss(그룹 내 positive vs 각 hard negative, margin=0.8)
- Inference: 24개(4!) 순열 전부 스코어링 → 최고 점수 순서를 정답으로 제출

## 2. 인프라 작업 기록

- 데이터: Kaggle 대회 `snuaichallenge`를 `kagglehub.competition_download()`로 받아서
  `/data/gyuyeonlim/hntv/data/snuaichallenge_data`에 위치 (train 4500여개 / test 819개, 각 4장 이미지)
- 모델: `Qwen/Qwen2-VL-7B-Instruct`를 HF에서 새로 받아 `/data/gyuyeonlim/hntv/models/`에 위치
- `config.py`의 `DATA_DIR`, `MODEL_PATH`를 프로젝트 상대 경로로 수정 (원래 SLURM 클러스터 절대경로였음)
- 이 서버(4farm)의 SLURM 제출 관례 파악 후 `run_train.sh` / `run_inference.sh` 수정:
  - account=`gpu`, partition=`gpu-4farm`, conda init=`/opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh`
  - **QOSMinGRES**: 이 클러스터의 `gpu-4farm` QOS는 **최소 GPU 4개**를 요구함 — `--gres=gpu:1`처럼 적게 요청하면
    GPU가 비어 있어도 영원히 PD(pending) 상태로 남음. inference처럼 GPU 1개만 써도 되는 작업도 **무조건 `gpu:4`로 요청**해야 스케줄링됨.
  - torch가 2.12.1(cu130)로 이미 깔려있었는데 torchvision이 없어서 `Qwen2VLVideoProcessor` 로드 실패 →
    `pip install torchvision`으로 해결 (0.27.1 설치됨)

## 3. 제출 1 결과

- `/data/gyuyeonlim/hntv/data/submission.csv` 제출
- **Kaggle 리더보드 score: 0.79057, 현재 1위** (2위 Whale 0.70680)
- 819개 test 샘플, 24-permutation exhaustive search로 추론 (약 65분 소요)

## 4. Val 진단 (val_diagnosis.csv, `src/diagnose_val.py`)

held-out val set(`checkpoints/_val_raw.csv`, ordering 있는 샘플 399개)에 대해
best checkpoint로 24-permutation 채점 → 정답(gt)과 예측(pred) 간 **켄달타우 거리**(필요한 adjacent swap 횟수)로 오답 분류.

### 핵심 수치
- **Val exact-match 정확도: 50.1%** (200/399)
  - 리더보드 점수(0.79)보다 훨씬 낮음 → **Kaggle 채점 metric이 exact-match가 아니라 부분점수(partial credit) 방식일 가능성이 높음**. 확인 필요.
- 오답 199개의 켄달타우 거리 분포:

  | 거리 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | 오답 수 | 44 | 26 | 60 | 32 | 33 | 4 |

- **chance(무작위 추측) 대비 정규화**가 핵심 발견:
  - n=4 순열 공간에서 distance=1(인접 스왑)인 오답 후보는 23개 중 3개(13%)뿐인데, 실제 오답의 22%가 distance=1
    → **모델이 가장 많이 틀리는 카테고리는 여전히 "인접 스왑"** (chance 대비 약 1.7배)
  - distance=6(완전 역순)은 4.3% 예상 → 실제 2% (오히려 잘 거름, 당연한 결과)
  - → 즉 **이미 학습 때 집중적으로 다룬 바로 그 hard negative 유형(인접 스왑)에서 여전히 가장 약함**
- gt_rank(정답이 24개 중 몇 등으로 채점됐는지) 분포:
  - 61/199(31%): 정답이 2등 — 아깝게 진 "근접 오답"
  - 9개: 정답이 21~23등(거의 꼴찌) — 모델이 장면/문장을 완전히 잘못 이해한 케이스로 추정. 라벨 오류 가능성도 있음.

### gt_rank≥20 (극단 오답) 10개 케이스 수동 확인 결과
- 공통점: 대부분 **문장이 짧고 모호함** (예: "She throws the javelin and then it is measured.",
  "He pours ice into a glass, then adds several kinds of liquor.") — 사건을 1~2개만 묘사하는데
  4프레임 순서를 다 맞춰야 해서 텍스트만으로는 정답이 잘 특정되지 않는 케이스들.

### 추가 발견: 문장 길이 ↔ 정확도 강한 상관관계
val 전체(399개)에서 문장 단어수와 exact-match 정확도를 비교:

| 문장 단어 수 | 정확도 | 샘플 수 |
|---|---|---|
| ≤15 | 26.5% | 98 |
| 15-20 | 20.5% (최저) | 39 |
| 20-25 | 58.5% | 41 |
| 25-30 | 51.0% | 102 |
| 30-35 | 73.8% | 80 |
| 35+ | 79.5% | 39 |

- 상관계수(sent_len vs correct) = **0.376** (꽤 뚜렷한 양의 상관)
- 정답 평균 27.8단어 vs 오답 20.8단어 vs 극단오답(gt_rank≥20) 17.4단어 — 일관된 추세
- 예측 순열 분포는 고르게 분산되어 있어 mode-collapse(특정 순열만 찍는 편향)는 아님 (최빈 예측도 6.5%에 불과)

**해석**: hard negative 다양화나 margin 조정 같은 학습 방식 문제가 아니라, **문장이 짧을수록 텍스트만으로 시간 순서를 특정하기 어려운 구조적 약점**으로 보임.

### 추가 발견: train/val과 test의 문장 길이 분포가 크게 다름 (분포 시프트)

| | 평균 단어 수 | 25%/50%/75% |
|---|---|---|
| train.csv 전체 (9535개) | 24.2 | 15 / 26 / 32 |
| val (_val_raw.csv, 476개) | 24.3 | 16 / 26 / 32 |
| **test.csv (819개)** | **42.0** | **27 / 33 / 64** |

test 문장이 train/val보다 **평균 거의 2배 김**. 우리가 찾은 "문장 길이 ↑ → 정확도 ↑" 관계를 적용하면,
test는 val보다 원래 더 "쉬운" 분포일 가능성이 큼.

**검증**: val의 구간별 정확도를 test의 문장 길이 분포로 재가중치(reweight)해서 추정한 결과:

```
구간      val_acc   test_비율
<=15      26.5%     10.3%
15-20     20.5%      3.9%
20-25     58.5%      5.9%
25-30     51.0%     18.7%
30-35     73.8%     18.8%
35+       79.5%     42.5%

재가중치 추정 정확도: 64.1%  (원래 val 단순평균: 50.1%)
```

→ leaderboard 0.79057과 val 50.1%의 격차 중 상당 부분(약 14%p)은 **test가 train/val보다 문장이 길어 더 쉬운 분포**라는 것으로 설명됨. 남은 격차(64% → 79%)는 Kaggle 채점이 exact-match가 아닌 부분점수 방식일 가능성으로 설명 가능.

## 5. 개선 아이디어 (논의 순서대로, 데이터 근거 포함)

### 5-1. (처음 가설, 데이터로 기각/보류) Hard negative 다양화
- 가설: 학습은 인접 스왑만 보는데 inference는 24개 전부 비교하니 train-test mismatch가 있을 것 → 더 다양한(먼) negative를 섞어 학습하면 좋아지지 않을까?
- **검증 결과**: 위 진단에서 모델은 먼 순열(distance↑)은 chance보다 오히려 잘 구분하고, **여전히 가장 취약한 건 인접 스왑(distance=1)**. 즉 다양화보다 "이미 다루고 있는 어려운 케이스를 더 세게" 학습하는 쪽이 우선순위가 높아 보임.
- 우선순위: 낮음 (보류, 나중에 재검토 가능)

### 5-2. 인접 스왑 negative를 더 세게 학습
- `MARGIN`(현재 0.8) 상향 또는 `RANKING_WEIGHT`(현재 1.0) 상향으로 인접 스왑에 대한 페널티 강화
- 혹은 인접 스왑 3종 중 손실이 가장 안 줄어드는 위치(0-1 vs 1-2 vs 2-3)가 있는지 확인 후 가중치 차등 부여
- 아직 미실행

### 5-3. (완료, 우선순위 1순위 발견) 짧은/모호한 문장에서의 약점 보완
- gt_rank≥20 케이스 수동 확인 + val 전체 통계로 **문장 길이가 정확도와 상관계수 0.376**으로 강하게 연관됨을 확인
  (≤15단어: 26.5% 정확도 / 35+단어: 79.5% 정확도). 자세한 내용은 위 "추가 발견" 절 참고.
- 이게 지금까지 발견한 것 중 **가장 설명력이 큰 약점**. 5-1, 5-2보다 잠재적 효과가 클 것으로 보임.
- 시도해볼 방향 (아직 미검증):
  - 짧은 문장 샘플에 한해 학습 시 가중치를 더 주거나, 같은 샘플을 더 많이 반복 노출
  - 짧은 문장에 대해 시각적 단서에 더 의존하도록 프롬프트 조정 (예: "문장이 짧으면 프레임 간 미세한 시각적 변화에 더 집중하라" 같은 지시 추가)
  - test set에서도 문장 길이 분포를 확인해서 실제로 짧은 문장 비율이 얼마나 되는지, 그게 점수에 미치는 영향 추정

### 5-4. 기타 가벼운 시도 (아직 미검증)
- `MAX_IMAGE_SIZE`(현재 448) 상향 → 시각적 디테일 증가, 단 메모리/속도 트레이드오프
- 여러 체크포인트/시드 앙상블
- 리더보드 metric이 부분점수 방식이라면, exact-match보다 **켄달타우 거리 자체를 줄이는 방향의 loss**(예: 모든 인접 쌍에 대해 거는 listwise ranking)를 고려

### 5-5. (기각) Pairwise 비교 + 토너먼트/정렬 방식 추론
- 아이디어: 지금은 "4장 전체가 통째로 맞는 순서냐"를 절대평가(24-permutation exhaustive search)하는데,
  모델 입장에선 "두 프레임 중 어느 게 먼저냐"는 pairwise 비교가 더 쉬울 수 있음. 정렬 알고리즘처럼
  pairwise 비교를 모아서 순서를 추론하면 24번 forward 대신 훨씬 적은 비교(예: 6번)로 충분하고
  정확도도 오를 가능성이 있다고 제안했었음.
- **기각 사유**: pairwise 비교 모델은 두 프레임만 보고 판단하므로 **문장 전체가 묘사하는 4프레임짜리
  global narrative(전역 사건 흐름)를 학습할 수 없음**. 예를 들어 문장이 "A→B→C→D" 4단계 사건을
  순서대로 묘사하는데, pairwise로 (A,C)만 비교하면 그 사이에 B가 끼어있다는 문맥 정보를 활용하지
  못함. 지금처럼 4장을 한 번에 보고 문장 전체와 대조하는 구조라야 이런 전역 일관성을 판단할 수 있음.
- 우선순위: 보류 (구조를 통째로 바꿔야 해서 작업량도 크고, 위 기각 사유가 본질적인 문제라 단순
  구현 이상의 고민이 필요함)

## 6. 다음 액션

- [x] `val_diagnosis.csv`에서 `gt_rank >= 20`인 10개 케이스 수동 확인 → 문장 길이 약점 발견
- [x] test set 문장 길이 분포 확인 → train/val 대비 거의 2배 길고, 분포 시프트가 leaderboard-val 격차의 상당 부분을 설명함
- [x] 짧은 문장 샘플 가중치 + 프롬프트 힌트 추가 후 재학습 실험 시작 (→ 실험 1, 실패)
- [x] MARGIN=1.2 + RANKING_WEIGHT=1.5 + LoRA r=64 조합 재학습 실험 시작 (→ 실험 2, 진행 중)
- [ ] Kaggle 채점 metric이 정확히 무엇인지 확인 (exact match vs partial credit)

## 7. 실험 1 (v2): 짧은 문장 보완 재학습 — 실패

### 변경 내용
기존 설정은 거의 그대로 두고 5-3에서 찾은 약점만 타겟으로 최소 변경:

1. **`config.py`**: `SHORT_SENTENCE_WORD_THRESHOLD = 20`, `SHORT_SENTENCE_OVERSAMPLE_WEIGHT = 1.5` 추가
2. **`src/dataset.py`**: `build_messages()`의 프롬프트에 조건부 힌트 추가 — 문장이 20단어 이하면
   "The sentence is brief... rely more heavily on subtle visual cues..." 문구를 프롬프트에 삽입.
3. **`src/train.py`**: `DataLoader`의 `shuffle=True`를 `WeightedRandomSampler`로 교체 — 문장이 20단어
   이하인 그룹(전체 7658개 그룹 중 약 34.5%)에 1.5배 샘플링 가중치 부여.

**안전장치**: 재학습 시작 전 기존 1위 체크포인트를 `checkpoints/best_v1_score0.79057`로 백업.

### 학습 진행 (Job 225461)
- Epoch 1 완료: val_acc = 0.3700
- Epoch 2 완료: val_acc = 0.4000 ← Best 저장
- Epoch 3 진행 중 → **도중에 중단** (epoch 2 체크포인트로 inference 제출 결정)

### 제출 결과 (Job 225486)
- **리더보드 score: 0.76** — v1(0.79057) 대비 **하락**

### 실패 원인 분석
1. **미수렴 상태로 제출**: epoch 2에서 val_acc=40%로 v1 최종(50.1%)보다 낮은 상태. epoch 5까지 돌렸다면 비슷하게 회복됐을 수도 있으나 검증 못함.
2. **잘못된 타겟**: val에서 짧은 문장(≤20단어)이 문제였지만, test 문장은 >40단어가 30.4%나 됨. 짧은 문장을 더 많이 보게 하는 것은 test 분포에 역효과 가능성이 있음.
3. **WeightedRandomSampler**: 짧은 문장 비율을 올리면 상대적으로 중간/긴 문장 학습 기회가 줄어들어 test 성능에 불리했을 수 있음.

### 교훈
- val에서 발견한 약점이 test에서도 같은 약점이 아닐 수 있음 (분포 시프트 때문)
- 실험을 중도에 멈추고 제출하는 것은 위험 — 5 epoch 완료 후 판단해야 함

---

## 8. 실험 2 (v3): MARGIN + LoRA rank 동시 개선 — 진행 중

### 변경 내용 (v1 기준 클린 베이스라인 + 2가지 동시 적용)

1. **MARGIN**: 0.8 → **1.2** (인접 스왑에 대한 페널티 강화)
2. **RANKING_WEIGHT**: 1.0 → **1.5** (ranking loss 비중 상향)
3. **LoRA r**: 32 → **64**, **alpha**: 64 → **128** (모델 표현력 2배)
4. WeightedRandomSampler 제거, SHORT_SENTENCE_HINT 제거 (v1 클린 베이스 유지)
5. 체크포인트 저장: `checkpoints/best_v3`

코드 변경:
- `src/train.py`: `argparse` 추가 (`--margin`, `--ranking_weight`, `--lora_r`, `--lora_alpha`, `--ckpt_name`)로 CLI에서 하이퍼파라미터 오버라이드 가능하게 일반화
- `src/model.py`: `load_model_and_processor(lora_r, lora_alpha)` 파라미터 추가
- `run_train_v3.sh`: 신규 생성

**SLURM Job**: 225491 (aic_hntv_v3), PENDING/Priority — node 자리 나면 바로 실행 예정

### 기대 효과 및 불확실성

**MARGIN/RANKING_WEIGHT 상향:**
- 인접 스왑 혼동(val 진단에서 chance 대비 1.7x)을 직접 타겟
- 문장 길이에 무관하게 모든 샘플에 적용되는 범용 개선
- 확신도: 중간 — 실제로 margin이 부족해서 틀리는 건지, 아니면 시각적으로 구분이 안 되는 건지 불분명

**LoRA r=64:**
- 미세한 프레임 간 차이 학습에 더 많은 파라미터 활용
- 확신도: 낮음 — r=32로 이미 0.79까지 도달했으므로 rank 증가의 한계이익 불분명. 오히려 train 분포 과적합 위험 있음.

**근본 한계**: train/val 평균 24단어 vs test 평균 42단어, >40단어 비율 1% vs 30.4% — 분포 시프트는 이번 실험으로도 해결되지 않음.

---

## 10. 대회 규칙 정리 및 전략적 시사점 (2026-07-01)

### 핵심 규칙 요약

| 항목 | 내용 |
|---|---|
| 평가 지표 | **Exact Match Accuracy** — 순서 하나라도 다르면 0점, 부분점수 없음 |
| 리더보드 | 예선 중 **Public 70%**만 반영. 예선 종료 후 전체(Public+Private) 기준 최종 순위 결정 |
| 외부 데이터 | **금지** — 제공된 학습 데이터만 사용 가능 |
| 외부 API | 추론 과정 금지. **데이터 전처리 목적 한해 허용** (총 3만 원 이하) |
| 모델 앙상블 | **금지** — 여러 모델 조합 불가 (같은 모델 여러 번 학습 후 조합도 불가) |
| 추론 전략 | Chain-of-Thought, Multi-turn Chat, **Test Time Augmentation(TTA) 허용** |
| 추론 시간 | test set 전체(819개)에 대해 **24시간 이내** 완료 필요 |
| 최종 제출 환경 | **NVIDIA RTX 3090 (24GB VRAM) 1개**, 1-GPU 단독 실행 가능해야 함 |
| 모델 크기 | 가중치 포함 전체 **80GB 이하** |
| 사전학습 모델 | **2026년 5월 31일 이전 공개된 가중치**에 한해 허용 (Qwen2-VL-7B: 2024년 공개 → ✓) |
| 데이터 누수 | 평가 데이터 특성 분석 후 학습/설계에 활용 **금지** |

### 우리 현황과의 대조

**확인된 것:**
- 리더보드 0.79057 = 70% public test에서의 **exact match** (부분점수 아님). 이전에 혼동했던 부분 정정.
- Qwen2-VL-7B + LoRA 체크포인트(~14GB + ~320MB) → 80GB 제한 내 ✓
- 현재 inference.py는 single GPU로 실행되므로 RTX 3090 1개 환경과 호환 가능성 있음

**주의가 필요한 것:**
1. **RTX 3090 24GB 호환성**: bfloat16 7B 모델(~14GB) + 4장 이미지(448px) 입력의 activation 메모리 합산이 24GB에 들어가는지 **반드시 검증 필요**. LoRA r=64(v3)는 파라미터 수는 늘지 않지만 activation이 커질 수 있음.
2. **데이터 누수 잠재 위험**: test 문장 길이 분포를 분석해서 v2 short-sentence 오버샘플링 설계에 활용했음. 이는 규칙("평가 데이터의 특성을 분석하여 학습 데이터 전처리에 활용")에 걸릴 수 있는 회색지대. v3에서 해당 방식을 제거한 것은 규칙 준수 측면에서도 적절한 결정.
3. **앙상블 금지**: 여러 체크포인트나 시드로 학습 후 조합하는 방식 불가. 단일 모델로 승부해야 함.

### 규칙 기반 새로운 개선 방향

#### (가능) TTA (Test Time Augmentation)
- 추론 전략은 허용된다고 명시됨
- 예시: 4장 이미지의 좌우 반전(flip) 버전도 동시에 스코어링해서 평균, 또는 다른 이미지 해상도로 두 번 추론 후 점수 평균
- 주의: 819샘플 × 24순열 × 추론 횟수가 24시간 이내여야 함 (현재 단일 기준 ~65분이므로 TTA 2~4배 여유 있음)

#### (실험 중) Reasoning Prefix Injection — 재학습 없이 CoT 효과 흉내

핵심 아이디어: 두 단계 inference로 CoT 효과를 근사

1. **1단계 (reasoning 생성)**: `build_messages` 프롬프트 + `add_generation_prompt` 뒤에
   `"Let me analyze the changes between consecutive frames: "` seed를 붙인 후
   `model.generate(max_new_tokens=80)`으로 추론 텍스트를 생성시킴
2. **2단계 (Yes/No 측정)**: 원본 프롬프트 + 생성된 추론 + `"\nFinal answer:"` 를 이어 붙인 후
   기존 방식대로 Yes/No logit 측정

구현: `src/inference_cot.py`, `run_inference_cot.sh`
- 체크포인트: `best_v1_score0.79057` (현재 최고 기록)
- fallback: 생성된 reasoning이 5토큰 미만이면 기존 Yes/No logit으로 자동 대체
- 추론 시간: 기존 대비 약 2~3배 증가 예상 (max 8시간 설정)

**불확실성:**
- LoRA fine-tuning이 base 모델의 reasoning 생성 능력을 얼마나 약화시켰는지 불명
- 생성된 reasoning이 틀릴 경우 오히려 Yes/No 판단을 방해할 수 있음
- "Final answer:" 이후 Yes/No가 명확히 예측되는지 불확실 (fine-tuning context와 다름)

결과는 제출 후 리더보드 점수로 판단

#### (재학습 필요) Chain-of-Thought (CoT) / 직접 순서 예측
- 현재 모델은 LoRA로 **Yes/No logit 스코어링 전용**으로 파인튜닝됨. 프롬프트만 CoT 형식으로 바꿔도:
  - LoRA 가중치가 모델을 Yes/No 방향으로 밀어붙여 추론 체인 생성 능력이 degraded됐을 가능성 높음
  - 우리가 쓰는 스코어는 마지막 토큰의 log P(Yes) - log P(No) → 중간 추론 텍스트가 스코어에 반영되지 않음
- 제대로 적용하려면 **학습 포맷 자체를 변경**해야 함:
  - "Answer only Yes/No" → "Reason step by step, then output [n,n,n,n]"
  - Loss도 cross-entropy(생성) 방식으로 재설계
  - 즉 전면 재학습 필요
- 잠재적 장점: 1번 forward pass로 정답 직접 생성 (24배 빠름), 모델 추론 능력 활용
- 단점: 학습 설계 복잡도 높음, 출력 파싱 필요, 현재 방식(24순열 exhaustive)보다 불안정할 수 있음

#### (가능, 비용 주의) API로 학습 데이터 문장 증강
- 짧은 훈련 문장(≤20단어)을 외부 API로 paraphrase/확장 → 더 긴 표현으로 학습
- 규칙: 전처리 목적 허용, 3만 원 이하
- 주의: test 문장 길이 분포를 보고 설계하면 데이터 누수로 간주될 수 있음 → train 내부 분포만 보고 설계해야 안전

#### (확인 필요) No_ordering 샘플 처리
- train에 `No_ordering=True` 샘플이 **1478개(15.5%)** 존재 — 이 샘플들은 프레임이 셔플되지 않아 항상 정답이 [1,2,3,4]
- 현재 학습에 포함되어 있음. 이 샘플들이 hard negative 학습에 방해가 되는지 확인 필요
- test.csv에는 No_ordering 컬럼이 없음 → test에 이런 샘플이 있어도 우리 방식(24순열 스코어링)으로 자연스럽게 처리 가능

### 현재 최우선 액션

1. **Reasoning Prefix Injection inference 실험** (Job 제출 예정, v3 취소 후 우선 시도)
2. **RTX 3090 호환성 테스트**: inference.py가 1-GPU 24GB에서 OOM 없이 돌아가는지 확인
3. 결과에 따라 v3 재학습 또는 추가 CoT 개선 결정

---

## 9. 문장 길이 분포 상세 분석 (2026-07-01)

```
=== Train (n=9535) ===
  mean=24.2  median=26  std=9.5  min=5  max=69
  ≤20단어: 3283 (34.4%)
  21~40단어: 6154 (64.5%)
  >40단어:  98 (1.0%)
  percentile [10,25,50,75,90,95]: [10, 15, 26, 32, 35, 37]

=== Test (n=819) ===
  mean=42.0  median=33  std=24.3  min=6  max=116
  ≤20단어: 116 (14.2%)
  21~40단어: 454 (55.4%)
  >40단어:  249 (30.4%)
  percentile [10,25,50,75,90,95]: [15, 27, 33, 64, 79, 87]
```

**해석:**
- mean=42 는 극단값(max=116)에 끌린 것. median 기준으론 train 26 vs test 33으로 격차가 줄어듦.
- 21~40단어 구간은 train(64.5%) / test(55.4%)로 상당히 겹치는 공통 구간.
- **진짜 문제 구간은 >40단어**: train 1% vs test 30.4% — 이 구간은 모델이 학습 중 거의 못 봤음.
- test std=24.3 (train 9.5의 2.6배) — test 문장 길이 자체가 매우 불균일함.

**Data Augmentation 가능성 검토:**
- 방향: 짧은 train 문장을 LLM으로 길게 paraphrase해서 >40단어 구간 커버
- 문제: 문장이 실제 영상 내용을 묘사해야 하므로, 모르는 영상에 대해 임의로 길게 늘리면 **사실과 다른 내용이 삽입될 위험** 있음 ("a person sits down" → 없는 행동이 묘사될 수 있음)
- Paraphrase-only(내용 안 늘리고 표현만 길게): 효과 제한적
- **결론: Aug는 구현 복잡도 대비 효과 불확실, v3 결과 보고 재검토**

---

## 11. 실험 3 (TPRU-7B): Temporal-specialized Base + LoRA — 설계 중 (2026-07-01)

### 배경 및 동기

**TPRU (Temporal and Procedural Reasoning with Reinforcement Learning)**
- 논문: arXiv:2602.18884, ICLR 2026 (Spotlight)
- 공개일: 2026-02-12 → 대회 규칙(2026-05-31 이전 공개) **충족**
- Base: **Qwen2.5-VL-7B-Instruct** (Qwen2-VL-7B의 후속 버전)
- 학습: GRPO (RL) 기반으로 temporal reordering task에 특화 fine-tuning
- HuggingFace: `Stephengzk/TPRU-7B`

우리 HNTV 태스크와 TPRU의 학습 목표가 거의 동일 — 비디오 프레임들의 시간 순서를 이해하는 것. 
TPRU는 이미 RL로 temporal reordering을 학습했으므로, 우리 LoRA의 출발점으로 쓰면:
- **TPRU가 이미 가진 것**: temporal reordering을 위한 시각-언어 정렬, 프레임 간 변화 감지 능력
- **우리 LoRA가 추가하는 것**: 이 대회 도메인(일반 영상 클립)에 대한 적응 + Yes/No binary 판별 형식

### 창의성 평가

경쟁팀 대비: 대부분 Qwen2-VL / Qwen2.5-VL을 직접 fine-tuning할 것으로 예상.
TPRU-7B는 같은 태스크에 RL 사전학습까지 된 모델을 시작점으로 쓴다는 점에서 차별화.
보고서 관점: "RL pre-aligned temporal reasoning + discriminative domain LoRA" 파이프라인으로 설명 가능.
학술적 novelty는 낮지만 대회 맥락에서는 영리하고 유효한 선택.

### 핵심 설계 결정

| 항목 | 값 | 이유 |
|---|---|---|
| Base 모델 | TPRU-7B (Qwen2.5-VL) | temporal RL 사전학습 |
| LR | **1e-5** (기존 2e-5의 절반) | TPRU의 temporal 지식 보존 (catastrophic forgetting 방지) |
| LoRA r | **32** (v3의 64보다 작게) | conservative: base 지식 유지 우선 |
| MARGIN | **1.2** | v3 설정 그대로 |
| RANKING_WEIGHT | **1.5** | v3 설정 그대로 |
| EPOCHS | 5 | 동일 |

### 구현

- 폴더: `/data/gyuyeonlim/hntv_TPRU/` (별도 독립 프로젝트)
- `config.py`: MODEL_PATH → `/data/gyuyeonlim/models/TPRU-7B`
- `src/model.py`: `_get_model_class()` — AutoConfig.model_type으로 Qwen2.5-VL vs Qwen2-VL 자동 감지  
  (현재 env에서 클래스명은 `Qwen2_5_VLForConditionalGeneration`)
- `src/train.py`: `--lr` 인자 추가로 LR CLI 오버라이드 가능
- `run_train_tpru.sh`: SLURM 스크립트 (4 GPU, 20h)

### 진행 상황

- [x] `Stephengzk/TPRU-7B` 모델 ID 확인
- [x] `hntv_TPRU/` 폴더 구조 생성, 코드 수정 완료
- [ ] TPRU-7B 다운로드 중 (PID 2792428, ~14GB, `logs/download_tpru.out` 로그)
- [ ] 다운로드 완료 후 `sbatch run_train_tpru.sh` 제출
- [ ] 학습 완료 후 inference 및 리더보드 제출

### TPRU 전이 가능성 추가 분석 (논문 ablation 기반)

**긍정적 증거 (MuirBench 독립 벤치마크):**
- MuirBench "Ordering" 서브태스크(Qwen/레고/GUI 외 일반 도메인): 14.06% → **34.38%** (2배 이상)
- 학습 도메인 밖에서도 "순서 판별 스킬"이 전이됨 = 도메인 특화 지식이 아닌 범용 판별 능력
- 비교: Qwen2.5-Omni 등 자연 영상 사전학습 모델은 이 서브태스크 최고 18.8% — TPRU의 34.38%보다 한참 낮음
  → "사람 행동 영상을 많이 본 것"이 아니라 "hard negative로 명시적 훈련"이 핵심

**핵심 ablation 결과:**
- hard negative 제거 시 LEGO-Puzzles + MuirBench 성능 급락
- TPRU의 장점 = 인접/모순 순서 샘플을 거부하도록 강제 학습 → **우리 margin ranking loss와 동일 원리**
- 전이되는 것은 "시각적 지식"보다 "hard negative 판별 방법론"

**불확실한 것:**
- EPIC-KITCHENS(부엌 행동) 서브셋 점수가 논문에 미공개 (전체 평균 50.33%→75.70%만 있음)
  → 대회 도메인(일반 생활 영상)에서의 정확한 성능은 실험으로만 확인 가능

### Zero-shot 기준선 비교 (확정)

| 모델 | Zero-shot val acc | 비고 |
|---|---|---|
| Qwen2-VL-7B base | **5.5%** | random 4.17%와 거의 동일, fine-tuning 필수 |
| TPRU-7B base | **26.3%** (105/399) | temporal RL로 6× 향상 |
| Qwen2-VL v1 fine-tuned | **50.1%** (200/399) | LoRA 학습 후 |

→ TPRU zero-shot이 Qwen2-VL base 대비 **+20.8%p** — temporal RL pre-training이 실질적인 task-relevant knowledge를 심어줬음을 확인.
→ TPRU fine-tuning 출발점(26.3%)이 높으므로 v1(5.5%→50.1%) 대비 수렴 빠르거나 최종 성능 더 높을 가능성 있음.

### 실험 순서 (수정)

**1단계: Zero-shot 기준선 확인 (완료)**
- TPRU-7B zero-shot = 26.3%, Qwen2-VL base zero-shot = 5.5%
- TPRU의 temporal RL 사전학습이 유효함 확인 → fine-tuning 진행

**2단계: Fine-tuning (SLURM)**
- `run_eval_zero_shot.sh` → `run_train_tpru.sh` 순서
- 목표: zero-shot 기준선 + fine-tuning delta 합이 v3(0.7x 이상)를 넘기

### 불확실성

- Qwen2.5-VL vs Qwen2-VL 아키텍처 차이로 LoRA target module 명칭이 달라질 수 있음 (학습 시 오류 나면 확인 필요)
- TPRU가 이미 "정렬 방향"을 강하게 학습했을 경우, 우리 BCE+ranking loss로 추가 학습할 때 충돌 가능성
- LR 1e-5이 너무 낮으면 수렴 느림, 너무 높으면 TPRU 사전지식 손상 — epoch당 val_acc 추이로 판단
