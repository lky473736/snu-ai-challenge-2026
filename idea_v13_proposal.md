# v13 이후 신규 아이디어 제안 (2026-07-07)

`idea.md`, `EDA.md`, `submission_eda.md` 전체 및 v8/v10/v12 실제 코드(`loss.py`, `hard_negative.py`,
`dataset.py`, `inference.py`, `config.py`)를 다 읽고, **v1~v12에서 시도된 적 없는 축**만 골라 정리했다.
전부 "제안"이며 구현/실험은 아직 안 함 — 실행 전 반드시 규칙 3.4(데이터 누수) 및 5-4(한 번에 하나씩
검증) 원칙을 지킬 것.

---

## 0. 방법론: 뭐가 "진짜 새로운" 것인가

v1~v12를 축으로 분류하면 지금까지 건드린 레버는 4개뿐이었다:

| 레버 | 시도된 것 |
|---|---|
| Base 모델 | Qwen2VL→TPRU→Qwen3VL, rank/LR/DoRA/PiSSA/rsLoRA |
| Loss | ListNet(v2, 저커버리지) → pairwise margin(AdaptiveDistanceLoss, v4~v10) → n_noun 축 재가중치(v7/v10, 실패) |
| 데이터/샘플링 | hard negative 커버리지 확장(v3→v4), live resampling(v6.5), No_ordering 통합(v6) |
| Vision | encoder LoRA 추가(v11 frozen-LLM / v12 joint) |

**한 번도 안 건드린 축**: (1) 손실함수의 "listwise/contrastive 정규화 방식" 자체(전부 독립 pairwise
margin이었지 InfoNCE류 joint softmax는 없었음), (2) 추론 시 이미지 인코딩과 순열 열거를 분리하는
"연산 구조" 자체, (3) EDA §10-2의 n_nouns 발견을 **프롬프트/추론 전략으로 직접 전환**한 적이 아직 없음
(지금까지는 전부 loss 재가중치 시도만 했고 실패), (4) 샘플 난이도에 따라 추론 연산을 차등 배분하는
adaptive test-time compute, (5) verification 패러다임을 유지하면서 "24개 후보 분류" 대신 "프레임별
점수 회귀+정렬"로 바꾸는 아키텍처.

아래 5개를 이 순서로 제안한다. Tier 0/1은 리스크 낮고 즉시 검증 가능, Tier 2는 중간 리스크의 구조적
개선, Tier 3은 대담한 스윙(고위험/고보상, 반드시 파일럿으로 먼저 검증).

---

## Tier 0. 즉시 A/B 가능 — n_nouns 발견을 실제 개입으로 전환 (거의 공짜)

### 배경
EDA §10-2가 6가지 방법으로 확정한 유일한 독립 신호는 `n_nouns`(문장의 명사 개수, r=0.467)다. 그런데
지금까지 이 발견은 전부 **loss 재가중치**로만 활용됐고(v7/v10, 둘 다 v6.5/v8보다 저조), **"명사가
적은 문장은 텍스트 단서가 부족해서 이미지에만 의존해야 한다"**는 해석 자체를 프롬프트/추론 전략으로
바꿔본 적은 없다.

### 제안: Grounding-CoT 프롬프트
현재 프롬프트(`dataset.py`/`inference.py`의 `PROMPT_4F`)는 바로 "Yes/No만 답하라"고 요구한다. 여기에
답변 직전 한 단계를 강제로 끼워 넣는다:

```
Sentence: {sentence}

These 4 frames are presented in this exact order.
First, briefly identify the key people/objects visible in each frame and how they change
from frame to frame (1 short sentence per frame).
Then, using both the caption and what you identified, decide:
Is this the correct chronological order of events?
Answer only with "Yes" or "No" (put your final answer on the last line).
```
스코어링은 여전히 마지막 토큰의 `logP(Yes)-logP(No)`로 계산(teacher-forcing으로 "Yes"/"No" 직전까지의
CoT 텍스트를 프롬프트에 강제 주입하거나, 짧게 생성 후 마지막 줄만 파싱). 이건 **CoT는 명시적으로
허용된 추론 전략**(규칙 3.3)이고 생성형 모델로 학습 데이터를 만드는 게 아니라 추론 시점에 같은 모델이
스스로 reasoning을 거치는 것이므로 100% 규칙 안전.

**핵심 아이디어**: 문장 자체의 명사가 적어도(n_nouns 낮음), 모델이 "이미지에서 스스로 명사(대상)를
추출"하게 강제하면 EDA가 말하는 "정보 부족" 문제를 텍스트가 아니라 **이미지 쪽에서 보완**하게 된다.
n_nouns가 정확도의 원인이지 결과가 아니라는 EDA의 인과 해석(10-2)을 정확히 뒤집어서 개입하는 것.

### 검증 방법 (이미 있는 인프라 사용)
`v8_lora128/prompt_ab_test.py`, `val_temporal_diagnosis.py`가 이미 존재 — best_v8 체크포인트에 대해
기존 프롬프트 vs CoT 프롬프트를 val(476개) 전체 A/B, **특히 n_nouns 하위 구간(1-6개, EDA 표 기준
정확도 29~35%인 구간)에서만 개선폭을 따로 측정**해야 함. 전체 정확도가 아니라 "약한 구간에서 얼마나
회복하는가"가 성공 기준.

### 리스크
CoT로 토큰 수가 늘어나 추론 시간 증가(24-permutation × 819 test라 원래도 24배인데 더 늘어남) — 그러나
idea.md 6절 실측상 5epoch+추론이 24h 중 4h 정도만 썼으므로 여유 충분. 짧은 CoT(프레임당 1문장)로
제한하면 무시할 만한 증가.

---

## Tier 0-2. Sample-adaptive test-time compute (추론 연산 차등 배분)

### 배경
24시간 추론 한도 + 819개 테스트 샘플이면 연산 여유가 크다(v6 기준 5epoch 학습+추론 합쳐 4시간).
그런데 지금은 **모든 샘플에 동일한 연산**(24-permutation 1회 스코어링)을 쓴다. "어려운 샘플에 더 많은
연산을 쓴다"는 아이디어는 v1~v12 어디에도 없다.

### 제안
24개 순열 스코어를 정렬했을 때 **1위와 2위 점수 차(margin)** 를 confidence proxy로 쓴다:
- margin이 충분히 크면(모델이 확신) 그대로 제출
- margin이 작으면(헷갈림 = 애매한 케이스, EDA로 보면 대개 n_nouns 낮고 d=1 근처) 추가 연산 투입:
  - Tier 0의 grounding-CoT 프롬프트로 top-2 후보만 재채점
  - 혹은 Tier 2(아래)의 adjacent-pair 검증을 top-2에 대해서만 실행

이건 "전부에 CoT를 걸면 느려진다"는 우려를 없애면서(대다수 쉬운 샘플엔 기존 방식 그대로, 비싼 연산은
애매한 소수에만), Tier 0/Tier 2 아이디어의 실행 비용을 사실상 무료로 만드는 프레임이다. 규칙상 문제
없음(추론 전략, test 라벨 안 봄 — margin은 우리 자신의 예측 점수일 뿐 정답이 아님).

---

## Tier 1. Listwise Joint-Softmax Contrastive Loss (InfoNCE식) + 거리별 hard-negative 가중치

### 배경
현재 `AdaptiveDistanceLoss`(`loss.py`)는 거리 d별로 **독립적으로** `softplus(neg-pos)` pairwise margin을
구하고, 6개 손실을 EMA 기반 softmax 가중치로 **선형 결합**한다. 이건 "d=1 negative가 d=6 negative보다
얼마나 더 어려운지"를 서로 비교시키지 않는다 — 각 d 그룹이 positive와만 독립적으로 비교되고 그 결과값을
사후에 가중합할 뿐이다.

**한 번도 안 쓴 것**: 그룹(1 pos + 7 neg) 전체를 **하나의 joint softmax**로 정규화하는 listwise 방식
(ListMLE/InfoNCE 계열). v2가 "ListNet"을 썼다고 기록되어 있지만 그때는 d1+d6 4개 negative만 커버하는
빈약한 버전이었고, v4부터는 아예 pairwise margin으로 바뀌어 지금까지 listwise 정규화 자체가 사라졌다.

### 구체적 설계

```python
# group_logits: (8,) — [pos, neg_d1_a, neg_d1_b, neg_d2, ..., neg_d6]
# dists:        (8,) — [0, 1, 1, 2, 3, 4, 5, 6]

def listwise_loss(group_logits, dists, dist_weights, temperature=1.0):
    # dist_weights: 기존 AdaptiveDistanceLoss의 EMA-softmax 가중치 재사용 (d=1..6)
    log_w = torch.zeros_like(group_logits)
    for i, d in enumerate(dists):
        if d > 0:
            log_w[i] = torch.log(dist_weights[d - 1] * 7)  # 균등 대비 상대 가중치, pos는 0
    adjusted = group_logits / temperature + log_w   # hard negative는 partition function에서 더 큰 비중
    pos_idx = dists.index(0)
    return -torch.log_softmax(adjusted, dim=0)[pos_idx]
```

이렇게 하면 **positive가 7개 negative 전체와 한 번에 경쟁**하고, 기존에 검증된 EMA 기반 거리별
난이도 진단(5-5절, "80%의 스텝에서 d=1이 실제로 제일 어렵다고 진단됨")을 `log_w`라는 형태로 재활용해
"어려운 negative일수록 partition function에서 더 강하게 끌어당기는" 효과를 낸다. 이는 대조학습
문헌에서 hard negative weighting(예: Robinson et al. "Contrastive Learning with Hard Negative Mining",
DCL/HCL 계열)의 표준 기법과 동일한 원리이며, 지금까지 이 프로젝트가 한 번도 쓰지 않은 정규화 방식이다.

**기존 AdaptiveDistanceLoss와의 차이 요약**:
| | 기존(v4~v10) | 제안(v13 Tier1) |
|---|---|---|
| 비교 단위 | d별 독립 pairwise (pos vs 각 neg) | 그룹 전체 joint (pos vs 7 neg 동시) |
| 결합 방식 | 손실값을 사후 가중합 | 가중치를 partition function 안에 반영 |
| 이론적 근거 | BPR(Bayesian Personalized Ranking)류 | InfoNCE/ListMLE류 (retrieval·contrastive 표준) |

### 검증 순서
5-4절 교훈("한 번에 하나만 바꾼다")을 지켜, **거리 가중치는 그대로 두고 손실 정규화 방식만** 바꿔서
v8 레시피(rank128, LR5e-5) 기준으로 A/B. 만약 listwise가 이기면, 그다음 단계로 temperature 튜닝만
별도로 검증.

---

## Tier 2. d=1 전용 — "top-2 근접 시에만" adjacent pairwise 재검증 (거부됐던 pairwise 아이디어의 안전한 재활용)

### 배경
8절(idea.md)에서 pairwise 비교 방식은 **"전체를 pairwise로만 판단"** 했을 때 global narrative context를
잃는다는 이유로 기각됐다. 하지만 **이미 검증된 24-permutation 전수 스코어링(글로벌 컨텍스트 유지)을
메인으로 쓰고, pairwise는 오직 tie-break 보조 신호로만** 쓰는 조합은 시도된 적이 없다. 5-1절이 반복
확인하듯 d=1(인접 스왑) 문제는 해상도(11절)로도, loss 재가중치 temperature(v7/v10)로도, vision LoRA
학습(v11/v12)으로도 전혀 안 줄었다 — 완전히 다른 종류의 레버가 필요하다는 뜻.

### 제안
24-permutation 스코어링 후 top-1, top-2 후보의 **켄달 거리가 정확히 1**(즉 인접 두 프레임만 순서가
다름)인 경우에만, 그 두 프레임에 대해 별도 pairwise 질문을 추가로 던진다:

```
Frame X와 Frame Y 중, 어느 쪽이 시간상 먼저 일어난 사건입니까?
(문장: {sentence})
"X" 또는 "Y"로만 답하세요.
```

이 pairwise 답변이 top-1/top-2 중 어느 쪽 순서와 일치하는지로 최종 tie-break. 전체 819개 중 아주
일부(top-2 margin이 작고 d=1인 경우만)에만 적용되므로 연산 비용도 작고(Tier 0-2의 adaptive compute
프레임과 자연 결합), "pairwise 단독 사용은 안 된다"는 기존 기각 사유를 정확히 피해간다(글로벌 컨텍스트
판단이 메인, pairwise는 순수 보조).

---

## Tier 3. 대담한 스윙 (고위험/고보상 — 반드시 파일럿 검증 먼저)

### 3-A. 이미지 인코딩 1회 + 순열을 텍스트 심볼로 표현 (연산량 8~24배 절감 구조)

**배경 — 왜 지금까지 못 했는지**: 5-7절 기록: "이미지 1회 인코딩 + 후보 전체 채점 아이디어를 검토했으나
Qwen3-VL의 M-RoPE가 KV 캐시 재사용과 호환 안 돼 구현 보류." 이 실패 원인을 자세히 보면, 지금 방식은
**후보 순열마다 4장의 이미지를 물리적으로 다른 순서로 배치**한다 — 그러면 이미지 토큰의 M-RoPE
position id 자체가 순열마다 달라져서 vision 인코딩을 캐싱/재사용할 수 없다.

**핵심 통찰**: 이미지를 물리적으로 재배열하는 대신, **이미지는 항상 고정된 정준 순서(예: 파일명 알파벳
순)로 한 번만 보여주고**, 각 이미지에 중립적 라벨(A/B/C/D)을 붙인 뒤, "후보 순서"는 순수하게 **텍스트로만**
표현한다:

```
Image A: [고정 위치]  Image B: [고정 위치]  Image C: [고정 위치]  Image D: [고정 위치]
Sentence: {sentence}
Candidate chronological order: C, A, D, B
Is this the correct order? Answer "Yes" or "No".
```

이러면 이미지 토큰의 position id가 **24개 후보 전부 동일**(이미지는 항상 같은 자리)하고, 달라지는 건
뒤에 붙는 짧은 텍스트(순열 설명)뿐 — 이건 표준적인 "공유 prefix + 다른 suffix" KV-cache 재사용 패턴
(vLLM의 automatic prefix caching, 또는 HF `past_key_values`로 직접 구현)에 정확히 들어맞는다. 5-7절이
가로막힌 지점(M-RoPE와 후보별 이미지 재배치의 충돌)을 우회가 아니라 **아예 그 충돌 자체를 설계에서
제거**하는 방식.

**왜 "고위험"인가**: 지금 방식이 잘 작동하는 이유 중 하나가 "인접한 두 이미지를 실제로 나란히 재배치해서
직접 대조"하는 것일 수 있다(11절, gap=1 시각적 유사도 신호를 모델이 물리적 인접 배치로 더 잘 감지할
가능성). 심볼(A/B/C/D)로 순서를 추상화하면 모델이 "라벨→이미지" 매핑과 "심볼 순열"을 동시에 다뤄야 해서
더 어려운 task로 변할 위험이 있다.

**얻는 것 (성공 시)**: 학습 시 그룹당 8개 후보가 이미지 인코딩을 공유하면 forward 연산이 최대 8배
절감 → 지금까지 OOM으로 막혔던 것들(512px 해상도, TRAIN_MINIBATCH 16, DoRA r=128, frame-diff 이미지
v7 보류분)이 전부 다시 열림. 추론 시에도 24배 절감 → RTX 3090 단일 GPU 최종 실행 검증(idea.md에서
반복 언급되는 미해결 리스크)이 훨씬 안전해지고, 절감된 연산을 Tier 0/2의 CoT·adaptive compute에 재투자
가능.

**검증 순서 (필수, 절대 바로 재학습 들어가면 안 됨)**:
1. 기존 best_v8 체크포인트로 **재학습 없이** zero-shot 프롬프트만 바꿔서 val 정확도 비교
   (`prompt_ab_test.py` 인프라 재사용) — "심볼 기반 순서 표현을 모델이 이해라도 하는지" 먼저 확인.
2. 1에서 참사 수준으로 떨어지면 즉시 폐기(v9 PiSSA+rsLoRA 붕괴 때처럼 무리하게 밀어붙이지 말 것).
3. 성능 손실이 작다면(예: -5pp 이내) 재학습해서 손실을 만회할 수 있는지 확인 — 재학습 후에도 못
   따라잡으면 폐기.
4. 최종적으로 "정확도는 비슷한데 연산은 훨씬 쌈"이 확인되면, 그 절감분을 Tier 0/1/2에 재투자하는 v14
   설계로 이어감.

### 3-B. 프레임별 시간 점수 회귀 + 정렬 (Score-and-Sort, verification 패러다임 유지)

**배경**: v5(listwise 생성)가 실패한 근본 이유는 5-7절이 정리한 대로 "생성이 검증보다 어렵다"였다.
지금 방식(v4~v8)은 "24개 후보를 각각 검증"해서 이 교훈을 지키고 있지만, 그 대가로 조합폭발(그룹당
8~24 forward)이 생겼다.

**제안**: 이미지는 4장을 정준 순서로 한 번만 보여주고(3-A와 결합 가능), 모델이 각 이미지 위치에 대응하는
hidden state에서 **스칼라 시간 점수** 하나씩을 뽑도록 작은 회귀 헤드를 얹는다(예: 각 "Image N:" 텍스트
토큰 뒤에 특수 마커 토큰을 삽입하고, 그 마커의 마지막 레이어 hidden state를 linear head에 통과). 4개
점수를 정렬하면 바로 순열이 나온다. 학습은 여전히 "생성"이 아니라 "비교/스코어링"이라 5-7절 교훈을
어기지 않으면서도, 정답 순열 1개당 forward가 **정확히 1회**(24회도 8회도 아님)로 줄어든다.

Loss는 4개 점수에 대해 정답 순서를 만족하는 쌍별 제약(6개 쌍, `score(pos_i) < score(pos_j)` for i<j in
GT)을 로지스틱/margin loss로 걸거나, 4개 점수 전체에 ListMLE를 적용.

**왜 "고위험"인가**: 지금 방식(binary classify a full candidate hypothesis)은 모델이 "이 특정 가설이
전체적으로 말이 되는가"를 한 번에 판단하게 해주는데, 프레임별 독립 점수로 분해하면 프레임 간 상호작용
(예: "이 두 프레임을 비교했을 때만 드러나는 단서")을 못 잡을 수 있다. 검증 안 된 새 아키텍처라 실험
우선순위는 Tier 1/2보다 낮게 — **시간이 남으면** 시도할 파일럿 스터디로 제안.

---

## 실행 우선순위 요약

| 순위 | 아이디어 | 리스크 | 예상 비용 | 기존 인프라 재사용도 |
|---|---|---|---|---|
| 1 | Tier 0: n_noun grounding-CoT 프롬프트 | 낮음 | 거의 없음(재학습 불필요) | `prompt_ab_test.py` 그대로 |
| 2 | Tier 0-2: sample-adaptive test-time compute | 낮음 | 거의 없음 | 24-perm 스코어 재사용 |
| 3 | Tier 2: d=1 top-2 adjacent tie-break | 낮음 | 낮음 | inference.py 확장 |
| 4 | Tier 1: listwise joint-softmax + hard-neg weight | 중간 | 재학습 1회(v8 레시피 기준) | `loss.py`만 교체 |
| 5 | Tier 3-A: 이미지 1회 인코딩 + 심볼 순열 | 높음 | zero-shot 파일럿 먼저, 통과 시 재학습 | 신규 엔지니어링 필요 |
| 6 | Tier 3-B: score-and-sort 회귀 헤드 | 높음(미검증 아키텍처) | 신규 헤드 학습 | 신규 아키텍처 |

**원칙(5-4절 재확인)**: 1~3은 재학습 없이 기존 best_v8/v6.5 체크포인트로 즉시 A/B 가능하니 가장 먼저
실행. 4는 검증되면 v8 레시피에 그대로 얹어서 v13으로. 5/6은 각각 독립 실험으로 격리하고, 하나가
실패해도 다른 하나와 원인이 섞이지 않게 절대 동시에 진행하지 말 것.

## 규칙 준수 체크 (전 항목 공통)

- 외부 데이터/생성형 모델 데이터 증강 없음(전부 같은 모델의 추론 전략 변경이거나 loss 재정의) ✅
- 앙상블 아님(단일 모델, 단일 체크포인트) ✅
- test 라벨 사용 없음 — margin/confidence는 우리 예측값이지 정답이 아님 ✅
- 모든 아이디어의 근거는 train/val 내부 EDA(§10-2 n_nouns, §11 이미지 유사도, §5-1 d=1)에서만 도출 ✅
- Tier 3-A/3-B는 재학습을 동반하므로 최종 3090 단일 GPU 실행 가능 여부를 v8/v10처럼 반드시 별도 검증
