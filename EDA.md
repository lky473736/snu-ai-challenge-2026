# HNTV Dataset EDA (전면 재작성판)

분석 이력:
- 2026-07-02: 최초 EDA (기본 구조, 문장 길이, 합성 문장, temporal word 초판, 기각 아이디어)
- 2026-07-04: 심층 EDA v2 (이미지 시각적 유사도 Cohen's d, 문장구조 vs gt_kendall, position bias, near-duplicate)
- 2026-07-05: **전면 재작성 + 확장** —
  - v3: spaCy 기반 temporal marker 정밀 재검증 (10→35단어, 오탐 제거)
  - v4: POS/시제/의존구문/TTR 전체 구조 분석
  - v5: **실제 학습된 모델(v6.5)의 val 정확도와 다변량(NLP+이미지) 상관관계 종합 분석** — 기존엔
    gt_kendall(정답 자체의 복잡도)과만 비교했었는데, "모델이 실제로 맞히는지"와 비교한 건 이번이 처음
  - v6: NER/명사구/가독성/구두점/형태소/의존관계/trigram/어휘빈도까지 train 전체(9,535개) 최대치 분석

데이터 경로: `/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data/`

---

## 0. 방법론 원칙 (★ 반드시 읽을 것)

이 문서의 모든 분석은 두 종류로 나뉜다:

1. **train/val 내부 분석** (train.csv 및 그 5% 분할인 val) — 대회 규칙상 완전히 자유롭게 사용 가능,
   학습 설계에 직접 반영해도 무방함.
2. 평가 데이터 특성 분석은 규칙 3.4(데이터 누수: "평가 데이터 특성 분석 후 학습 설계에 반영 금지")에
   저촉되므로, 본 문서에는 **일체 기재하지 않는다**. 과거 v2에서 이와 관련된 실수를 저지른 전례가
   있었으나(이미 제출된 건 되돌릴 수 없음) v3부터는 완전히 배제했고, 이 문서 자체에서도 관련 통계를
   전부 제거했다.

**실전 규칙**: 어떤 발견을 학습 설계에 반영하려면 반드시 "①에서 나온 결과인가?"를 self-check할 것.

---

## 1. 기본 구조

```
train: 9,535개 | columns: Id, Input_1~4, Sentence, Answer, No_ordering
test:    819개 | columns: Id, Input_1~4, Sentence   (Answer 없음)
```

**Answer 포맷**: `[a, b, c, d]` — Input_i가 시간순 몇 번째인지
예) Answer=[3,1,2,4] → Input_1은 3번째, Input_2는 1번째(가장 먼저), Input_3은 2번째, Input_4는 4번째

**파일명 구조**: `{Id}_{랜덤3글자}.jpg` (알파벳 순 정렬로 Input_1~4 대응)

---

## 2. No_ordering 분포 [train 내부]

```python
train['No_ordering'].value_counts()
# False    8057  (84.5%)
# True     1478  (15.5%)
```

- **No_ordering=True (1478개)**: 프레임이 이미 시간순 → 답이 **항상 [1,2,3,4]**
- **No_ordering=False (8057개)**: 프레임이 섞여 있음 → 답이 **절대 [1,2,3,4] 아님 (0.0%)**

두 그룹은 완전히 다른 분포이며, v6/v6.5는 이 둘을 **동일하게 hard-negative 대조학습** 시킨다
(구버전의 "No_ordering은 positive만 5배 증강" 방식은 검증 안 된 가정이라 폐기됨 — 5-6절 참고).

---

## 3. 정답(Answer) 분포 — No_ordering=False (8057개) [train 내부]

24가지 순열에 걸쳐 거의 균등 분포:

```
기대값 (균등): 335.7개
실제 min/max:  305 / 386
std:            16.9
```

특정 순열 편향 없음 — 균등 설계. (참고: Input_i→정답위치 배정에는 경미한 편향이 있음, 12-1절 참고)

---

## 4. 정답 순열의 복잡도 분포 [train 내부]

### 켄달타우 거리 (정방향 [1,2,3,4] 대비 필요한 swap 횟수)

```python
gt_kendall
1    1054  (13.1%)   ← 1번 swap으로 만들 수 있는 순열
2    1716  (21.3%)
3    2082  (25.8%)   ← 가장 많음
4    1766  (21.9%)
5    1073  (13.3%)
6     366   (4.5%)   ← 완전 역순 [4,3,2,1]
```

중간 복잡도(τ=3)가 가장 많고, 쉬운 케이스(τ=1,2)와 어려운 케이스(τ=4,5) 고루 분포.
τ=1~6 각각의 순열 개수 자체는 **4개 원소 순열의 조합론적 상수**(1,3,5,6,5,3,1 형태, τ=0 정답 제외)이며,
우리가 임의로 정한 게 아니라 4프레임 문제 구조 자체에서 나오는 값이다.

### 인접 위치 역전 수

```
adj_inv=1    3820  (47.4%)  ← 인접 쌍 1개만 역전
adj_inv=2    3871  (48.0%)  ← 인접 쌍 2개 역전
adj_inv=3     366   (4.5%)  ← 3개 전부 역전 (= [4,3,2,1])
```

→ 데이터 대부분이 인접 역전 1~2개 케이스. hard negative 설계(인접 스왑 중심)가 데이터 분포와 일치.

---

## 5. 문장 길이 분포 — train 내부 [①]

```python
# train (No_ordering=False)
mean=24.2  median=26  std=9.5   min=5  max=69
```

### 구간별 비율

| 구간 | train 비율 |
|---|---|
| ≤15단어 | 25.2% |
| 15-20단어 | 9.3% |
| 20-25단어 | 12.1% |
| 25-30단어 | 23.5% |
| 30-35단어 | 19.9% |
| 35단어 이상 | 10.1% |

8절에서 보듯 "train/val 내부에서 문장이 길수록 모델이 더 잘 맞힌다"는 **train만으로 독립적으로 나오는
발견**을 설계 근거로 삼는다(평가 데이터 특성은 규칙 3.4에 따라 어떤 형태로도 참고하지 않음).

---

## 6. (제거됨 — 평가 데이터 특성 분석 내용이라 규칙 3.4 준수를 위해 본 문서에서 삭제)

---

## 7. Temporal/Discourse Marker 분석 [train 내부] — 초판 → 정밀화(v3) 전체 과정

### 7-1. 초판 (2026-07-02, 크루드 substring 매칭)

10개 단어(`then/before/after/while/as/next/finally/first/second/third`)를 단순 부분 문자열
카운트(`s.lower().count(w)`)로 셈:

```
train (No_ord=F) temporal_words 평균: 1.68개/문장
```

→ 이 초판 방식은 "as"의 비시간적 용법("as a model")이나 "after"가 "afternoon"에 우연히 포함되는 등의
**오탐 가능성**이 있어, v3에서 spaCy 기반으로 재검증했다.

### 7-2. 정밀화 (v3, 2026-07-05) — spaCy lemma + POS 기반

**방법**: spaCy(`en_core_web_sm`)로 전체 문장을 토큰화하고, lemma 기준으로 5개 언어학적 카테고리
(35개 단어)를 재정의:

| 카테고리 | 단어 |
|---|---|
| SEQUENCE | first, second, third, fourth, fifth, next, then, subsequently, afterward(s), finally, lastly, eventually, initially, later |
| SIMULTANEITY | while, meanwhile, during, as, simultaneously, throughout, when |
| PRIOR | before, previously, earlier, prior, beforehand |
| POSTERIOR | after, following, thereafter |
| CAUSAL | therefore, thus, consequently, hence, so |

기존 10개 대비 **+25개 단어 추가**(`therefore, subsequently, meanwhile, previously, thereafter` 등
"놓쳤을 수도 있다"고 의심됐던 단어들 전부 포함).

**놀라운 결과 — 정밀화하니 오히려 카운트가 줄었다**:

```
신규 방식(35단어, lemma+정밀매칭) 평균: 1.472개/문장
구 방식(10단어, substring)        평균: 1.682개/문장
차이: -0.210  (신규가 더 적음)
```

단어를 25개나 더 추가했는데도 평균이 줄어든 이유는, **구 방식의 substring 매칭이 만든 오탐 제거 효과가
새 단어 추가분보다 컸기 때문**이다. 구체적 예: "as"는 lemma 전체 카운트로 3638회 잡히지만, 품사를
ADV/SCONJ(시간/양보의 부사·접속사 역할)로 한정하면 2183회로 줄어든다 — **차이 1455회는 "as a model"류의
비시간적 용법**이었다는 뜻이다.

**단어별 실제 등장 빈도** (train, No_ordering=False, 8057개 기준):

| 단어 | 카테고리 | 빈도 |
|---|---|---|
| then | SEQUENCE | 4911 |
| as | SIMULTANEITY | 3638 |
| while | SIMULTANEITY | 1281 |
| finally | SEQUENCE | 792 |
| before | PRIOR | 592 |
| after | POSTERIOR | 167 |
| next | SEQUENCE | 97 |
| first | SEQUENCE | 67 |
| when | SIMULTANEITY | 43 |
| second | SEQUENCE | 34 |
| meanwhile | SIMULTANEITY | 33 |
| throughout | SIMULTANEITY | 29 |
| third | SEQUENCE | 26 |
| eventually | SEQUENCE | 24 |
| subsequently / initially | SEQUENCE | 16 / 16 |
| during / later | SIMULTANEITY / SEQUENCE | 15 / 15 |
| afterwards / afterward | SEQUENCE | 14 / 13 |
| so | CAUSAL | 13 |
| fourth | SEQUENCE | 10 |
| simultaneously | SIMULTANEITY | 7 |
| following / previously / fifth | POSTERIOR / PRIOR / SEQUENCE | 2 / 2 / 2 |
| lastly | SEQUENCE | 1 |

### 7-3. 데이터 기반 발견 — "놓친 단어가 있는가?" (POS=ADV/SCONJ 최빈 40개 전수 확인)

혹시 사전 정의 리스트 밖에 중요한 temporal 단어가 있는지, ADV/SCONJ 품사 토큰 전체를 빈도순으로
확인했다. **결론: 놓친 temporal 단어는 없었다.** 리스트 밖 최빈 단어들은 전부 **공간/방식 부사**였다:

```
slightly(947) away(439) right(406) forward(390) close(340) upward(317) back(299)
out(276) where(161) downward(160) together(152) again(108) down(91) more(82)
far(81) around(79) nearby(52) smoothly(51) still(51) closely(51) partially(48)
upright(47) gradually(44) up(39) behind(37) energetically(37) gracefully(35)
overhead(35) outside(33) ...
```

이들은 "어떻게/어디로" 움직이는지(동작 묘사)를 나타내지 "언제"를 나타내지 않는다 — 이 데이터셋의
문장은 "카메라가 부드럽게(smoothly) 줌인한다"류의 **동작 묘사 부사가 매우 풍부**하다는 별도 특징이
드러났다(8절 bigram 분석과 연결됨).

### 7-4. gt_kendall(정답 복잡도)과의 상관관계 — 신규/구 방식 모두 무의미

```
신규 방식 전체: r = -0.0195
구 방식 전체:   r = -0.0201
```

카테고리별로 쪼개도 전부 |r| < 0.021 수준. **temporal word 밀도는 "정답이 얼마나 뒤섞여 있는지"와는
전혀 무관하다** — 이건 8절에서 다루는 "모델이 실제로 맞히는지"와는 다른 질문이라는 점에 유의.

---

## 8. 전체 언어학적 구조 분석 [train 내부, NEW v4] — POS·시제·의존구문·어휘다양성

spaCy로 train(No_ordering=False, 8057개) 전체를 품사 태깅 + 의존구문 파싱 + 형태소 분석.

### 8-1. 전체 품사(POS) 분포

| 품사 | 비율 |
|---|---|
| NOUN | 29.1% |
| DET | 18.7% |
| VERB | 16.3% |
| ADP(전치사) | 13.9% |
| ADV | 6.4% |
| ADJ | 4.0% |
| PRON | 3.3% |
| CCONJ(등위접속사) | 2.8% |
| SCONJ(종속접속사) | 2.0% |
| PART | 1.5% |
| AUX | 1.1% |
| PROPN(고유명사) | 0.6% |
| NUM | 0.4% |

고유명사가 0.6%뿐 — 이 데이터셋 문장은 "the woman", "a man" 같은 **일반 명사구로 사람을 지칭**하고
고유명사(이름 등)는 거의 안 쓴다는 뜻.

### 8-2. 문장당 평균 통계 (spaCy 정밀 측정)

| 지표 | 평균 | 표준편차 | 범위 |
|---|---|---|---|
| 토큰 수(spaCy) | 24.26 | 9.62 | 5~70 |
| 동사 수 (사건 수 프록시) | 4.22 | 1.98 | 0~15 |
| 명사 수 | 7.20 | 3.52 | 0~20 |
| 형용사 수 | 0.96 | 1.08 | 0~8 |
| 부사 수 | 1.54 | 1.33 | 0~9 |
| 등위접속사 수 | 0.67 | 0.71 | 0~7 |
| 종속접속사 수 | 0.49 | 0.67 | 0~3 |
| 의존구문 최대 깊이 | 5.79 | 1.92 | 2~19 |
| 어휘다양성(TTR) | 0.840 | 0.082 | 0.54~1.0 |
| **수동태 문장 비율** | **15.8%** | — | — |

토큰 수(24.26)가 기존 `.split()` 기반 단어 수(24.2)와 거의 동일 — 기존 방식이 이미 충분히 정확한
근사였음을 확인(단순 공백 분리로도 큰 오차 없었다는 뜻).

### 8-3. 시제(Tense) 분포 — 압도적 현재형

```
동사 단위: Pres 84.5%  /  Past 15.5%
문장 단위(지배적 시제): Pres 94.7%  /  Past 4.5%  /  판정불가 0.9%
```

**train 문장의 94.7%가 현재시제로 서술된다** — 스포츠 중계/영상 캡션 특유의 "지금 일어나는 일을
서술하는" 문체(예: "The camera zooms in...", "A man throws the ball...")가 압도적. 이는 대회
데이터셋의 캡션 생성 방식(아마도 프레임을 보고 실시간으로 서술하는 방식)을 시사하는 특징.

### 8-4. 최빈 bigram Top 15 — 촬영 기법 묘사 문체 확인

```
the camera(2987) as the(2228) on the(1533) to a(1310) to the(1212)
then the(1106) with a(1102) camera zooms(1036) from the(983) on a(844)
the person(823) followed by(821) a person(790) in the(727) then a(712)
```

"camera zooms in/out", "shifts to", "begins to" 등 **카메라 움직임을 명시적으로 서술하는 문체**가
매우 흔함 — 7-3절에서 발견한 "동작 부사가 풍부하다"는 특징과 일치.

### 8-5. gt_kendall과의 상관관계 — 전 축 총정리 (전부 무의미)

| 특징 | r |
|---|---|
| TTR(어휘다양성) | +0.0274 |
| 토큰 수 | -0.0247 |
| 명사 수 | -0.0224 |
| 부사 수 | -0.0163 |
| 의존구문 깊이 | -0.0151 |
| 종속접속사 수 | -0.0151 |
| 동사 수 | -0.0091 |
| 형용사 수 | -0.0037 |
| 등위접속사 수 | -0.0008 |
| 수동태 여부 | +0.0073 |

**언어학적으로 측정 가능한 어떤 축을 봐도 gt_kendall(정답의 객관적 뒤섞임 정도)과 상관이 없다.**
n_verbs별 gt_kendall 평균도 0~10개 전 구간에서 3.0~3.25 사이로 평평함. dominant_tense별로도
Pres 3.15 / Past 3.16으로 차이 없음. → **이 데이터셋은 "문장을 보면 정답이 얼마나 복잡할지 티가 나는"
편향이 없이 잘 설계되었다**는 뜻이기도 하다 (긍정적 해석: 텍스트로 난이도를 치팅할 여지가 없는 공정한
벤치마크).

---

## 9. 최대치 NLP 분석 [train 전체 9,535개, NEW v6] — NER·명사구·가독성·구두점·형태소·의존관계·어휘빈도

8절(POS/시제/의존구문/TTR)에서 다루지 않은 축까지 spaCy 전체 파이프라인(NER 포함)으로 총동원 분석.
test는 전혀 사용 안 함.

### 9-1. 개체명 인식(NER)

```
개체명이 하나라도 있는 문장: 1486/9535 (15.6%)

타입별 빈도: CARDINAL 853, ORG 227, ORDINAL 157, PERSON 144, WORK_OF_ART 101,
             DATE 85, PRODUCT 74, GPE 52, NORP 43, FAC 23, TIME 20, ...
```

최빈 개체명은 CARDINAL(수사) "two"(471), "one"(245) — 사람/물체 개수를 세는 표현이지 진짜 고유명사가
아님. ORDINAL로 분류된 "first"(72), "second"(41), "third"(30)는 사실 7절의 temporal marker와 같은
단어인데 spaCy가 개체명으로도 이중 분류한 것. 진짜 PERSON은 144회뿐(예: "joyce" 9회) — 8-1절에서
확인한 "고유명사 0.6%"와 일관되게, **이 데이터셋은 특정 인물/장소를 지칭하지 않는 일반적 행동 묘사가
압도적**임을 재확인.

### 9-2. 명사구(Noun Chunk) 분석

```
문장당 평균 명사구 개수: 6.28개
명사구 평균 길이: 2.08 단어
```

최빈 명사구: "it"(1408), "the camera"(1269), "the camera zooms"(1037, 청크 경계 특이케이스),
"a person"(842), "the person"(810), "the man"(433), "the woman"(415), "a close-up"(256),
"a hand"(230) — 8-4절 bigram 분석과 일관되게 **카메라·일반인물·신체부위 중심 어휘**.

### 9-3. 가독성 지표 (Flesch, 외부 사전 없이 모음그룹 기반 음절 카운트)

```
Flesch Reading Ease 평균: 64.3   (0~100, 높을수록 쉬움 — 64.3은 "약간 쉬움~보통" 구간)
Flesch-Kincaid Grade Level 평균: 10.2학년 수준
단어당 평균 음절수: 1.398  (대부분 1~2음절의 단순 단어)
```

### 9-4. 불용어 비율 / 어휘밀도

```
불용어 비율 평균: 0.493   (거의 절반이 the/a/to 등 기능어)
어휘밀도(내용어 비율) 평균: 0.557
```

### 9-5. 구두점 패턴

| 구두점 | 문장당 평균 |
|---|---|
| comma | 1.407 |
| semicolon | 0.058 |
| period(문장 내부) | 0.023 |
| exclaim | 0.004 |
| question | 0.000 |
| dash | 0.088 |

콤마로 절을 이어붙이는 게 압도적이고, 세미콜론/느낌표/물음표는 거의 안 씀 — **단순하고 정형화된
문장 구조**(설명문체, 감탄/의문 없음).

### 9-6. 형태소 자질 (Number, VerbForm)

```
Number:   Sing 77633 (83.6%)  /  Plur 15282 (16.4%)
VerbForm: Fin(정동사) 20204  /  Part(분사) 15498  /  Inf(부정사) 4156  /  Ger(동명사) 286
```

분사(Part)가 정동사(Fin)의 77% 수준으로 매우 많음 — "the camera **zooming** in", "a man **raising**
his hand"처럼 **진행형/분사구문으로 동시 동작을 묘사하는 문체**가 흔하다는 뜻.

### 9-7. 의존관계(주어/목적어 등) 카운트 — 의미역 프록시

| 의존관계 | 문장당 평균 |
|---|---|
| pobj(전치사목적어) | 2.752 |
| dobj(직접목적어) | 1.640 |
| nsubj(주어) | 1.614 |
| conj(등위연결) | 0.834 |
| nsubjpass(수동주어) | 0.169 |
| attr(서술보어) | 0.007 |

pobj가 가장 많다는 건 **전치사구("on the X", "to the Y", "from the Z")로 공간/대상을 세밀하게
묘사**하는 문체임을 시사 — 8-4절의 "the camera", "to a", "on the" bigram과 일관됨.

### 9-8. 동사-동사 연쇄(사건 연쇄) 카운트

```
문장당 평균 동사연쇄(conj, VERB-VERB) 수: 0.584
연쇄 0개인 문장: 54.8%  /  1개: 40.6%  /  2개 이상: 4.6%
```

절반 이상의 문장이 명시적 "A하고 B한다"는 **동사 등위연결 없이** 서술됨 — "then" 같은 부사로만
사건을 잇거나, 애초에 단일 사건만 묘사하는 문장이 많다는 뜻.

### 9-9. Trigram 분석 (Top 5)

```
"as the camera"(1773) "the camera zooms"(1239) "camera zooms in"(702)
"followed by a"(502) "camera zooms out"(497)
```

### 9-10. 문장 시작 토큰 — 압도적으로 관사

```
the(5687) a(1880) he(391) she(277) then(129) they(124) two(100) after(73) ...
```

문장의 80% 이상이 정관사/부정관사로 시작 — 즉 **"먼저 등장인물/사물을 소개하고 나서 행동을 서술"**
하는 전형적 서술 구조. temporal marker로 문장을 시작하는 경우(then, after 등)는 소수.

### 9-11. Corpus 내부 단어 빈도 (Zipf 분석, train 자체 통계 — 외부 데이터 아님)

```
고유 내용어(lemma) 수: 4,046개
최빈 내용어: camera(3774) move(2368) person(2133) zoom(1796) man(1739) shift(1575)
             hand(1480) right(1312) begin(1282) transition(1134) slightly(1105)
             follow(1069) woman(1050) finally(928) reveal(867) leave(858)

Zipf 법칙 확인: 최빈 10% 단어가 전체 등장의 75.3% 차지 (전형적 power-law 분포)
```

어휘가 "camera/zoom/shift/pan/tilt/transition"(촬영 기법) + "person/man/woman/hand"(인물/신체)
두 축에 극도로 집중되어 있음 — 도메인이 일반 웹 영상보다는 **정형화된 촬영 문법을 따르는 콘텐츠**일
가능성.

### 9-12. gt_kendall과의 상관관계 — 역시 전부 무의미

| 특징 | r |
|---|---|
| 명사구 개수 | -0.0246 |
| dobj 개수 | -0.0219 |
| nsubj 개수 | -0.0182 |
| 동사연쇄 개수 | +0.0154 |
| Flesch 가독성 | +0.0116 |
| 불용어 비율 | -0.0083 |
| 개체명 개수 | +0.0045 |
| 어휘밀도 | +0.0030 |
| 음절수/단어 | +0.0000 |

8절 결과와 합쳐 **이제 25개 이상의 언어학적 축 전부에서 gt_kendall과 무상관**임이 확인됨 —
"문장을 보고 정답 복잡도를 예측할 수 있는 텍스트 신호는 없다"는 결론이 매우 견고해짐.

---

## 10. ⭐ 실제 모델 정확도와의 상관관계 [train/val 내부 — 가장 중요한 신규 발견]

**8절까지는 전부 "언어적 특징 vs 정답의 객관적 복잡도(gt_kendall)"였다. 그런데 이건 우리가 진짜
궁금한 질문("모델이 실제로 맞히는가")과는 다른 질문이다.** 이번에 처음으로 실제 학습된 체크포인트
(v6.5 `best_v6_5`)를 val(train의 5% 분할, 476개 중 No_ordering=False 399개)에 돌려서 **샘플별
정답/오답 여부**를 얻고, 언어적 특징과 직접 상관관계를 계산했다. **test 데이터는 전혀 사용하지 않음.**

### 방법론

v1의 `diagnose_val.py`(문장 길이 vs 정확도, r=0.376)와 동일한 방법론을 v6.5 체크포인트로 재현 +
temporal word 축을 추가 (`v6.5_resample/val_temporal_diagnosis.py`, job 226492).

### 결과

```
Val Exact-Match Accuracy (best_v6_5, n=399): 0.5689 (227/399)

상관계수 (sent_len vs correct)      = 0.435   (v1 기준 0.376보다 더 강해짐)
상관계수 (temporal_words vs correct) = 0.221   (신규 검증 — 유의미한 양의 상관)
```

**sent_len 구간별 정확도 — 매우 뚜렷한 단조 증가**:

| 구간 | 정확도 | n |
|---|---|---|
| ≤15단어 | 28.6% | 98 |
| 15-20단어 | 30.8% | 39 |
| 20-25단어 | 46.3% | 41 |
| 25-30단어 | 67.6% | 102 |
| 30-35단어 | 80.0% | 80 |
| **35단어 이상** | **89.7%** | 39 |

**temporal_words 구간별 정확도 — 대체로 증가 후 꼬리 노이즈**:

| 구간(개수) | 정확도 | n |
|---|---|---|
| 0개 | 35.0% | 80 |
| 1개 | 47.0% | 83 |
| 2개 | 69.2% | 130 |
| 3개 | 68.2% | 85 |
| 4개 | 64.7% | 17 |
| 5개 이상 | 25.0% | 4 (표본 극소, 노이즈로 판단) |

### 10-1. 다변량 확장 — sent_len/temporal_words 말고 다른 축들도 확인 (v5)

sent_len/temporal_words 두 개만 보고 끝내지 않고, **8,9절의 NLP 축 전부 + 이미지 유사도까지 같은
399개 val 샘플에 대해 전부 계산해서 종합 랭킹**을 매겼다 (`deep_eda_v5_full_correlation.py`).

**정확도(correct)와의 상관관계 종합 랭킹**:

| 특징 | r | 강도 |
|---|---|---|
| n_nouns (명사 수) | **+0.467** | ★★★ 최강 |
| sent_len (문장 길이) | +0.435 | ★★★ 강함 |
| max_dep_depth (의존구문 깊이) | +0.301 | ★★★ 강함 |
| n_verbs (동사 수) | +0.278 | ★★ 중간 |
| temporal_words | +0.221 | ★★ 중간 |
| ttr (어휘다양성) | **-0.211** | ★★ 중간 (음의 상관!) |
| n_adv (부사 수) | +0.200 | ★★ 중간 |
| max_adj_rmse (인접프레임 최대 시각차) | +0.171 | ★★ 중간 |
| n_sconj (종속접속사 수) | +0.171 | ★★ 중간 |
| mean_adj_rmse (인접프레임 평균 시각차) | +0.155 | ★★ 중간 |
| is_passive (수동태 여부) | +0.115 | ★ 약함 |
| n_adj (형용사 수) | +0.088 | ★ 약함 |
| min_adj_rmse (인접프레임 최소 시각차) | +0.055 | ★ 약함 |
| n_cconj (등위접속사 수) | -0.002 | 무의미 |

**놀랍게도 1위는 sent_len이 아니라 n_nouns(명사 수, r=0.467)** — 문장이 길어서가 아니라 **구체적인
명사(대상/객체)가 많이 언급될수록** 모델이 더 잘 맞힌다는 뜻. 다만 n_nouns, sent_len, max_dep_depth,
n_verbs는 서로 강하게 얽혀있을 가능성이 높다(전부 "문장에 담긴 정보량"의 다른 측정치) — 독립적인
개별 신호라기보다 **"정보 밀도"라는 하나의 근본 축이 여러 지표로 드러난 것**으로 해석하는 게 안전하다.

**ttr(어휘다양성)이 음의 상관(-0.211)인 이유**: 짧은 문장은 반복 단어가 적어 TTR이 자연히 높고,
긴 문장은 "the/camera" 등 기능어 반복으로 TTR이 낮아진다 — 즉 **TTR은 독립 신호가 아니라 sent_len의
역방향 프록시**일 가능성이 크다.

**이미지 유사도 특징의 흥미로운 비대칭**: `mean_adj_rmse`/`max_adj_rmse`(평균/최대 시각차)는 약한
양의 상관(+0.15~0.17)이 있지만, **`min_adj_rmse`(가장 유사한 인접쌍의 유사도)는 거의 무관(+0.055)**.
11절에서 "gap=1이 gap=3보다 평균적으로 더 비슷하다"는 **집단(aggregate) 수준** 효과(Cohen's d=0.349)를
확인했었는데, **샘플 개별 단위**로는 "가장 헷갈리는 한 쌍이 얼마나 비슷한지"가 정확도를 잘 예측하지
못한다 — 오히려 "전체적으로 프레임 변화가 큰 영상"(mean/max_adj_rmse 높음)일수록 전반적으로 쉽다는
뜻에 가깝다. `min_adj_rmse` 5분위 구간별 정확도도 54.3%/53.2%/59.5%/53.8%/**63.8%**로 최상위
구간(가장 다름)만 살짝 높고 나머지는 거의 평평함 — 뚜렷한 단조 관계는 아니다.

**수동태 문장이 유리함**: 수동태(15.8%, n=66) 정확도 69.7% vs 능동태 54.4% — 15.3%p 차이지만 표본이
작아 추가 검증 필요.

### 10-2. ⭐⭐⭐ 최종 확정 — 다중공선성 제거 후 진짜 원인은 `n_nouns` 하나뿐

10-1절 표의 상위 특징들(n_nouns, sent_len, max_dep_depth, n_verbs, temporal_words)이 서로 r=0.36~0.89로
심하게 얽혀있어(특히 sent_len↔n_nouns r=**0.892**), "독립적인 6개 신호"라는 해석이 맞는지 6가지 방법으로
직접 검증했다.

**검증 1 — 로지스틱 회귀(n_nouns + sent_len 동시 투입)**:
```
n_nouns:  coef=0.290  p=0.0002   ← 유의함, 살아남음
sent_len: coef=0.015  p=0.580   ← sent_len을 같이 넣으면 완전히 유의성 상실
```

**검증 2 — 5개 특징 전부 동시 투입**: n_nouns만 p=0.0001로 유의. sent_len(p=0.64), max_dep_depth(p=0.72),
n_verbs(p=0.097), temporal_words(p=0.084, 계수 **음수**로 반전)는 전부 유의하지 않음.

**검증 3 — Bootstrap 95% CI (2000회 리샘플링)**:
```
n_nouns  partial r (sent_len 통제):  0.193,  CI=[0.092, 0.291]  → 0 미포함, 유의함
sent_len partial r (n_nouns 통제):   0.048,  CI=[-0.051, 0.151] → 0 포함, 유의하지 않음
```

**검증 4 — Spearman 순위상관(이상치/비선형성에 강건)**: rho=0.469, p=3.27e-23 — Pearson과 거의 동일,
이상치로 인한 착시가 아님.

**검증 5 — val을 반으로 쪼갠 내부 재현성**:
```
전반부(n=199): n_nouns r=0.382  sent_len r=0.380  (거의 동률)
후반부(n=200): n_nouns r=0.551  sent_len r=0.490  (n_nouns가 더 높음)
```
양쪽 독립 표본에서 일관되게 n_nouns ≥ sent_len — 우연이 아님.

**검증 6 — n_nouns 구간별 정확도 (가장 깔끔한 단조 증가)**:

| 명사 개수 | 정확도 | n |
|---|---|---|
| 1-4개 | 29.4% | 119 |
| 5-6개 | 35.6% | 59 |
| 7-8개 | 64.1% | 64 |
| 9-10개 | 83.3% | 78 |
| 11-15개 | 82.3% | 79 |

**최종 결론**: sent_len, max_dep_depth, n_verbs, temporal_words가 정확도와 상관관계를 보인 건 전부
**n_nouns의 부산물**이었다 (문장이 길면 명사도 자연히 많아지고, temporal word도 같이 늘어나는 식의
공기(co-occurrence) 때문). **진짜 독립적인 원인은 "문장에 언급된 구체적 명사(대상/객체) 개수" 하나뿐이다.**
해석: 명사가 많다 = 이미지와 대조할 구체적 시각 앵커(대상)가 많다 = 텍스트-이미지 정렬이 쉬워진다.
단순히 "문장이 길다"나 "시간 접속사가 많다"는 부수 현상이지 원인이 아니었다.

**⚠️ 설계 시사점**: 향후 loss 재가중치나 프롬프트 설계는 sent_len이나 temporal_words가 아니라
**`n_nouns`를 유일한 근거 축으로 삼아야 한다.** (v7 설계에 직접 반영, 아래 §14 참고)

### 해석 — 8,9절과의 대비가 핵심

- **8,9절**: 언어적 특징은 "정답이 얼마나 복잡한가(gt_kendall)"와 무관 (25개+ 축 전부 r≈0) —
  데이터셋 설계가 공정함.
- **10절(이 절)**: 그런데 언어적 특징(특히 n_nouns, sent_len, max_dep_depth)은 "**모델이 실제로
  맞히는가**"와는 **뚜렷하게 유의미한 상관**이 있다 (최대 r=0.467).

이 둘을 합치면 결론은 명확하다: **문장이 짧고 temporal word가 적을 때 모델이 틀리는 이유는 "문제
자체가 객관적으로 더 어려워서"가 아니라, "텍스트에 순서를 유추할 명시적 단서가 부족해서 이미지에만
의존해야 하기 때문"**이다. 이는 5-1절(d=1 인접 스왑 문제)의 근본 원인과 같은 계열 — 짧은 문장은
이미지만으로 판단해야 하는데, 인접 프레임은 시각적으로 유사(11절, Cohen's d)해서 이중으로 불리하다.

### 규칙 준수 확인

이 절의 모든 수치는 **train.csv를 SEED=42로 분할한 val(5%) 내부**에서만 나온 것이며, `best_v6_5`
체크포인트도 이 train/val만으로 학습된 것이다. test.csv는 이 분석 어디에도 사용되지 않았다 —
따라서 이 발견을 근거로 학습 설계(예: AdaptiveDistanceLoss를 length/temporal-word 축으로 확장)를
바꾸는 것은 규칙 3.4에 저촉되지 않는, 완전히 합법적인 데이터 기반 개선이다.

---

## 11. 이미지 시각적 유사도 vs 시간적 간격 [train 내부] (2026-07-04)

### 배경

v1~v4 전 버전에서 공통 관찰: d=1(인접 스왑) 오답이 hard negative 학습·손실 재가중치를 아무리
강하게 걸어도 잘 안 줄어듦. "손실함수 문제가 아니라 애초에 시각적으로 구분이 어려운 게 아닐까"라는
가설을 직접 검증.

### 방법

train(No_ordering=False)에서 1500개 샘플 추출 → 각 샘플의 4장을 정답 순서대로 정렬 → 모든 쌍(6쌍/샘플)
에 대해 시간적 간격(gap=1,2,3)별로 두 시각적 거리 측정: **pixel RMSE**(48×48 흑백 축소본),
**hist_dist**(64×64 RGB 히스토그램 교집합 거리).

### 결과

| gap(시간 간격) | n_pairs | pixel RMSE 평균 | hist_dist 평균 |
|---|---|---|---|
| 1 (바로 다음 사건) | 4500 | **0.2607** | **0.3182** |
| 2 | 3000 | 0.2895 | 0.3831 |
| 3 (첫 vs 마지막) | 1500 | 0.2981 | 0.3977 |

gap=1 vs gap=3: pixel RMSE 비율 1.143x, hist_dist 비율 1.250x. **Cohen's d = 0.349** (small~medium).

**해상도 의존성 검증**: 32px~448px 전 구간에서 Cohen's d = 0.303~0.306으로 거의 동일 →
**이 시각적 유사성 신호는 해상도와 무관** (미세 디테일이 아니라 구도/색감 수준의 "큰 그림" 유사성).

### 해석

가설이 확인됨 — 단, "완전한 벽"은 아니고 "중간 정도의 불리한 조건". 시간상 인접한 두 프레임이
통계적으로 유의미하게 더 비슷하게 생겼다. d=1 hard negative는 손실함수 설계와 무관하게 원천적으로
시각적 신호가 더 약한 조건에서 시작한다. 효과크기가 압도적이진 않아(d=0.349) 완전히 못 푸는 벽은
아니지만, **해상도 상향으로는 해결이 안 됨**(512px 시도 → OOM 리스크만 있고 실익 없음, 6절 v6 기록 참고).

---

## 12. 기타 [train 내부]

### 12-1. Input_i(파일명 알파벳 순) → 정답 위치 배정 편향 (경미하지만 존재)

| Input_i \ 정답위치 | pos1 | pos2 | pos3 | pos4 |
|---|---|---|---|---|
| Input_1 | **21.8%** | 25.7% | 26.5% | 26.1% |
| Input_2 | 25.5% | **22.0%** | 26.1% | 26.3% |
| Input_3 | 26.3% | 26.0% | **21.5%** | 26.2% |
| Input_4 | 26.4% | 26.4% | 25.9% | **21.4%** |

대각선("제자리에 그대로 있는" 경우)이 다른 칸보다 일관되게 ~4~5%p 낮음. 완전 균등(25%)은 아니지만
효과가 작고, 어차피 24-permutation 전수조사를 쓰고 있어 영향 없음 — 참고용 기록.

### 12-2. 근접중복 이미지 탐지 — 문제 없음

300개 샘플 pairwise(44,850쌍) average-hash 비교 결과 near-duplicate(hamming≤2) **16쌍(0.036%)**만
발견 — train 내 동일/거의동일 소스 영상 재사용은 미미한 수준. val split(랜덤 5%) 신뢰성에 문제 없음.

---

## 13. Type A/B 프레임워크에 대한 재평가 [과거 표현 수정]

기존(2026-07-02) 문서는 평가 데이터 특성 분석에서 나온 프레임워크로 접근법을 평가했었다. 이는 학습
설계에 직접 반영하면 규칙 3.4 위반 소지가 있어 폐기했다. 10절에서 train/val 내부만으로 독립적으로
재현한 결과, "텍스트에 명시적 시간 단서가 많을수록 모델이 더 잘 맞힌다"는 유사한 패턴이 확인되므로,
실질적으로 참고할 수 있는 것은 10절의 train/val 내부 결과뿐이다.

---

## 14. 종합 결론 및 다음 설계 방향 시사점

1. **⭐ 확정된 핵심 신호(10-2절)**: `n_nouns`(문장에 언급된 명사 개수)가 **유일하게 독립적인** 정확도
   예측 신호다. sent_len(r=0.435), max_dep_depth(r=0.301), n_verbs(r=0.278), temporal_words(r=0.221)는
   전부 n_nouns와의 다중공선성(r=0.36~0.89) 때문에 생긴 **부산물**이었음이 로지스틱 회귀·bootstrap
   95% CI·split-half 재현성 등 6가지 방법으로 확정됨 (n_nouns 통제 시 나머지는 전부 p>0.08로 유의성
   상실, 심지어 temporal_words는 계수가 음수로 반전). n_nouns 구간별 정확도는 29.4%→83.3%로 거의
   3배 차이 나는 가장 깔끔하고 강력한 신호. **v7 설계는 n_nouns 하나만을 근거 축으로 삼는다.**
2. **d=1 문제는 구조적으로 일부 잔존**(11절, Cohen's d=0.349, 해상도 무관) — 별도 lever 필요
   (예: 프레임 diff 이미지 등, 진행 중). 다만 개별 샘플 단위 `min_adj_rmse`는 정확도와 거의 무관
   (r=0.055) — Cohen's d는 gap별 **집단** 평균 비교에서만 유의미하고, 샘플 단위 예측력은 약함.
3. **언어적 복잡도는 정답의 객관적 난이도(gt_kendall)와는 완전히 무관**(8,9절, 25개 이상 축 전부
   |r|<0.05) — 데이터셋이 "텍스트로 난이도를 치팅"할 수 없게 공정하게 설계됨. 이는 긍정적이지만,
   동시에 "짧은 문장=이미지에만 의존해야 함=더 어려움"이라는 10절의 발견과 모순되지 않는다
   (별개의 두 축: 정답의 객관적 복잡도 vs 모델이 풀 때 쓸 수 있는 단서의 양).
4. **평가 데이터 특성은 어떤 형태로도 설계에 반영하지 않는다** — 대신 train/val 내부에서 독립적으로
   재현되는 패턴(10절)만을 근거로 삼을 것.
5. 데이터셋 문체 특징(8,9절): 94.7% 현재시제, 카메라 움직임 묘사 풍부(camera/zoom/shift/pan/tilt가
   최빈 내용어), 고유명사 거의 없음(0.6%), 개체명도 대부분 숫자(two/one)나 오분류된 서수(first 등),
   불용어 비율 49.3%, 가독성 Flesch 64.3(중간 난이도) — 프롬프트 설계 시 참고 가능한 배경지식.
6. **수동태 문장(15.8%)이 능동태보다 정확도가 높음**(69.7% vs 54.4%, 10절) — 표본이 작아(n=66)
   단정하긴 이르지만, 흥미로운 후속 검증 후보.

---

## 15. 재현 코드 경로

| 스크립트 | 내용 |
|---|---|
| `deep_eda.py` | 초판: gt_kendall vs 문장길이/action word/comma 상관관계 (4,5,6,7절 관련) |
| `deep_eda_v2.py` | 이미지 시각적 유사도(Cohen's d), 해상도 의존성 (11절) |
| `deep_eda_v3_nlp.py` | spaCy 기반 temporal marker 정밀 재검증 + 데이터 기반 발견 (7절) |
| `deep_eda_v4_full_nlp.py` | POS/시제/의존구문/TTR/bigram 전체 분석 (8절) |
| `deep_eda_v5_full_correlation.py` | **val 다변량 특징(NLP+이미지) vs 실제 정확도 종합 상관관계** (10절) |
| `deep_eda_v6_maximal_nlp.py` | NER/명사구/가독성/구두점/형태소/의존관계/trigram/어휘빈도 최대치 분석 (9절) |
| `v6.5_resample/val_temporal_diagnosis.py` | 체크포인트 val 정확도 vs sent_len/temporal_words (10절, job 226492) |

### 기본 통계 재현 코드 (train/test 로드 + 기본 파생 변수)

```python
import pandas as pd
import numpy as np
import ast
from pathlib import Path

DATA_DIR = Path('/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
train['answer_parsed'] = train['Answer'].apply(ast.literal_eval)
real = train[train['No_ordering'] == False].copy()

def kendall_dist(perm):
    dist = 0
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            if perm[i] > perm[j]:
                dist += 1
    return dist
real['gt_kendall'] = real['answer_parsed'].apply(kendall_dist)

def count_adj_inv(perm):
    return sum(1 for i in range(len(perm)-1) if perm[i] > perm[i+1])
real['adj_inv'] = real['answer_parsed'].apply(count_adj_inv)
```

spaCy 기반 정밀 분석(7,8절)의 전체 코드는 `deep_eda_v3_nlp.py`, `deep_eda_v4_full_nlp.py` 파일
원본을 참고할 것 (카테고리 정의, 의존구문 깊이 계산, 시제 추출 로직 등 상세 포함).
