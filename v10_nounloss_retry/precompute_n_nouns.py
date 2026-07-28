"""
train.csv 전체(9,535개)에 대해 spaCy로 n_nouns(명사 개수)를 미리 계산해서 캐싱.
EDA.md §10-2에서 유일하게 검증된 독립 신호. 매 epoch/샘플 접근마다 spaCy를 다시 돌리면
낭비이므로 학습 시작 전 1회만 계산해서 Id -> n_nouns 룩업 테이블로 저장.
"""
from pathlib import Path
from collections import Counter

import pandas as pd
import spacy

DATA_DIR = Path("/data/gyuyeonlim/snu_ai_challenge/data/snuaichallenge_data")
OUT_PATH = Path(__file__).parent / "checkpoints" / "_n_nouns_lookup.csv"

print("spaCy 로딩...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
# n_nouns 계산엔 POS 태깅만 필요 (parser/ner 불필요 -> 속도 향상)
nlp.enable_pipe("tok2vec") if "tok2vec" in nlp.pipe_names else None

train = pd.read_csv(DATA_DIR / "train.csv")
print(f"전체 {len(train)}개 문장 spaCy 처리 중...")

docs = nlp.pipe(train["Sentence"].tolist(), batch_size=256)
n_nouns_list = []
for doc in docs:
    pos_counts = Counter(t.pos_ for t in doc if t.is_alpha)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_nouns_list.append(n_nouns)

train["n_nouns"] = n_nouns_list
out_df = train[["Id", "n_nouns"]]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)

print(f"n_nouns 분포:\n{train['n_nouns'].describe()}")
print(f"저장: {OUT_PATH}")
