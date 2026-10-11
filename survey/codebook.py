"""생활서사 코드북: 페르소나 텍스트의 교육·진로 단서를 0~2점으로 점수화하고, 이를 리커트 응답으로 추정한다.

참고: 유승철·임승희(2026), 「소비자 합성 페르소나 데이터를 활용한 생활서사 기반 광고 전략 탐색」, 광고연구 149호.
페르소나 텍스트를 통합해 코드북·키워드 규칙으로 카테고리 단서 강도를 0~2점으로 계량화한 절차를
대학·진로교육 설문에 맞게 옮겼다. 추정 응답은 실제 의견의 예측값이 아니라 가설 생성용 지표다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .questions import Question

CODEBOOK_PATH = Path(__file__).with_name("codebook.json")
DEMO_TERMS = ["edu_high", "young", "old"]
HIGH_EDU_WORDS = ["4년제", "대학원", "석사", "박사"]


def load_codebook(path: Path = CODEBOOK_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cue_labels(codebook: dict) -> dict[str, str]:
    return {k: v["label"] for k, v in codebook["cues"].items()}


def combined_text(row: pd.Series, fields: list[str]) -> str:
    return " ".join(str(row[f]) for f in fields if f in row and pd.notna(row[f]) and str(row[f]).strip())


def score_text(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """서로 다른 키워드가 0개면 0점, 1개면 1점(약한 단서), 2개 이상이면 2점(반복·구체적 단서)."""
    hits = [k for k in keywords if k in text]
    return min(len(hits), 2), hits


def score_cues(sample: pd.DataFrame, codebook: dict) -> tuple[pd.DataFrame, pd.Series]:
    """응답자별 단서 점수표와, 단서 근거(찾은 키워드) 설명을 돌려준다."""
    fields = codebook["text_fields"]
    scores, notes = {}, []
    texts = [combined_text(r, fields) for _, r in sample.iterrows()]
    for cue, spec in codebook["cues"].items():
        res = [score_text(t, spec["keywords"]) for t in texts]
        scores[cue] = [r[0] for r in res]
        notes.append([f"{spec['label']}: {', '.join(r[1])}" if r[1] else "" for r in res])
    has_text = pd.Series([bool(t) for t in texts], index=sample.index)
    evidence = pd.Series(["; ".join(n for n in parts if n) for parts in zip(*notes)], index=sample.index)
    evidence[~has_text] = "(페르소나 텍스트 없음: 인구통계만 사용)"
    return pd.DataFrame(scores, index=sample.index), evidence


def demographic_terms(sample: pd.DataFrame) -> pd.DataFrame:
    edu = sample.get("education_level", pd.Series("", index=sample.index)).astype(str)
    age = pd.to_numeric(sample.get("age", pd.Series(40, index=sample.index)), errors="coerce").fillna(40)
    return pd.DataFrame({
        "edu_high": edu.apply(lambda e: int(any(w in e for w in HIGH_EDU_WORDS))),
        "young": (age < 35).astype(int),
        "old": (age >= 60).astype(int),
    }, index=sample.index)


def weight_table(questions: list[Question], codebook: dict) -> pd.DataFrame:
    """문항 × (기준점, 단서, 인구통계 항) 가중치표. 코드북에 없는 문항은 기준점 3, 가중치 0."""
    cols = ["intercept", *codebook["cues"], *DEMO_TERMS]
    rows = []
    for q in questions:
        w = codebook.get("weights", {}).get(q.id, {})
        rows.append({"문항": q.id, **{c: float(w.get(c, 3.0 if c == "intercept" else 0.0)) for c in cols}})
    return pd.DataFrame(rows).set_index("문항")


def estimate_responses(sample: pd.DataFrame, questions: list[Question], codebook: dict,
                       weights: pd.DataFrame | None = None, noise_sd: float | None = None,
                       seed: int = 0) -> pd.DataFrame:
    """단서 점수와 인구통계로 문항별 1~5 응답을 추정한다. 같은 사람·같은 설정이면 같은 값이 나온다."""
    weights = weight_table(questions, codebook) if weights is None else weights
    noise_sd = codebook.get("noise_sd", 0.7) if noise_sd is None else noise_sd
    cues, evidence = score_cues(sample, codebook)
    X = pd.concat([cues, demographic_terms(sample)], axis=1)
    out = sample.copy()
    for col in cues:
        out[f"단서_{col}"] = cues[col]
    for q in questions:
        w = weights.loc[q.id] if q.id in weights.index else pd.Series({"intercept": 3.0})
        linear = w.get("intercept", 3.0) + sum(X[c] * w.get(c, 0.0) for c in X.columns)
        noise = np.array([_rng(u, q.id, seed).normal(0, noise_sd) for u in sample["uuid"]])
        out[q.id] = np.clip(np.rint(linear + noise), 1, 5).astype(int)
    out["comment"] = evidence
    out["error"] = ""
    return out


def _rng(uuid: str, qid: str, seed: int) -> np.random.Generator:
    h = int(hashlib.md5(f"{uuid}|{qid}|{seed}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def cue_summary(cues: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cues.columns:
        x = cues[c]
        dist = x.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0) * 100
        rows.append({"단서": labels.get(c, c), "N": len(x), "평균(0~2)": x.mean(), "표준편차": x.std(ddof=1),
                     "0점(%)": dist[0], "1점(%)": dist[1], "2점(%)": dist[2]})
    return pd.DataFrame(rows).round(2)


def cue_friedman(cues: pd.DataFrame) -> dict:
    """같은 사람이 여러 단서 점수를 갖는 반복측정 구조 → Friedman 검정."""
    from scipy import stats
    if cues.shape[1] < 3 or len(cues) < 3:
        return {}
    r = stats.friedmanchisquare(*[cues[c] for c in cues.columns])
    return {"통계량": round(float(r.statistic), 3), "p값": float(r.pvalue)}
