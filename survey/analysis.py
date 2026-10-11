"""리커트 응답 통계 처리: 기술통계, 척도 신뢰도, 집단 비교."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from .data import COLUMN_LABELS
from .questions import Question

MIN_GROUP_N = 5  # 이보다 작은 집단은 비교 검정에서 뺀다


def _ci95(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 2:
        return (np.nan, np.nan)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - half, x.mean() + half)


def item_summary(df: pd.DataFrame, questions: list[Question]) -> pd.DataFrame:
    """문항별 응답 수, 평균, 표준편차, 95% 신뢰구간, 응답 분포(%)."""
    rows = []
    for q in questions:
        x = pd.to_numeric(df[q.id], errors="coerce").dropna()
        lo, hi = _ci95(x)
        dist = x.value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
        rows.append({
            "문항": q.id, "영역": q.category, "내용": q.text, "역문항": "예" if q.reverse else "",
            "N": len(x), "평균": x.mean(), "표준편차": x.std(ddof=1),
            "95%CI 하한": lo, "95%CI 상한": hi,
            **{f"{k}점(%)": dist[k] for k in [1, 2, 3, 4, 5]},
            "긍정(4+5,%)": dist[4] + dist[5], "부정(1+2,%)": dist[1] + dist[2],
        })
    return pd.DataFrame(rows).round(2)


def recoded(df: pd.DataFrame, questions: list[Question]) -> pd.DataFrame:
    """역문항을 6-x로 바꾼 점수표."""
    out = pd.DataFrame(index=df.index)
    for q in questions:
        x = pd.to_numeric(df[q.id], errors="coerce")
        out[q.id] = 6 - x if q.reverse else x
    return out


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 2:
        return np.nan
    var_sum = items.var(ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_sum / total_var)) if total_var > 0 else np.nan


def scale_summary(df: pd.DataFrame, questions: list[Question]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """영역별 척도 점수(역문항 변환 후 평균)와 신뢰도. (요약표, 응답자별 점수) 반환."""
    rec = recoded(df, questions)
    scores = pd.DataFrame(index=df.index)
    rows = []
    for cat in dict.fromkeys(q.category for q in questions):
        ids = [q.id for q in questions if q.category == cat]
        scores[cat] = rec[ids].mean(axis=1)
        s = scores[cat].dropna()
        lo, hi = _ci95(s)
        rows.append({
            "영역": cat, "문항 수": len(ids), "문항": ", ".join(ids), "N": len(s),
            "평균": s.mean(), "표준편차": s.std(ddof=1), "95%CI 하한": lo, "95%CI 상한": hi,
            "Cronbach α": cronbach_alpha(rec[ids]),
        })
    return pd.DataFrame(rows).round(3), scores


def group_compare(df: pd.DataFrame, values: pd.DataFrame, by: str) -> pd.DataFrame:
    """`by` 집단별 평균과 차이 검정. values는 문항 점수 또는 영역 점수 열."""
    groups = df[by].astype(str)
    counts = groups.value_counts()
    keep = counts[counts >= MIN_GROUP_N].index
    rows = []
    for col in values.columns:
        data = pd.DataFrame({"g": groups, "y": pd.to_numeric(values[col], errors="coerce")}).dropna()
        data = data[data["g"].isin(keep)]
        samples = [d["y"].values for _, d in data.groupby("g")]
        row = {"항목": col}
        for name, d in data.groupby("g"):
            row[f"{name} (n={len(d)})"] = round(d["y"].mean(), 2)
        if len(samples) == 2:
            t = stats.ttest_ind(*samples, equal_var=False)
            u = stats.mannwhitneyu(*samples)
            pooled = np.sqrt((samples[0].var(ddof=1) + samples[1].var(ddof=1)) / 2)
            row.update({"검정": "Welch t", "통계량": t.statistic, "p값": t.pvalue,
                        "비모수 p(Mann-Whitney)": u.pvalue,
                        "효과크기": (samples[0].mean() - samples[1].mean()) / pooled if pooled else np.nan,
                        "효과크기 종류": "Cohen d"})
        elif len(samples) > 2:
            f = stats.f_oneway(*samples)
            h = stats.kruskal(*samples)
            grand = data["y"].mean()
            ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
            ss_total = ((data["y"] - grand) ** 2).sum()
            row.update({"검정": "일원분산분석", "통계량": f.statistic, "p값": f.pvalue,
                        "비모수 p(Kruskal-Wallis)": h.pvalue,
                        "효과크기": ss_between / ss_total if ss_total else np.nan,
                        "효과크기 종류": "η²"})
        else:
            row["검정"] = f"비교 불가(n≥{MIN_GROUP_N} 집단이 2개 미만)"
        if "p값" in row:
            row["유의(p<.05)"] = "*" if row["p값"] < 0.05 else ""
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def demographics_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parts = []
    for col in columns:
        vc = df[col].astype(str).value_counts()
        parts.append(pd.DataFrame({
            "변수": COLUMN_LABELS.get(col, col), "범주": vc.index,
            "명": vc.values, "%": (vc.values / vc.sum() * 100).round(1),
        }))
    return pd.concat(parts, ignore_index=True)


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def build_report(responses: pd.DataFrame, questions: list[Question],
                 compare_by: list[str], meta: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """엑셀 시트 묶음: 설정, 응답자 구성, 문항 통계, 영역 통계, 집단 비교, 원자료."""
    valid = responses[responses["error"].fillna("") == ""] if "error" in responses else responses
    sheets: dict[str, pd.DataFrame] = {}
    if meta is not None:
        sheets["설정"] = meta
    sheets["응답자 구성"] = demographics_table(valid, ["sex", "age_group", "province", "education_level"])
    sheets["문항 통계"] = item_summary(valid, questions)
    scale_table, scores = scale_summary(valid, questions)
    sheets["영역 통계"] = scale_table
    for by in compare_by:
        label = COLUMN_LABELS.get(by, by)
        sheets[f"비교_{label}_문항"] = group_compare(valid, valid[[q.id for q in questions]], by)
        sheets[f"비교_{label}_영역"] = group_compare(valid, scores, by)
    sheets["원자료"] = responses
    return sheets
