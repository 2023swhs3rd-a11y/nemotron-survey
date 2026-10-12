"""리커트 응답 통계 처리: 기술통계, 척도 신뢰도, 집단 비교."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from .data import COLUMN_LABELS
from .questions import Question

MIN_GROUP_N = 5  # 이보다 작은 집단은 비교 검정에서 뺀다
POSTHOC_MAX_GROUPS = 12  # 사후검정은 집단 쌍이 많으면 느려지므로 이 수 이하일 때만 한다
P_COLUMNS = ["p값", "p(Holm)", "비모수 p(Mann-Whitney)", "비모수 p(Kruskal-Wallis)", "p(Games-Howell)"]


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


def _ordered_groups(col: pd.Series) -> pd.Series:
    """집단 열을 문자열로 바꾸되, 순서가 있는 범주(연령대·학력)는 그 순서를 지킨다."""
    if isinstance(col.dtype, pd.CategoricalDtype) and col.cat.ordered:
        order = [str(c) for c in col.cat.categories]
        return pd.Series(pd.Categorical(col.astype(str), categories=order, ordered=True), index=col.index)
    return col.astype(str)


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm 단계적 보정. 한 표(같은 집단 변수로 비교한 문항들)를 한 묶음으로 본다."""
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().sort_values()
    m = len(valid)
    adjusted = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * valid.to_numpy()))
    return pd.Series(adjusted, index=valid.index).reindex(p.index)


def welch_anova(samples: list[np.ndarray]) -> tuple[float, float, float, float]:
    """집단 분산이 달라도 되는 Welch 일원분산분석. (F, df1, df2, p)"""
    k = len(samples)
    n = np.array([len(s) for s in samples], dtype=float)
    m = np.array([s.mean() for s in samples])
    w = n / np.array([s.var(ddof=1) for s in samples])
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df2 = (k ** 2 - 1) / (3 * tmp)
    return f, k - 1, df2, float(stats.f.sf(f, k - 1, df2))


def _comparison_data(df: pd.DataFrame, values: pd.DataFrame, by: str, col: str) -> pd.DataFrame:
    groups = _ordered_groups(df[by])
    counts = groups.value_counts()
    data = pd.DataFrame({"g": groups, "y": pd.to_numeric(values[col], errors="coerce")}).dropna()
    return data[data["g"].isin(counts[counts >= MIN_GROUP_N].index)]


def group_compare(df: pd.DataFrame, values: pd.DataFrame, by: str) -> pd.DataFrame:
    """`by` 집단별 평균과 차이 검정. values는 문항 점수 또는 영역 점수 열.
    두 집단은 Welch t, 세 집단 이상은 Welch 분산분석(집단 분산이 다를 수 있으므로)과 비모수 검정을 함께 내고,
    표 안의 여러 검정은 Holm 방식으로 보정한 p값으로 유의 여부를 표시한다."""
    rows = []
    for col in values.columns:
        data = _comparison_data(df, values, by, col)
        parts = list(data.groupby("g", observed=True))
        samples = [d["y"].values for _, d in parts]
        row = {"항목": col}
        for name, d in parts:
            row[f"{name} (n={len(d)})"] = round(d["y"].mean(), 2)
        if len(samples) == 2:
            t = stats.ttest_ind(*samples, equal_var=False)
            u = stats.mannwhitneyu(*samples)
            pooled = np.sqrt((samples[0].var(ddof=1) + samples[1].var(ddof=1)) / 2)
            row.update({"검정": "Welch t", "통계량": t.statistic, "p값": t.pvalue,
                        "비모수 p(Mann-Whitney)": u.pvalue,
                        "효과크기": (samples[0].mean() - samples[1].mean()) / pooled if pooled else np.nan,
                        "효과크기 종류": f"Cohen d ({parts[0][0]} − {parts[1][0]})"})
        elif len(samples) > 2:
            if all(s.var(ddof=1) > 0 for s in samples):
                f_stat, _, _, f_p = welch_anova(samples)
                test = "Welch 분산분석"
            else:  # 한 집단이라도 응답이 모두 같으면 Welch 가중치를 못 구한다
                f_stat, f_p = stats.f_oneway(*samples)
                test = "일원분산분석"
            h = stats.kruskal(*samples)
            grand = data["y"].mean()
            ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
            ss_total = ((data["y"] - grand) ** 2).sum()
            row.update({"검정": test, "통계량": f_stat, "p값": f_p,
                        "비모수 p(Kruskal-Wallis)": h.pvalue,
                        "효과크기": ss_between / ss_total if ss_total else np.nan,
                        "효과크기 종류": "η²"})
        else:
            row["검정"] = f"비교 불가(n≥{MIN_GROUP_N} 집단이 2개 미만)"
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p값" in out:
        out.insert(out.columns.get_loc("p값") + 1, "p(Holm)", holm(out["p값"]))
        out["유의(Holm p<.05)"] = np.where(out["p(Holm)"] < 0.05, "*", "")
    # p값은 반올림하지 않는다(0.0으로 보이지 않게). 화면 표기는 p_display에서 한다.
    rest = [c for c in out.columns if c not in P_COLUMNS]
    out[rest] = out[rest].round(4)
    return out


def games_howell(df: pd.DataFrame, values: pd.DataFrame, by: str) -> pd.DataFrame:
    """세 집단 이상에서 어느 두 집단이 다른지 보는 사후검정(집단 분산이 달라도 됨)."""
    rows = []
    for col in values.columns:
        parts = list(_comparison_data(df, values, by, col).groupby("g", observed=True))
        k = len(parts)
        if not 3 <= k <= POSTHOC_MAX_GROUPS:
            continue
        for i in range(k):
            for j in range(i + 1, k):
                (a, ya), (b, yb) = parts[i], parts[j]
                x, y = ya["y"].to_numpy(), yb["y"].to_numpy()
                vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
                diff = x.mean() - y.mean()
                if vx + vy == 0:
                    p = np.nan
                    dof = np.nan
                else:
                    dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
                    q = abs(diff) / np.sqrt((vx + vy) / 2)
                    # 자유도가 2000을 넘으면 무한대 근사(p 차이 .001 미만)로 계산 시간을 줄인다
                    p = float(stats.studentized_range.sf(q, k, dof if dof < 2000 else np.inf))
                rows.append({"항목": col, "집단 A": a, "집단 B": b, "평균 차이(A−B)": diff,
                             "자유도": dof, "p(Games-Howell)": p,
                             "유의(p<.05)": "*" if p < 0.05 else ""})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rest = [c for c in out.columns if c not in P_COLUMNS]
    out[rest] = out[rest].round(3)
    return out


def gap_decomposition(df: pd.DataFrame, full: pd.DataFrame, reduced: pd.DataFrame, by: str) -> pd.DataFrame:
    """집단 차이 가운데 학력·연령 항(가중치로 직접 넣은 부분)이 만든 몫.

    전체 모형에서 평균이 가장 높은 집단과 가장 낮은 집단을 고르고, 같은 두 집단의 차이를 서사 단서만 쓴
    모형(같은 오차)과 비교한다. 몫이 크면 그 차이는 데이터에서 나온 발견이 아니라 가정에서 나온 값이다."""
    groups = _ordered_groups(df[by])
    counts = groups.value_counts()
    keep = counts[counts >= MIN_GROUP_N].index
    rows = []
    for col in full.columns:
        f = full[col].groupby(groups, observed=True).mean()
        r = reduced[col].groupby(groups, observed=True).mean()
        f, r = f[f.index.isin(keep)], r[r.index.isin(keep)]
        if len(f) < 2:
            continue
        hi, lo = f.idxmax(), f.idxmin()
        gap, gap_cue = f[hi] - f[lo], r[hi] - r[lo]
        rows.append({"항목": col, "비교한 두 집단": f"{hi} − {lo}", "전체 모형 차이": gap,
                     "서사 단서만 쓴 차이": gap_cue,
                     "학력·연령 항이 만든 몫(%)": (gap - gap_cue) / gap * 100 if gap >= 0.05 else np.nan})
    return pd.DataFrame(rows).round({"전체 모형 차이": 3, "서사 단서만 쓴 차이": 3, "학력·연령 항이 만든 몫(%)": 1})


def p_text(p: float) -> str:
    """APA 방식 p값 표기: .001 미만은 '<.001', 나머지는 소수 셋째 자리."""
    if pd.isna(p):
        return ""
    return "<.001" if p < 0.001 else f"{p:.3f}"


def p_display(table: pd.DataFrame) -> pd.DataFrame:
    """화면용 사본: p값 열을 '<.001' 같은 표기로 바꾼다. 엑셀에는 원래 숫자를 남긴다."""
    out = table.copy()
    for c in P_COLUMNS:
        if c in out:
            out[c] = out[c].map(p_text)
    return out


def demographics_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parts = []
    for col in columns:
        vc = _ordered_groups(df[col]).value_counts()
        vc = vc[vc > 0].sort_index() if isinstance(vc.index, pd.CategoricalIndex) else vc
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
        posthoc = games_howell(valid, scores, by)
        if not posthoc.empty:
            sheets[f"사후_{label}_영역"] = posthoc
    sheets["원자료"] = responses
    return sheets
