"""통계적으로 의미 있는 표본 크기 계산과 비례 층화 표본 추출."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

DEFAULT_STRATA = ["sex", "age_group", "province"]


@dataclass
class SampleSizePlan:
    population: int
    confidence: float
    margin: float
    n_estimate: int          # 전체 추정용 최소 표본
    groups: int              # 비교할 하위집단 수 (0이면 비교 안 함)
    effect_size: float       # Cohen's d
    power: float
    n_per_group: int         # 집단 비교용 집단당 최소 표본
    n_total: int             # 최종 권장 표본

    def explain(self) -> str:
        lines = [
            f"모집단 {self.population:,}명, 신뢰수준 {self.confidence:.0%}, "
            f"허용 오차 ±{self.margin:.1%} 기준 최소 표본은 {self.n_estimate:,}명입니다 "
            "(Cochran 공식, 유한모집단 보정, p=0.5).",
        ]
        if self.groups >= 2:
            lines.append(
                f"{self.groups}개 집단을 비교하려면 효과크기 d={self.effect_size}, "
                f"검정력 {self.power:.0%}에서 집단당 {self.n_per_group:,}명이 필요합니다."
            )
        lines.append(f"권장 표본: {self.n_total:,}명")
        return "\n".join(lines)


def cochran_n(population: int, confidence: float = 0.95, margin: float = 0.05,
              p: float = 0.5) -> int:
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n0 = z**2 * p * (1 - p) / margin**2
    n = n0 / (1 + (n0 - 1) / population) if population > 0 else n0
    return min(population, math.ceil(n))


def two_group_n(effect_size: float = 0.3, alpha: float = 0.05, power: float = 0.8) -> int:
    """두 집단 평균 비교(양측)에 필요한 집단당 표본."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return math.ceil(2 * ((z_a + z_b) / effect_size) ** 2)


def plan_sample_size(population: int, confidence: float = 0.95, margin: float = 0.05,
                     groups: int = 0, effect_size: float = 0.3,
                     power: float = 0.8) -> SampleSizePlan:
    n_est = cochran_n(population, confidence, margin)
    n_group = two_group_n(effect_size, 1 - confidence, power) if groups >= 2 else 0
    n_total = min(population, max(n_est, n_group * groups))
    return SampleSizePlan(population, confidence, margin, n_est, groups,
                          effect_size, power, n_group, n_total)


def stratified_sample(df: pd.DataFrame, n: int, strata: list[str] | None = None,
                      seed: int = 42) -> pd.DataFrame:
    """층별 모집단 비율대로 n명을 뽑는다(최대잔여법으로 반올림)."""
    if n >= len(df):
        return df.copy()
    strata = [c for c in (strata or DEFAULT_STRATA) if c in df and df[c].nunique() > 1]
    if not strata:
        return df.sample(n=n, random_state=seed)

    sizes = df.groupby(strata, observed=True).size()
    quota = sizes / sizes.sum() * n
    alloc = quota.apply(math.floor)
    remainder = int(n - alloc.sum())
    if remainder > 0:
        order = (quota - alloc).sort_values(ascending=False).index[:remainder]
        alloc.loc[order] += 1

    parts = []
    for key, group in df.groupby(strata, observed=True):
        k = int(alloc.get(key, 0))
        if k > 0:
            parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def compare_distribution(population: pd.DataFrame, sample: pd.DataFrame,
                         column: str) -> pd.DataFrame:
    """모집단과 표본의 구성비를 나란히 보여 준다."""
    out = pd.DataFrame({
        "모집단(%)": population[column].value_counts(normalize=True) * 100,
        "표본(%)": sample[column].value_counts(normalize=True) * 100,
    }).fillna(0).round(1)
    return out.sort_values("모집단(%)", ascending=False)
