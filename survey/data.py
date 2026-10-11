"""Nemotron-Personas-Korea 데이터 로드와 인구집단 필터링."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

REPO_ID = "nvidia/Nemotron-Personas-Korea"

DEMOGRAPHIC_COLUMNS = [
    "uuid", "sex", "age", "marital_status", "military_status", "family_type",
    "housing_type", "education_level", "bachelors_field", "occupation",
    "district", "province",
]
PERSONA_COLUMNS = [
    "uuid", "persona", "professional_persona", "family_persona",
    "cultural_background", "career_goals_and_ambitions", "hobbies_and_interests",
]

# 화면과 통계표에 쓰는 한국어 이름
COLUMN_LABELS = {
    "sex": "성별", "age": "나이", "age_group": "연령대", "marital_status": "혼인 상태",
    "military_status": "병역", "family_type": "가구 유형", "housing_type": "주거 형태",
    "education_level": "학력", "bachelors_field": "전공 계열", "occupation": "직업",
    "district": "시군구", "province": "시도",
}

AGE_BINS = [19, 30, 40, 50, 60, 70, 200]
AGE_LABELS = ["19-29세", "30대", "40대", "50대", "60대", "70세 이상"]


def data_dir() -> Path:
    """데이터 parquet 파일이 있는 폴더. 없으면 Hugging Face에서 내려받는다."""
    local = os.environ.get("NEMOTRON_DATA_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["*.parquet"])
    return Path(path)


def _dataset(root: Path) -> ds.Dataset:
    files = sorted(str(p) for p in root.rglob("*.parquet") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"{root} 아래에 parquet 파일이 없습니다.")
    return ds.dataset(files, format="parquet")


def load_demographics(root: Path | None = None) -> pd.DataFrame:
    """인구통계 열만 읽는다(텍스트 열은 빼서 메모리를 아낀다)."""
    root = root or data_dir()
    cache = root / "_demographics_cache.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = _dataset(root).to_table(columns=DEMOGRAPHIC_COLUMNS).to_pandas()
        try:
            df.to_parquet(cache, index=False)
        except OSError:
            pass
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return df


def load_personas(uuids: list[str], root: Path | None = None) -> pd.DataFrame:
    """표본으로 뽑힌 uuid의 페르소나 텍스트만 읽는다."""
    root = root or data_dir()
    dataset = _dataset(root)
    if not len(uuids):
        return dataset.schema.empty_table().select(PERSONA_COLUMNS).to_pandas()
    values = pa.array(list(uuids), type=dataset.schema.field("uuid").type)
    table = dataset.to_table(columns=PERSONA_COLUMNS, filter=ds.field("uuid").isin(values))
    return table.to_pandas()


@dataclass
class PopulationFilter:
    """사용자가 고른 인구집단 조건. 빈 리스트는 '전체'를 뜻한다."""
    age_min: int = 19
    age_max: int = 99
    sex: list[str] = field(default_factory=list)
    province: list[str] = field(default_factory=list)
    district: list[str] = field(default_factory=list)
    education_level: list[str] = field(default_factory=list)
    marital_status: list[str] = field(default_factory=list)
    housing_type: list[str] = field(default_factory=list)
    family_type: list[str] = field(default_factory=list)
    bachelors_field: list[str] = field(default_factory=list)
    occupation_keyword: str = ""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["age"].between(self.age_min, self.age_max)
        for col in ["sex", "province", "district", "education_level", "marital_status",
                    "housing_type", "family_type", "bachelors_field"]:
            values = getattr(self, col)
            if values:
                mask &= df[col].isin(values)
        if self.occupation_keyword.strip():
            keywords = [k.strip() for k in self.occupation_keyword.split(",") if k.strip()]
            pattern = "|".join(keywords)
            mask &= df["occupation"].fillna("").str.contains(pattern, regex=True)
        return df[mask]

    def describe(self) -> str:
        parts = [f"나이 {self.age_min}~{self.age_max}세"]
        for col in ["sex", "province", "district", "education_level", "marital_status",
                    "housing_type", "family_type", "bachelors_field"]:
            values = getattr(self, col)
            if values:
                parts.append(f"{COLUMN_LABELS[col]}: {', '.join(values)}")
        if self.occupation_keyword.strip():
            parts.append(f"직업 키워드: {self.occupation_keyword}")
        return " / ".join(parts)
