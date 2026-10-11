"""Nemotron-Personas-Korea 데이터 로드와 인구집단 필터링."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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


# 데이터 출처
#  - 로컬: NEMOTRON_DATA_DIR 폴더의 parquet 파일
#  - 원격(기본): Hugging Face가 제공하는 parquet 변환본에서 필요한 열만 범위 요청으로 읽는다(전체 2GB를 받지 않음).
#    표본 페르소나 텍스트는 Hugging Face datasets-server의 /rows API로 행 번호를 지정해 가져온다.
PARQUET_REVISION = "refs%2Fconvert%2Fparquet"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
CATEGORY_COLUMNS = ["sex", "marital_status", "military_status", "family_type", "housing_type",
                    "education_level", "bachelors_field", "district", "province"]


def local_dir() -> Path | None:
    local = os.environ.get("NEMOTRON_DATA_DIR")
    return Path(local) if local else None


def _get_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def remote_parquet_paths() -> list[str]:
    info = _get_json(f"{DATASETS_SERVER}/parquet?dataset={REPO_ID}")
    names = sorted(f["filename"] for f in info["parquet_files"]
                   if f.get("config") == "default" and f.get("split") == "train")
    return [f"datasets/{REPO_ID}@{PARQUET_REVISION}/default/train/{n}" for n in names]


def _dataset(root: Path) -> ds.Dataset:
    files = sorted(str(p) for p in root.rglob("*.parquet") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"{root} 아래에 parquet 파일이 없습니다.")
    return ds.dataset(files, format="parquet")


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORY_COLUMNS:
        if col in df:
            df[col] = df[col].astype("category")
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("category")
    return df


def load_demographics(root: Path | None = None) -> pd.DataFrame:
    """인구통계 열만 읽는다(텍스트 열은 빼서 메모리를 아낀다). row_idx는 전체 데이터에서의 행 번호."""
    root = root or local_dir()
    if root is not None:
        df = _dataset(root).to_table(columns=DEMOGRAPHIC_COLUMNS).to_pandas()
    else:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        paths = remote_parquet_paths()

        def read(path: str) -> pd.DataFrame:
            with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f:
                return pq.read_table(f, columns=DEMOGRAPHIC_COLUMNS).to_pandas()

        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(read, paths))
        df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row_idx", range(len(df)))
    return _compact(df)


def _fetch_row(idx: int) -> dict:
    data = _get_json(f"{DATASETS_SERVER}/rows?dataset={REPO_ID}&config=default&split=train"
                     f"&offset={int(idx)}&length=1")
    row = data["rows"][0]["row"]
    return {c: row.get(c) for c in PERSONA_COLUMNS}


def load_personas(sample: pd.DataFrame, root: Path | None = None, workers: int = 8) -> pd.DataFrame:
    """표본으로 뽑힌 사람들의 페르소나 텍스트만 읽는다."""
    root = root or local_dir()
    uuids = sample["uuid"].astype(str).tolist()
    if root is not None:
        dataset = _dataset(root)
        if not uuids:
            return dataset.schema.empty_table().select(PERSONA_COLUMNS).to_pandas()
        values = pa.array(uuids, type=dataset.schema.field("uuid").type)
        return dataset.to_table(columns=PERSONA_COLUMNS, filter=ds.field("uuid").isin(values)).to_pandas()
    if not uuids:
        return pd.DataFrame(columns=PERSONA_COLUMNS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_fetch_row, sample["row_idx"].tolist()))
    out = pd.DataFrame(rows)
    return out[out["uuid"].isin(uuids)]  # 행 번호가 어긋나면 잘못된 텍스트를 붙이지 않도록 uuid로 확인


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
                mask &= df[col].astype(str).isin(values)
        if self.occupation_keyword.strip():
            keywords = [k.strip() for k in self.occupation_keyword.split(",") if k.strip()]
            pattern = "|".join(keywords)
            mask &= df["occupation"].astype(str).str.contains(pattern, regex=True, na=False)
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
