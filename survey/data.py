"""Nemotron-Personas-Korea 데이터 로드와 인구집단 필터링."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
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
    "cultural_background", "career_goals_and_ambitions", "hobbies_and_interests", "skills_and_expertise",
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
# 학력은 가나다순이 아니라 낮은 학력 → 높은 학력 순으로 보여 준다
EDUCATION_ORDER = ["무학", "초등학교", "중학교", "고등학교", "2~3년제 전문대학", "4년제 대학교", "대학원"]


# 표기가 조금 달라도 순서를 정할 수 있도록 단어로 판단한다(위에서부터 먼저 맞는 규칙)
_EDU_RULES = [("대학원", 6), ("석사", 6), ("박사", 6), ("4년제 미만", 4), ("2~3년제", 4), ("2-3년제", 4),
              ("전문대", 4), ("4년제", 5), ("대학교", 5), ("고등", 3), ("중학", 2), ("초등", 1),
              ("무학", 0), ("미취학", 0), ("없음", 0)]


def education_rank(value: str) -> tuple[int, str]:
    """낮은 학력 → 높은 학력 순서 값. 어떤 규칙에도 맞지 않으면 맨 뒤."""
    v = str(value)
    if v in EDUCATION_ORDER:
        return EDUCATION_ORDER.index(v), v
    for key, rank in _EDU_RULES:
        if key in v:
            return rank, v
    return len(EDUCATION_ORDER), v


# 시도는 두 글자 약칭으로 통일하고 행정구역 순서로 보여 준다
PROVINCE_ORDER = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
                  "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
_PROVINCE_PREFIX = {"충청북": "충북", "충청남": "충남", "전라북": "전북", "전북특별": "전북", "전라남": "전남",
                    "경상북": "경북", "경상남": "경남"}


def short_province(value) -> str:
    """'경상북도'·'경상북' → '경북', '서울특별시' → '서울'처럼 두 글자로 줄인다."""
    v = str(value).strip()
    for prefix, short in _PROVINCE_PREFIX.items():
        if v.startswith(prefix):
            return short
    return v[:2]


# 데이터 출처
#  - 로컬: NEMOTRON_DATA_DIR 폴더의 parquet 파일
#  - 원격(기본): Hugging Face가 제공하는 parquet 변환본에서 필요한 열만 범위 요청으로 읽는다(전체 2GB를 받지 않음).
#    표본 페르소나 텍스트는 Hugging Face datasets-server의 /filter API로 uuid를 지정해 가져오고,
#    실패한 사람만 /rows API로 행 번호를 지정해 다시 가져온다.
PARQUET_REVISION = "refs%2Fconvert%2Fparquet"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "nemotron-survey/1.0 (+https://github.com/2023swhs3rd-a11y/nemotron-survey)"
CATEGORY_COLUMNS = ["sex", "marital_status", "military_status", "family_type", "housing_type",
                    "education_level", "bachelors_field", "district", "province"]


def local_dir() -> Path | None:
    local = os.environ.get("NEMOTRON_DATA_DIR")
    return Path(local) if local else None


def _get_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
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
    if "province" in df:
        df["province"] = df["province"].map(short_province)
    for col in CATEGORY_COLUMNS:
        if col in df:
            df[col] = df[col].astype("category")
    if "education_level" in df:
        levels = sorted(df["education_level"].cat.categories, key=education_rank)
        df["education_level"] = df["education_level"].cat.reorder_categories(levels, ordered=True)
    if "province" in df:
        rank = {p: i for i, p in enumerate(PROVINCE_ORDER)}
        levels = sorted(df["province"].cat.categories, key=lambda p: (rank.get(p, len(rank)), p))
        df["province"] = df["province"].cat.reorder_categories(levels, ordered=True)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("category")
    return df


def dataset_revision(root: Path | None = None) -> str:
    """보고서에 남길 데이터 버전. 원격이면 Hugging Face 커밋(main과 parquet 변환본), 로컬이면 폴더."""
    root = root or local_dir()
    if root is not None:
        return f"로컬 파일 ({root})"
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        main = api.dataset_info(REPO_ID, timeout=10).sha
        parquet = api.dataset_info(REPO_ID, revision="refs/convert/parquet", timeout=10).sha
        return f"{REPO_ID} main@{main[:10]} · parquet 변환본@{parquet[:10]}"
    except Exception as e:  # 버전 확인이 안 돼도 조사는 진행한다
        return f"{REPO_ID} (커밋 확인 실패: {type(e).__name__})"


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

        with ThreadPoolExecutor(max_workers=len(paths)) as pool:  # 파일마다 동시에 읽는다
            parts = list(pool.map(read, paths))
        df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row_idx", range(len(df)))
    df = _compact(df)
    df.attrs["data_source"] = dataset_revision(root)  # 읽은 시점의 데이터 버전
    return df


def _fetch_row(idx: int) -> dict:
    data = _get_json(f"{DATASETS_SERVER}/rows?dataset={REPO_ID}&config=default&split=train"
                     f"&offset={int(idx)}&length=1")
    row = data["rows"][0]["row"]
    return {c: row.get(c) for c in PERSONA_COLUMNS}


def _fetch_by_uuid(uuids: list[str]) -> list[dict]:
    """uuid 최대 50개를 한 번의 /filter 요청으로 가져온다."""
    where = " OR ".join(f"\"uuid\"='{u}'" for u in uuids)
    query = urllib.parse.urlencode({"dataset": REPO_ID, "config": "default", "split": "train",
                                    "where": where, "length": 100})
    data = _get_json(f"{DATASETS_SERVER}/filter?{query}")
    return [{c: r["row"].get(c) for c in PERSONA_COLUMNS} for r in data.get("rows", [])]


def load_personas(sample: pd.DataFrame, root: Path | None = None, workers: int = 8,
                  progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
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

    chunks = [uuids[i:i + 50] for i in range(0, len(uuids), 50)]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_by_uuid, c) for c in chunks]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                rows.extend(f.result())
            except Exception:  # 필터 요청이 실패한 묶음은 아래에서 행 번호로 다시 가져온다
                pass
            if progress:
                progress(i, len(chunks))
    got = {r["uuid"] for r in rows}
    missing = sample[~sample["uuid"].astype(str).isin(got)]
    if len(missing):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(lambda i: _safe(_fetch_row, i), missing["row_idx"].tolist()):
                if r:
                    rows.append(r)
    out = pd.DataFrame(rows, columns=PERSONA_COLUMNS).drop_duplicates("uuid")
    return out[out["uuid"].isin(uuids)]  # 엉뚱한 행이 섞이지 않도록 uuid로 확인


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


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
            if values:  # 범주형 열은 문자열로 바꾸지 않고 바로 비교한다(100만 행에서 수십 배 빠름)
                column = df[col] if isinstance(df[col].dtype, pd.CategoricalDtype) else df[col].astype(str)
                mask &= column.isin(values)
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
                parts.append(f"{COLUMN_LABELS[col]}: {', '.join(v or '(해당 없음)' for v in values)}")
        if self.occupation_keyword.strip():
            parts.append(f"직업 키워드: {self.occupation_keyword}")
        return " / ".join(parts)


def filter_options(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """필터 화면에 쓸 선택지(열별 값 목록)와 시도별 시군구 목록. 한 번만 계산해 두고 쓴다."""
    cols = ["sex", "province", "district", "education_level", "marital_status",
            "housing_type", "family_type", "bachelors_field"]
    opts = {}
    for c in cols:
        present = set(df[c].dropna().unique())
        if isinstance(df[c].dtype, pd.CategoricalDtype) and df[c].cat.ordered:  # 학력 등 순서가 있는 범주
            opts[c] = [str(v) for v in df[c].cat.categories if v in present]
        else:
            opts[c] = sorted(str(v) for v in present)
    pairs = df[["province", "district"]].drop_duplicates().dropna()
    districts = {str(p): sorted(g["district"].astype(str).unique()) for p, g in pairs.groupby("province", observed=True)}
    return opts, districts
