"""응답 파일 업로드: 양식 만들기와 올린 파일 검사."""
from __future__ import annotations

import pandas as pd

from .questions import Question

TEMPLATE_INFO = ["uuid", "sex", "age", "province", "district", "education_level", "occupation", "persona"]


def response_template(sample: pd.DataFrame, qs: list[Question]) -> pd.DataFrame:
    out = sample[[c for c in TEMPLATE_INFO if c in sample]].copy()
    for q in qs:
        out[q.id] = pd.NA
    return out


def read_responses(file, sample: pd.DataFrame, qs: list[Question]) -> pd.DataFrame:
    """올린 응답 파일을 검사해 표본 정보와 합친다."""
    data = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    missing = [c for c in ["uuid"] + [q.id for q in qs] if c not in data.columns]
    if missing:
        raise ValueError(f"응답 파일에 다음 열이 없습니다: {', '.join(missing)}")
    data["uuid"] = data["uuid"].astype(str)
    data = data[data["uuid"].isin(sample["uuid"].astype(str))]
    if data.empty:
        raise ValueError("응답 파일의 uuid가 지금 표본과 맞지 않습니다. 이번 표본에서 내려받은 양식을 쓰세요.")
    ids = [q.id for q in qs]
    vals = data[ids].apply(pd.to_numeric, errors="coerce")
    bad = vals.notna() & ~vals.isin([1, 2, 3, 4, 5])
    if bad.any().any():
        raise ValueError(f"1~5 이외의 값이 있습니다: {', '.join(vals.columns[bad.any()])}")
    data[ids] = vals
    empty = vals.isna().all(axis=1)
    data["error"] = empty.map({True: "응답 없음", False: ""})
    data["comment"] = data.get("comment", "")
    base = sample.drop(columns=[c for c in data.columns if c in sample and c != "uuid"])
    base = base.assign(uuid=base["uuid"].astype(str))
    return base.merge(data[["uuid", *ids, "comment", "error"]], on="uuid", how="inner")
