"""명령줄에서 가상 설문을 한 번에 돌리고 엑셀 보고서를 outputs/에 저장한다.

예) 서울·경기 19~29세, 성별·연령대 비교, 생활서사 코드북 추정:
    python cli.py --age 19 29 --province 서울 경기 --compare sex age_group
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from survey import analysis
from survey.codebook import estimate_responses, load_codebook
from survey.data import PopulationFilter, load_demographics, load_personas
from survey.questions import load_all_questions
from survey.sampling import plan_sample_size, stratified_sample
from survey.simulate import DEFAULT_MODEL, run_survey


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--age", nargs=2, type=int, default=[19, 99], metavar=("MIN", "MAX"))
    for col in ["sex", "province", "district", "education", "marital", "housing", "family", "field"]:
        p.add_argument(f"--{col}", nargs="*", default=[])
    p.add_argument("--occupation", default="", help="직업 키워드(쉼표 구분)")
    p.add_argument("--n", type=int, default=0, help="표본 수(0이면 자동 계산)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--compare", nargs="*", default=["sex", "age_group"])
    p.add_argument("--strata", nargs="*", default=["sex", "age_group", "province"])
    p.add_argument("--mode", choices=["codebook", "demo", "llm"], default="codebook",
                   help="codebook: 생활서사 코드북 추정(기본), demo: 규칙 기반 임의 응답, llm: Claude 응답")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    flt = PopulationFilter(age_min=a.age[0], age_max=a.age[1], sex=a.sex, province=a.province,
                           district=a.district, education_level=a.education, marital_status=a.marital,
                           housing_type=a.housing, family_type=a.family, bachelors_field=a.field,
                           occupation_keyword=a.occupation)
    population = flt.apply(load_demographics())
    print(f"대상: {flt.describe()} → {len(population):,}명")
    if population.empty:
        raise SystemExit("조건에 맞는 페르소나가 없습니다.")

    groups = population[a.compare[0]].nunique() if a.compare else 0
    plan = plan_sample_size(len(population), a.confidence, a.margin, groups=groups)
    print(plan.explain())
    n = a.n or plan.n_total

    sample = stratified_sample(population, n, a.strata, a.seed)
    sample = sample.merge(load_personas(sample), on="uuid", how="left")
    questions = load_all_questions()
    print(f"{len(sample):,}명 × {len(questions)}문항 응답 생성({a.mode})")
    if a.mode == "codebook":
        responses = estimate_responses(sample, questions, load_codebook())
    else:
        responses = run_survey(sample, questions, mode=a.mode, model=a.model, effort=a.effort,
                               workers=a.workers,
                               progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True))
        print()

    rows = {"실행 시각": datetime.now().isoformat(timespec="seconds"), "응답 방식": a.mode,
            "대상 조건": flt.describe(), "모집단 수": len(population), "표본 수": len(sample),
            "층화 기준": ", ".join(a.strata), "표본 시드": a.seed}
    if a.mode == "llm":
        rows["모델"] = a.model
    meta = pd.DataFrame({"항목": list(rows), "값": [str(v) for v in rows.values()]})
    sheets = analysis.build_report(responses, questions, a.compare, meta)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    path = out / f"survey_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    path.write_bytes(analysis.to_excel(sheets))
    print(sheets["문항 통계"][["문항", "N", "평균", "표준편차", "긍정(4+5,%)"]].to_string(index=False))
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    main()
