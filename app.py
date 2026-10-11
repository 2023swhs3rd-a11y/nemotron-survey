"""대학·진로교육 인식 가상 설문 (Nemotron-Personas-Korea 기반) - Streamlit 앱.

실행: streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survey import analysis
from survey.data import COLUMN_LABELS, PopulationFilter, load_demographics, load_personas
from survey.questions import (Question, load_custom_questions, load_default_questions,
                              load_scale, next_custom_id, save_custom_questions)
from survey.sampling import compare_distribution, plan_sample_size, stratified_sample
from survey.simulate import DEFAULT_MODEL, MODEL_CHOICES, run_survey

st.set_page_config(page_title="대학·진로교육 인식 가상 설문", layout="wide")
# 리커트 1~5점 색: 부정(빨강) - 중립(회색) - 긍정(파랑)
LIKERT_COLORS = ["#c7383a", "#ec8a84", "#b5b4ae", "#6da7ec", "#1c5cab"]
COMPARE_CHOICES = ["sex", "age_group", "province", "education_level", "marital_status",
                   "housing_type", "family_type", "bachelors_field"]


@st.cache_data(show_spinner="페르소나 100만 건의 인구통계 정보를 불러오는 중...")
def get_demographics() -> pd.DataFrame:
    return load_demographics()


def options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(df[col].dropna().astype(str).unique().tolist())


def likert_chart(summary: pd.DataFrame, scale: dict[int, str]) -> go.Figure:
    fig = go.Figure()
    labels = [f"{r['문항']} {r['내용'][:28]}…" if len(r["내용"]) > 28 else f"{r['문항']} {r['내용']}"
              for _, r in summary.iterrows()]
    for k in [1, 2, 3, 4, 5]:
        vals = summary[f"{k}점(%)"]
        fig.add_bar(
            y=labels, x=vals, name=f"{k} {scale[k]}", orientation="h",
            marker=dict(color=LIKERT_COLORS[k - 1], line=dict(color="white", width=2)),
            text=[f"{v:.0f}" if v >= 6 else "" for v in vals], textposition="inside",
            hovertemplate="%{y}<br>" + f"{k}점 {scale[k]}: " + "%{x:.1f}%<extra></extra>",
        )
    fig.update_layout(barmode="stack", height=110 + 42 * len(summary), xaxis_title="응답 비율(%)",
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", traceorder="normal", x=0, y=1.0,
                                  xanchor="left", yanchor="bottom"),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


st.title("대학·진로교육 인식 가상 설문")
st.caption("NVIDIA Nemotron-Personas-Korea 합성 페르소나 100만 명 중 원하는 인구집단을 골라 "
           "리커트 5점 척도 설문을 시뮬레이션하고 바로 통계를 냅니다.")

# ---------------- 사이드바: 응답 생성 설정 ----------------
with st.sidebar:
    st.header("응답 생성 설정")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    mode = st.radio(
        "응답 방식", ["llm", "demo"], index=0 if env_key else 1,
        format_func=lambda m: "Claude가 페르소나로 응답" if m == "llm" else "데모(규칙 기반 임의 응답)",
    )
    api_key = env_key
    if mode == "llm":
        typed = st.text_input("Claude API 키", type="password", placeholder="sk-ant-api03-...",
                              help="console.anthropic.com의 API Keys에서 발급합니다. "
                                   "이 브라우저 세션에서만 쓰이고 저장되지 않습니다.").strip()
        api_key = typed or env_key
        if typed:
            st.caption("입력한 키를 사용합니다.")
        elif env_key:
            st.caption(f"터미널에 설정된 키를 사용합니다 ({env_key[:10]}…). 다른 키를 쓰려면 위에 입력하세요.")
        else:
            st.warning("Claude API 키를 입력하세요.")
        model = st.selectbox("모델", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
        effort = st.select_slider("추론 강도(effort)", ["low", "medium", "high"], value="low")
        workers = st.slider("동시 요청 수", 1, 32, 8)
    else:
        st.info("데모 응답은 화면과 통계 흐름 확인용입니다. 연구 결과로 쓰지 마세요.")
        model, effort, workers = DEFAULT_MODEL, "low", 8
    has_key = bool(api_key)

demo = get_demographics()

# ---------------- 1. 인구집단 선택 ----------------
st.header("1. 조사 대상 인구집단")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.slider("나이", 19, 99, (19, 99))
    sex = st.multiselect("성별", options(demo, "sex"))
    marital = st.multiselect("혼인 상태", options(demo, "marital_status"))
with c2:
    province = st.multiselect("시도", options(demo, "province"))
    district_pool = demo[demo["province"].isin(province)] if province else demo
    district = st.multiselect("시군구", options(district_pool, "district"))
    housing = st.multiselect("주거 형태", options(demo, "housing_type"))
with c3:
    education = st.multiselect("학력", options(demo, "education_level"))
    field = st.multiselect("전공 계열", options(demo, "bachelors_field"))
    family = st.multiselect("가구 유형", options(demo, "family_type"))
occupation_kw = st.text_input("직업 키워드(쉼표로 여러 개)", placeholder="예: 교사, 교수, 강사")

flt = PopulationFilter(age_min=age[0], age_max=age[1], sex=sex, province=province, district=district,
                       education_level=education, marital_status=marital, housing_type=housing,
                       family_type=family, bachelors_field=field, occupation_keyword=occupation_kw)
population = flt.apply(demo)
st.metric("조건에 맞는 페르소나 수", f"{len(population):,}명")
if population.empty:
    st.stop()

# ---------------- 2. 표본 크기 ----------------
st.header("2. 표본 크기와 추출")
s1, s2, s3, s4 = st.columns(4)
confidence = s1.selectbox("신뢰수준", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}")
margin = s2.selectbox("허용 오차", [0.03, 0.05, 0.07, 0.10], index=1, format_func=lambda x: f"±{x:.0%}")
plan_by = s3.selectbox("집단 비교 계획", ["(없음)"] + COMPARE_CHOICES,
                       format_func=lambda c: COLUMN_LABELS.get(c, c))
effect = s4.selectbox("검출할 효과크기(d)", [0.2, 0.3, 0.5], index=1,
                      help="0.2 작음, 0.5 중간. 작을수록 더 많은 표본이 필요합니다.")
groups = population[plan_by].nunique() if plan_by != "(없음)" else 0
plan = plan_sample_size(len(population), confidence, margin, groups=groups, effect_size=effect)
st.info(plan.explain())
n = st.number_input("실제 표본 수", min_value=10, max_value=len(population),
                    value=min(plan.n_total, len(population)), step=10)
k1, k2 = st.columns(2)
strata = k1.multiselect("층화 기준(모집단 비율 유지)", COMPARE_CHOICES,
                        default=["sex", "age_group", "province"],
                        format_func=lambda c: COLUMN_LABELS.get(c, c))
seed = k2.number_input("난수 시드(같은 값이면 같은 표본)", value=42, step=1)

if st.button("표본 추출", type="primary"):
    sample = stratified_sample(population, int(n), strata, int(seed))
    personas = load_personas(sample["uuid"].tolist())
    st.session_state.sample = sample.merge(personas, on="uuid", how="left")
    st.session_state.pop("responses", None)

sample = st.session_state.get("sample")
if sample is not None:
    st.success(f"{len(sample):,}명을 뽑았습니다.")
    with st.expander("모집단과 표본 구성 비교"):
        cols = st.columns(3)
        for i, col in enumerate(["sex", "age_group", "province"]):
            cols[i].write(f"**{COLUMN_LABELS[col]}**")
            cols[i].dataframe(compare_distribution(population, sample, col))
    with st.expander("표본 페르소나 미리 보기"):
        st.dataframe(sample.head(20))

# ---------------- 3. 문항 ----------------
st.header("3. 설문 문항 (리커트 5점)")
scale = load_scale()
st.caption(" · ".join(f"{k}={v}" for k, v in scale.items()))
defaults = load_default_questions()
custom = load_custom_questions()
all_q = defaults + custom
table = pd.DataFrame([{"사용": True, "번호": q.id, "영역": q.category, "문항": q.text,
                       "역문항": q.reverse} for q in all_q])
edited = st.data_editor(table, hide_index=True, disabled=["번호", "영역", "문항", "역문항"],
                        width="stretch")
questions = [q for q, use in zip(all_q, edited["사용"]) if use]

with st.expander("문항 추가하기"):
    with st.form("add_question", clear_on_submit=True):
        text = st.text_input("문항 내용")
        cats = sorted({q.category for q in all_q} | {"사용자 추가"})
        category = st.selectbox("영역", cats + ["(새 영역)"])
        new_cat = st.text_input("새 영역 이름(위에서 '새 영역'을 고른 경우)")
        reverse = st.checkbox("역문항(점수가 높을수록 영역 개념과 반대)")
        if st.form_submit_button("추가") and text.strip():
            cat = new_cat.strip() if category == "(새 영역)" and new_cat.strip() else category
            custom.append(Question(next_custom_id(all_q), text.strip(), cat, reverse))
            save_custom_questions(custom)
            st.rerun()
    if custom:
        remove = st.multiselect("삭제할 추가 문항", [q.id for q in custom])
        if st.button("선택 문항 삭제") and remove:
            save_custom_questions([q for q in custom if q.id not in remove])
            st.rerun()

# ---------------- 4. 실행 ----------------
st.header("4. 가상 설문 실행")
if sample is None:
    st.warning("먼저 2단계에서 표본을 추출하세요.")
    st.stop()
if not questions:
    st.warning("문항을 하나 이상 고르세요.")
    st.stop()
st.write(f"{len(sample):,}명 × {len(questions)}문항, 응답자당 요청 1회")
if st.button("설문 시작", type="primary", disabled=(mode == "llm" and not has_key)):
    bar = st.progress(0.0, text="응답 생성 중")
    responses = run_survey(sample, questions, mode=mode, model=model, effort=effort, workers=workers,
                           api_key=api_key or None,
                           progress=lambda d, t: bar.progress(d / t, text=f"응답 생성 중 {d}/{t}"))
    st.session_state.responses = responses
    st.session_state.meta = pd.DataFrame({
        "항목": ["실행 시각", "응답 방식", "모델", "effort", "대상 조건", "모집단 수", "표본 수", "문항"],
        "값": [datetime.now().isoformat(timespec="seconds"), mode, model if mode == "llm" else "-",
              effort if mode == "llm" else "-", flt.describe(), f"{len(population):,}", len(sample),
              ", ".join(q.id for q in questions)],
    })

responses = st.session_state.get("responses")
if responses is None:
    st.stop()

# ---------------- 5. 결과 ----------------
st.header("5. 통계 결과")
errors = responses[responses["error"].fillna("") != ""]
if len(errors):
    causes = errors.loc[~errors["error"].str.startswith("중단됨"), "error"]
    first = causes.iloc[0] if len(causes) else errors["error"].iloc[0]
    st.warning(f"{len(errors)}명은 응답 생성에 실패해 분석에서 뺐습니다. 원인: {first}")
valid = responses[responses["error"].fillna("") == ""]
q_ids = [q.id for q in questions if q.id in valid]
questions = [q for q in questions if q.id in q_ids]
if valid.empty:
    st.error("유효한 응답이 없습니다.")
    st.stop()
if mode == "demo":
    st.info("데모 모드 결과입니다. 수치는 파이프라인 점검용이며 의미가 없습니다.")

items = analysis.item_summary(valid, questions)
st.subheader("문항별 응답 분포")
st.plotly_chart(likert_chart(items, scale), width="stretch", config={"displayModeBar": False})
st.subheader("문항별 기술통계")
st.dataframe(items, hide_index=True, width="stretch")

st.subheader("영역별 점수와 신뢰도")
scale_table, scores = analysis.scale_summary(valid, questions)
st.dataframe(scale_table, hide_index=True, width="stretch")
st.caption("역문항은 6-점수로 바꿔 영역 점수를 냈습니다. Cronbach α 0.7 이상이면 문항들이 같은 개념을 "
           "일관되게 측정한다고 봅니다.")

st.subheader("집단 비교")
compare_by = st.multiselect("비교할 변수", COMPARE_CHOICES, default=["sex", "age_group"],
                            format_func=lambda c: COLUMN_LABELS.get(c, c))
for by in compare_by:
    st.markdown(f"**{COLUMN_LABELS.get(by, by)}별 영역 점수**")
    st.dataframe(analysis.group_compare(valid, scores, by), hide_index=True, width="stretch")
    with st.expander(f"{COLUMN_LABELS.get(by, by)}별 문항 점수"):
        st.dataframe(analysis.group_compare(valid, valid[q_ids], by), hide_index=True,
                     width="stretch")
st.caption(f"응답자가 {analysis.MIN_GROUP_N}명 미만인 집단은 검정에서 뺐습니다.")

if "comment" in valid:
    with st.expander("응답 이유 예시"):
        st.dataframe(valid[["sex", "age", "province", "education_level", "occupation", "comment"]].head(30),
                     hide_index=True, width="stretch")

sheets = analysis.build_report(responses, questions, compare_by, st.session_state.get("meta"))
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
d1, d2 = st.columns(2)
d1.download_button("엑셀 보고서 내려받기", analysis.to_excel(sheets), f"survey_{stamp}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d2.download_button("원자료 CSV 내려받기", responses.to_csv(index=False).encode("utf-8-sig"),
                   f"survey_raw_{stamp}.csv", mime="text/csv")
