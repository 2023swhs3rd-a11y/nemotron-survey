"""대학·진로교육 인식 가상 설문 (Nemotron-Personas-Korea 기반) - Streamlit 앱.

실행: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survey import analysis
from survey.codebook import (cue_friedman, cue_labels, cue_summary, estimate_responses,
                             load_codebook, weight_table)
from survey.data import (COLUMN_LABELS, PopulationFilter, filter_options, load_demographics,
                         load_personas)
from survey.questions import (Question, load_custom_questions, load_default_questions,
                              load_scale, next_custom_id)
from survey.sampling import compare_distribution, plan_sample_size, stratified_sample
from survey.simulate import DEFAULT_MODEL, MODEL_CHOICES, run_survey

st.set_page_config(page_title="대학·진로교육 인식 가상 설문", layout="wide")
# 리커트 1~5점 색: 부정(빨강) - 중립(회색) - 긍정(파랑)
LIKERT_COLORS = ["#c7383a", "#ec8a84", "#b5b4ae", "#6da7ec", "#1c5cab"]
COMPARE_CHOICES = ["sex", "age_group", "province", "education_level", "marital_status",
                   "housing_type", "family_type", "bachelors_field"]


# cache_resource: 100만 행 표를 매번 복사하지 않고 한 번 읽은 것을 모든 접속자가 함께 쓴다(읽기 전용).
@st.cache_resource(show_spinner="페르소나 100만 건의 인구통계 정보를 불러오는 중...")
def get_demographics() -> pd.DataFrame:
    return load_demographics()


@st.cache_resource
def get_options() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    return filter_options(get_demographics())


PILLS_MAX = 20  # 선택지가 이보다 적으면 한눈에 보이는 버튼(pills), 많으면 검색되는 목록


def choose(col: str, choices: list[str]) -> list[str] | None:
    """기본은 전체. '전체'를 끄면 항목을 직접 고른다. 전체면 [], 끄고 아무것도 안 고르면 None."""
    label = COLUMN_LABELS[col]
    if st.checkbox(f"{label}: 전체", value=True, key=f"all_{col}"):
        return []
    fmt = lambda v: v or "(해당 없음)"
    if len(choices) <= PILLS_MAX:
        picked = st.pills(label, choices, selection_mode="multi", format_func=fmt, key=f"pick_{col}",
                          label_visibility="collapsed")
    else:
        picked = st.multiselect(label, choices, format_func=fmt, key=f"pick_{col}",
                                label_visibility="collapsed", placeholder="검색해서 고르세요")
    if not picked:
        st.caption(f":red[{label}을(를) 하나 이상 고르세요.]")
        return None
    return list(picked)


def questions_json(qs: list[Question]) -> bytes:
    return json.dumps([asdict(q) for q in qs], ensure_ascii=False, indent=2).encode("utf-8")


@st.cache_data(max_entries=8, show_spinner="통계표와 보고서를 만드는 중...")
def make_report(run_id: str, q_ids: tuple[str, ...], compare_by: tuple[str, ...], _responses: pd.DataFrame,
                _questions: list[Question], _meta: pd.DataFrame | None,
                _rules: pd.DataFrame | None) -> tuple[dict[str, pd.DataFrame], bytes]:
    """결과 시트와 엑셀 파일. 같은 실행·같은 문항·같은 비교 변수면 다시 계산하지 않는다
    (밑줄로 시작하는 인자는 캐시 열쇠에서 빠지고 run_id 등이 대신한다)."""
    sheets = analysis.build_report(_responses, _questions, list(compare_by), _meta)
    valid = _responses[_responses["error"].fillna("") == ""]
    cue_cols = [c for c in valid.columns if c.startswith("단서_")]
    if cue_cols:
        labels = cue_labels(load_codebook())
        cues = valid[cue_cols].rename(columns=lambda c: c.removeprefix("단서_"))
        sheets["단서 통계"] = cue_summary(cues, labels)
        named = cues.rename(columns=labels)
        for by in compare_by:
            sheets[f"단서비교_{COLUMN_LABELS.get(by, by)}"] = analysis.group_compare(valid, named, by)
    if _rules is not None:
        sheets["추정 규칙"] = _rules.reset_index()
    return sheets, analysis.to_excel(sheets)


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


REPO_URL = "https://github.com/2023swhs3rd-a11y/nemotron-survey"

st.title("대학·진로교육 인식 가상 설문")
st.caption("NVIDIA Nemotron-Personas-Korea 합성 페르소나 100만 명 중 원하는 인구집단을 골라 "
           "리커트 5점 척도 설문을 시뮬레이션하고 바로 통계를 냅니다.")

# ---------------- 사이드바: 응답 방식 ----------------
def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")) or os.environ.get(name, "")
    except Exception:  # secrets.toml이 없으면 예외가 난다
        return os.environ.get(name, "")


env_key = secret("ANTHROPIC_API_KEY")
claude_enabled = bool(env_key) or secret("ENABLE_CLAUDE") == "1"
MODES = {"codebook": "생활서사 코드북 추정", "demo": "빠른 점검(데모)"}
if claude_enabled:
    MODES["llm"] = "Claude가 페르소나로 응답"

with st.sidebar:
    st.header("응답 방식")
    mode = st.radio("응답을 어떻게 만들까요?", list(MODES), format_func=MODES.get)
    api_key, model, effort, workers = env_key, DEFAULT_MODEL, "low", 8
    if mode == "codebook":
        st.markdown(
            "**연구용 추정 방식(권장)**\n\n"
            "각 응답자의 페르소나 문장(직업·가족·성장 배경·커리어 목표 등)을 읽어 교육·진로 관련 단서 6가지"
            "(학업 관여, 커리어 성장, 평생학습, 실무·현장 지향, 자녀교육 관심, 경제적 부담)를 0–2점으로 매기고, "
            "단서 점수와 학력·연령에 문항별 가중치를 곱해 1–5점 응답을 추정합니다.\n\n"
            "- **근거**: 사람마다 다른 생활서사 텍스트. 그래서 같은 학력·나이라도 서사에 따라 응답이 달라집니다.\n"
            "- **투명성**: 단서 키워드와 가중치를 4단계에서 보고 고칠 수 있고, 엑셀 보고서에 함께 저장됩니다.\n"
            "- **재현성**: 같은 표본·같은 설정이면 언제나 같은 결과가 나옵니다.\n"
            "- **시간**: 표본 인원의 페르소나 문장을 받아야 해서 조금 걸립니다. 표본을 뽑는 즉시 뒤에서 미리 받기 시작합니다.\n"
            "- **해석**: 실제 의견의 예측이 아니라 '이런 차이가 있을 수 있다'는 가설을 세우는 용도입니다. "
            "(유승철·임승희, 2026의 생활서사 코드북 방법을 응용)")
    elif mode == "demo":
        st.markdown(
            "**화면·통계 흐름 점검용**\n\n"
            "페르소나 문장을 읽지 않고, 학력(대졸 이상 여부)·나이와 문항 영역에 따른 고정 규칙에 무작위 값을 더해 "
            "응답을 만듭니다. 텍스트를 받지 않으므로 표본이 커도 즉시 끝납니다.\n\n"
            "- **쓰임**: 문항 구성, 집단 비교 설정, 엑셀 보고서 모양을 미리 확인할 때.\n"
            "- **한계**: 규칙이 단순하고 고정되어 있어 같은 학력·나이면 응답 경향이 같고, 추가한 문항은 "
            "대부분 '보통(3)' 근처의 무작위 값이 됩니다. **결과 수치를 연구에 쓰지 마세요.**")
    else:
        typed = st.text_input("Claude API 키", type="password", placeholder="sk-ant-api03-...",
                              help="console.anthropic.com의 API Keys에서 발급합니다. "
                                   "이 브라우저 세션에서만 쓰이고 저장되지 않습니다.").strip()
        api_key = typed or env_key
        if not api_key:
            st.warning("Claude API 키를 입력하세요.")
        model = st.selectbox("모델", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
        effort = st.select_slider("추론 강도(effort)", ["low", "medium", "high"], value="low")
        workers = st.slider("동시 요청 수", 1, 32, 8)

    st.divider()
    st.markdown(
        "**이용 안내**\n"
        "- 합성 페르소나의 가상 응답이므로 실제 여론이 아닙니다. 문항 점검과 가설 탐색에 쓰세요.\n"
        "- 추가 문항, 고친 가중치, 결과는 이 브라우저 세션에만 남고 다른 사람과 공유되지 않습니다. "
        "필요한 것은 내려받아 두세요.\n"
        f"- 내 문항·코드북으로 계속 쓰려면 [저장소]({REPO_URL})를 Fork해 내 버전을 배포하세요. "
        f"방법은 [README]({REPO_URL}#내-버전-만들기)에 있습니다.")
    has_key = bool(api_key)

demo = get_demographics()
opts, districts_by_province = get_options()

# ---------------- 1. 인구집단 선택 ----------------
st.header("1. 조사 대상 인구집단")
st.caption("모든 항목의 기본값은 '전체'입니다. 범위를 좁히려면 '전체'를 끄고 포함할 항목을 고르세요.")
age = st.slider("나이", 19, 99, (19, 99))
c1, c2, c3 = st.columns(3)
with c1:
    sex = choose("sex", opts["sex"])
    marital = choose("marital_status", opts["marital_status"])
    housing = choose("housing_type", opts["housing_type"])
with c2:
    province = choose("province", opts["province"])
    district_choices = (sorted({d for p in province for d in districts_by_province.get(p, [])})
                        if province else opts["district"])
    district = choose("district", district_choices)
    family = choose("family_type", opts["family_type"])
with c3:
    education = choose("education_level", opts["education_level"])
    field = choose("bachelors_field", opts["bachelors_field"])
    occupation_kw = st.text_input("직업 키워드(쉼표로 여러 개, 비우면 전체)", placeholder="예: 교사, 교수, 강사")

picks = [sex, marital, housing, province, district, family, education, field]
if any(p is None for p in picks):
    st.warning("'전체'를 끈 항목은 하나 이상 골라야 합니다.")
    st.stop()
flt = PopulationFilter(age_min=age[0], age_max=age[1], sex=sex, province=province, district=district,
                       education_level=education, marital_status=marital, housing_type=housing,
                       family_type=family, bachelors_field=field, occupation_keyword=occupation_kw)
population = flt.apply(demo)
st.metric("조건에 맞는 페르소나 수", f"{len(population):,}명")
if population.empty:
    st.stop()

@st.cache_resource
def prefetch_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)  # 페르소나 문장 미리 받기용(접속자 공용)


def start_prefetch(sample: pd.DataFrame) -> dict:
    """페르소나 문장을 백그라운드 스레드에서 받는다. 진행 상황은 dict에 적어 화면이 읽어 간다."""
    job = {"done": 0, "total": 1}
    job["future"] = prefetch_pool().submit(
        load_personas, sample, progress=lambda d, t: job.update(done=d, total=t))
    return job


def with_personas(bar) -> pd.DataFrame:
    """표본에 페르소나 문장을 붙인다. 미리 받는 중이면 끝나기를 기다리고, 실패하면 인구통계만으로 진행."""
    sample = st.session_state.sample
    if "persona" in sample:
        return sample
    job = st.session_state.get("prefetch") or start_prefetch(sample)
    future = job["future"]
    while not future.done():
        if bar is not None:
            bar.progress(job["done"] / job["total"],
                         text=f"페르소나 문장 가져오는 중 {job['done']}/{job['total']} 묶음")
        time.sleep(0.3)
    try:
        merged = sample.merge(future.result(), on="uuid", how="left")
    except Exception as e:
        st.session_state.pop("prefetch", None)  # 다음 실행 때 다시 시도
        st.warning(f"페르소나 문장을 가져오지 못해 인구통계 정보만 씁니다({type(e).__name__}).")
        return sample
    st.session_state.sample = merged
    return merged


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
    st.session_state.sample = stratified_sample(population, int(n), strata, int(seed))
    st.session_state.pop("responses", None)
    # 표본 추출은 바로 끝내고, 페르소나 문장은 문항을 고르는 동안 뒤에서 미리 받아 둔다
    st.session_state.prefetch = start_prefetch(st.session_state.sample)

sample = st.session_state.get("sample")
if sample is not None:
    st.success(f"{len(sample):,}명을 뽑았습니다.")
    with st.expander("모집단과 표본 구성 비교"):
        cols = st.columns(3)
        for i, col in enumerate(["sex", "age_group", "province"]):
            cols[i].write(f"**{COLUMN_LABELS[col]}**")
            cols[i].dataframe(compare_distribution(population, sample, col))
    with st.expander("표본 미리 보기"):
        preview = sample.head(20)
        if "persona" not in preview and st.checkbox("페르소나 설명 문장도 보기"):
            with st.spinner("Hugging Face에서 페르소나 문장을 가져오는 중..."):
                preview = with_personas(None).head(20)
        st.dataframe(preview, hide_index=True)

# ---------------- 3. 문항 ----------------
st.header("3. 설문 문항 (리커트 5점)")
scale = load_scale()
st.caption(" · ".join(f"{k}={v}" for k, v in scale.items()))
defaults = load_default_questions()
if "custom_questions" not in st.session_state:
    st.session_state.custom_questions = load_custom_questions()
custom: list[Question] = st.session_state.custom_questions
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
            st.rerun()
    if custom:
        remove = st.multiselect("삭제할 추가 문항", [q.id for q in custom])
        if st.button("선택 문항 삭제") and remove:
            st.session_state.custom_questions = [q for q in custom if q.id not in remove]
            st.rerun()
    st.caption("추가한 문항은 이 브라우저 세션 동안 유지됩니다. 다음에도 쓰려면 파일로 내려받아 두었다가 불러오세요.")
    c_down, c_up = st.columns(2)
    c_down.download_button("추가 문항 내려받기(JSON)", questions_json(custom), "custom_questions.json",
                           mime="application/json", disabled=not custom)
    uploaded_q = c_up.file_uploader("추가 문항 불러오기", type=["json"], key="q_upload")
    if uploaded_q is not None and st.session_state.get("q_upload_done") != uploaded_q.file_id:
        loaded = [Question(**q) for q in json.loads(uploaded_q.getvalue().decode("utf-8"))]
        used = {q.id for q in defaults}
        st.session_state.custom_questions = [q for q in loaded if q.id not in used]
        st.session_state.q_upload_done = uploaded_q.file_id
        st.rerun()

# ---------------- 4. 실행 ----------------
st.header("4. 응답 만들기")
if sample is None:
    st.warning("먼저 2단계에서 표본을 추출하세요.")
    st.stop()
if not questions:
    st.warning("문항을 하나 이상 고르세요.")
    st.stop()
def save_meta(rules: pd.DataFrame | None = None) -> None:
    """엑셀 '설정' 시트에 실행 조건을, 코드북 방식이면 '추정 규칙' 시트에 쓴 가중치를 남긴다."""
    st.session_state.rules = rules
    st.session_state.run_id = datetime.now().isoformat()  # 결과 표·보고서 캐시의 열쇠
    st.session_state.meta = pd.DataFrame({
        "항목": ["실행 시각", "응답 방식", "모델", "effort", "대상 조건", "모집단 수", "표본 수", "표본 시드", "문항"],
        "값": [datetime.now().isoformat(timespec="seconds"), MODES[mode], model if mode == "llm" else "-",
              effort if mode == "llm" else "-", flt.describe(), f"{len(population):,}", len(sample), int(seed),
              ", ".join(q.id for q in questions)],
    })


if mode == "codebook":
    codebook = load_codebook()
    labels = cue_labels(codebook)
    with st.expander("코드북: 어떤 단서를 어떻게 찾나요?"):
        st.markdown("페르소나의 " + ", ".join(f"`{f}`" for f in codebook["text_fields"]) +
                    " 문장을 하나로 합친 뒤, 단서별 키워드를 찾습니다. **0점** 키워드 없음, "
                    "**1점** 키워드 1개(약한 단서), **2점** 서로 다른 키워드 2개 이상(반복·구체적 단서).")
        st.dataframe(pd.DataFrame([{"단서": v["label"], "정의": v["definition"], "키워드": ", ".join(v["keywords"])}
                                   for v in codebook["cues"].values()]), hide_index=True, width="stretch")
    with st.expander("추정 규칙: 문항별 가중치 (고칠 수 있음)"):
        st.markdown(
            "**응답 = 기준점 + Σ(가중치 × 단서 점수) + Σ(가중치 × 인구통계 항) + 무작위 오차**, 결과를 1–5로 반올림합니다.\n"
            "- 단서 점수는 0, 1, 2 중 하나입니다. 인구통계 항은 해당하면 1, 아니면 0입니다: "
            "`edu_high` 4년제 대학 이상, `young` 35세 미만, `old` 60세 이상.\n"
            "- 양수 가중치는 '그렇다' 쪽, 음수는 '그렇지 않다' 쪽으로 응답을 옮깁니다.\n"
            "- 예) Q01에서 학업 관여 2점, 4년제 대학 졸업자: 3 + 0.5×2 + 0.4 = 4.4 → 오차를 더해 대체로 4점.\n"
            "- 추가한 문항은 기준점 3, 가중치 0으로 시작하니 여기서 정해 주세요. 표의 칸을 눌러 바로 고칠 수 있습니다.")
        wt = weight_table(questions, codebook).rename(columns={"intercept": "기준점", **labels})
        wt_edit = st.data_editor(wt, width="stretch", key="weights_editor")
        noise_sd = st.slider("무작위 오차 크기(표준편차)", 0.0, 1.5, float(codebook.get("noise_sd", 0.7)), 0.1,
                             help="0이면 같은 단서를 가진 사람은 모두 같은 응답을 합니다.")
    st.write(f"{len(sample):,}명 × {len(questions)}문항")
    if st.button("응답 추정", type="primary"):
        bar = st.progress(0.0, text="페르소나 문장 가져오는 중")
        run_sample = with_personas(bar)
        weights = wt_edit.rename(columns={"기준점": "intercept", **{v: k for k, v in labels.items()}})
        st.session_state.responses = estimate_responses(run_sample, questions, codebook, weights, noise_sd)
        save_meta(wt_edit.assign(**{"오차 표준편차": noise_sd}))
        bar.empty()
else:
    unit = "응답자당 요청 1회" if mode == "llm" else "규칙 기반 임의 응답"
    st.write(f"{len(sample):,}명 × {len(questions)}문항, {unit}")
    if st.button("설문 시작", type="primary", disabled=(mode == "llm" and not has_key)):
        bar = st.progress(0.0, text="페르소나 문장 가져오는 중")
        run_sample = with_personas(bar) if mode == "llm" else sample
        bar.progress(0.0, text="응답 생성 중")
        st.session_state.responses = run_survey(
            run_sample, questions, mode=mode, model=model, effort=effort, workers=workers,
            api_key=api_key or None,
            progress=lambda d, t: bar.progress(d / t, text=f"응답 생성 중 {d}/{t}"))
        save_meta()

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
cue_cols = [c for c in valid.columns if c.startswith("단서_")]
if cue_cols:
    no_text = valid["comment"].astype(str).str.startswith("(페르소나 텍스트 없음").sum()
    st.info("코드북 추정 결과입니다. 페르소나 생활서사의 단서와 4단계 가중치로 만든 추정치이며, "
            "실제 응답이 아닙니다. 집단 차이는 '이런 차이가 있을 수 있다'는 가설로 읽고 실제 조사로 확인하세요."
            + (f" 페르소나 문장이 없는 {no_text}명은 인구통계만으로 추정했습니다." if no_text else ""))

compare_by = st.multiselect("집단 비교 변수(아래 집단 비교·단서 분석·엑셀 보고서에 적용)", COMPARE_CHOICES,
                            default=["sex", "age_group"], format_func=lambda c: COLUMN_LABELS.get(c, c))
sheets, excel = make_report(st.session_state.run_id, tuple(q_ids), tuple(compare_by), responses, questions,
                            st.session_state.get("meta"), st.session_state.get("rules"))

items = sheets["문항 통계"]
st.subheader("문항별 응답 분포")
st.plotly_chart(likert_chart(items, scale), width="stretch", config={"displayModeBar": False})
st.subheader("문항별 기술통계")
st.dataframe(items, hide_index=True, width="stretch")

st.subheader("영역별 점수와 신뢰도")
st.dataframe(sheets["영역 통계"], hide_index=True, width="stretch")
st.caption("역문항은 6-점수로 바꿔 영역 점수를 냈습니다. Cronbach α 0.7 이상이면 문항들이 같은 개념을 "
           "일관되게 측정한다고 봅니다.")

st.subheader("집단 비교")
for by in compare_by:
    label = COLUMN_LABELS.get(by, by)
    st.markdown(f"**{label}별 영역 점수**")
    st.dataframe(sheets[f"비교_{label}_영역"], hide_index=True, width="stretch")
    with st.expander(f"{label}별 문항 점수"):
        st.dataframe(sheets[f"비교_{label}_문항"], hide_index=True, width="stretch")
st.caption(f"응답자가 {analysis.MIN_GROUP_N}명 미만인 집단은 검정에서 뺐습니다.")

if cue_cols:
    st.subheader("생활서사 단서 분석")
    st.dataframe(sheets["단서 통계"], hide_index=True, width="stretch")
    fr = cue_friedman(valid[cue_cols].rename(columns=lambda c: c.removeprefix("단서_")))
    if fr:
        st.caption(f"단서 간 차이(Friedman 검정): χ²={fr['통계량']}, p={fr['p값']:.4f}. "
                   "같은 사람이 여러 단서 점수를 가지므로 반복측정 검정을 썼습니다.")
    for by in compare_by:
        label = COLUMN_LABELS.get(by, by)
        with st.expander(f"{label}별 단서 점수"):
            st.dataframe(sheets[f"단서비교_{label}"], hide_index=True, width="stretch")
    st.caption("표본이 크면 작은 차이도 유의하게 나오므로 p값보다 효과크기(d, η²)를 함께 보세요.")

if "comment" in valid:
    with st.expander("응답 근거 예시" if cue_cols else "응답 이유 예시"):
        st.dataframe(valid[["sex", "age", "province", "education_level", "occupation", "comment"]].head(30),
                     hide_index=True, width="stretch")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
d1, d2 = st.columns(2)
d1.download_button("엑셀 보고서 내려받기", excel, f"survey_{stamp}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d2.download_button("원자료 CSV 내려받기", responses.to_csv(index=False).encode("utf-8-sig"),
                   f"survey_raw_{stamp}.csv", mime="text/csv")
