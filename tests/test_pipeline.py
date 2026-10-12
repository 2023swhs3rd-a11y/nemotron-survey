import numpy as np
import pandas as pd
import pytest

from survey import analysis
from survey.data import PopulationFilter, load_demographics, load_personas
from survey.questions import load_default_questions
from survey.sampling import cochran_n, plan_sample_size, stratified_sample, two_group_n
from survey.simulate import _schema, _user_prompt, run_survey
from tests.make_mock_data import make


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make(4000).to_parquet(root / "train.parquet", index=False)
    return root


def test_sample_size_formulas():
    assert cochran_n(1_000_000) == 384
    assert cochran_n(1_000) == 278          # 유한모집단 보정
    assert cochran_n(100) == 80
    assert two_group_n(0.3) == 175
    plan = plan_sample_size(1_000_000, groups=3, effect_size=0.3)
    assert plan.n_total == 525


def test_filter_and_stratified_sample(data_root):
    demo = load_demographics(data_root)
    pop = PopulationFilter(age_min=19, age_max=39, province=["서울", "경기"]).apply(demo)
    assert pop["age"].between(19, 39).all() and set(pop["province"]) <= {"서울", "경기"}
    s = stratified_sample(pop, 200, ["sex", "age_group", "province"])
    assert len(s) == 200 and s["uuid"].is_unique
    share_pop = pop["province"].value_counts(normalize=True)
    share_s = s["province"].value_counts(normalize=True)
    assert (share_pop - share_s).abs().max() < 0.02
    one = PopulationFilter(sex=["남자"], province=["서울"]).apply(demo)  # 층 기준이 하나만 남는 경우
    assert len(stratified_sample(one, 100)) == 100
    assert len(stratified_sample(pop, 150, ["province"])) == 150
    personas = load_personas(s.head(5), data_root)
    assert len(personas) == 5 and "persona" in personas


def test_demo_run_and_report(data_root):
    demo = load_demographics(data_root)
    s = stratified_sample(demo, 300)
    s = s.merge(load_personas(s, data_root), on="uuid", how="left")
    qs = load_default_questions()
    res = run_survey(s, qs, mode="demo", workers=4)
    assert len(res) == 300 and res[[q.id for q in qs]].isin([1, 2, 3, 4, 5]).all().all()
    items = analysis.item_summary(res, qs)
    assert np.allclose(items[[f"{k}점(%)" for k in range(1, 6)]].sum(axis=1), 100, atol=0.05)
    sheets = analysis.build_report(res, qs, ["sex", "age_group"])
    assert {"문항 통계", "영역 통계", "비교_성별_영역"} <= set(sheets)
    assert sheets["비교_성별_영역"]["검정"].eq("Welch t").all()
    assert sheets["비교_연령대_영역"]["검정"].eq("일원분산분석").all()
    assert len(analysis.to_excel(sheets)) > 1000


def test_cronbach_alpha_perfect_items():
    x = pd.Series([1, 2, 3, 4, 5] * 4)
    assert analysis.cronbach_alpha(pd.DataFrame({"a": x, "b": x, "c": x})) == pytest.approx(1.0)


def test_llm_prompt_and_schema():
    qs = load_default_questions()
    schema = _schema(qs)
    assert schema["properties"]["answers"]["required"] == [q.id for q in qs]
    row = pd.Series({"uuid": "x", "sex": "여자", "age": 25, "province": "서울", "district": "마포구",
                     "persona": "대학원생", "education_level": "4년제 대학교"})
    prompt = _user_prompt(row, qs)
    assert "Q10" in prompt and "서울 마포구" in prompt


def test_ask_claude_request_shape():
    import json
    from types import SimpleNamespace
    from survey.simulate import ask_claude

    qs = load_default_questions()
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            body = {"answers": {q.id: 4 for q in qs}, "comment": "테스트"}
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=json.dumps(body))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    row = pd.Series({"uuid": "x", "sex": "남자", "age": 50, "province": "부산", "district": "사하구"})
    out = ask_claude(client, row, qs, "claude-opus-5-5")
    assert out["answers"]["Q01"] == 4
    assert seen["fallbacks"] == "default" and seen["output_config"]["format"]["type"] == "json_schema"
    ask_claude(client, row, qs, "claude-haiku-5-5")
    assert "fallbacks" not in seen or seen["model"] == "claude-haiku-5-5"


def test_remote_loaders(monkeypatch, data_root):
    """원격 경로: 행 번호로 페르소나를 가져오고 uuid가 어긋난 행은 버린다."""
    from survey import data as d

    demo = load_demographics(data_root)
    full = pd.read_parquet(data_root / "train.parquet")
    monkeypatch.setattr(d, "local_dir", lambda: None)
    monkeypatch.setattr(d, "_fetch_row", lambda i: {c: full.iloc[i][c] for c in d.PERSONA_COLUMNS})
    by_uuid = full.set_index("uuid", drop=False)

    def fake_filter(uuids):  # 첫 묶음은 실패시켜 행 번호 경로도 함께 확인
        if uuids[0] == first:
            raise OSError("boom")
        return [{c: by_uuid.loc[u][c] for c in d.PERSONA_COLUMNS} for u in uuids]

    monkeypatch.setattr(d, "_fetch_by_uuid", fake_filter)
    s = stratified_sample(demo, 120)
    first = s["uuid"].iloc[0]
    got = d.load_personas(s)
    assert len(got) == 120 and set(got["uuid"]) == set(s["uuid"])
    monkeypatch.setattr(d, "_get_json", lambda url: {"parquet_files": [
        {"config": "default", "split": "train", "filename": "0001.parquet"},
        {"config": "default", "split": "train", "filename": "0000.parquet"}]})
    paths = d.remote_parquet_paths()
    assert paths[0].endswith("@refs%2Fconvert%2Fparquet/default/train/0000.parquet")


def test_codebook_scoring_and_estimation():
    from survey.codebook import estimate_responses, load_codebook, score_text, weight_table

    cb = load_codebook()
    assert score_text("평범한 하루", cb["cues"]["academic"]["keywords"])[0] == 0
    assert score_text("대학 시절", cb["cues"]["academic"]["keywords"])[0] == 1
    assert score_text("대학에서 전공 공부", cb["cues"]["academic"]["keywords"])[0] == 2
    assert score_text("배우자와 산다", cb["cues"]["lifelong"]["keywords"])[0] == 0  # '배우자'는 평생학습 아님

    qs = load_default_questions()
    people = pd.DataFrame({
        "uuid": ["a", "b"], "age": [45, 45], "education_level": ["대학원", "고등학교"],
        "persona": ["대학에서 전공 공부와 연구를 했다.", "현장에서 손재주로 장사를 해 왔다."],
    })
    r = estimate_responses(people, qs, cb, noise_sd=0.0)
    assert r.loc[0, "Q01"] > r.loc[1, "Q01"]            # 학업 단서·고학력 → 대학 필요성 더 높게
    assert r.loc[0, "Q04"] < r.loc[1, "Q04"]            # 실무 단서 → 조기 취업 선호 더 높게
    assert "학업 관여" in r.loc[0, "comment"]
    again = estimate_responses(people, qs, cb)
    assert again[[q.id for q in qs]].equals(estimate_responses(people, qs, cb)[[q.id for q in qs]])  # 재현성
    assert weight_table(qs, cb).loc["Q01", "intercept"] == 3.0


def test_province_and_education_order():
    from survey.data import _compact, filter_options

    df = pd.DataFrame({
        "uuid": list("abcdef"), "age": [20, 30, 40, 50, 60, 70],
        "province": ["경상북", "서울특별시", "전라남", "경기", "전북특별자치도", "서울"],
        "district": list("uvwxyz"),
        "education_level": ["대학원", "고등학교", "무학", "4년제 대학교", "2~3년제 전문대학", "초등학교"],
        "sex": "남자", "marital_status": "미혼", "housing_type": "아파트", "family_type": "1인 가구",
        "bachelors_field": "",
    })
    out = _compact(df)
    opts, _ = filter_options(out)
    assert opts["province"] == ["서울", "경기", "전북", "전남", "경북"]
    assert opts["education_level"] == ["무학", "초등학교", "고등학교", "2~3년제 전문대학", "4년제 대학교", "대학원"]
    s = pd.concat([out] * 5, ignore_index=True)  # 집단마다 5명 이상이어야 비교표에 들어간다
    s = s.assign(score=range(len(s)))
    table = analysis.group_compare(s, s[["score"]], "education_level")
    assert [c.split(" (")[0] for c in table.columns[1:3]] == ["무학", "초등학교"]


def test_p_value_display_and_effect_direction():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["가"] * 200 + ["나"] * 200,
                       "y": np.r_[rng.normal(4, 0.5, 200), rng.normal(2, 0.5, 200)]})
    table = analysis.group_compare(df, df[["y"]], "g")
    assert 0 < table.loc[0, "p값"] < 1e-10  # 반올림으로 0이 되지 않는다
    assert table.loc[0, "효과크기 종류"] == "Cohen d (가 − 나)" and table.loc[0, "효과크기"] > 0
    shown = analysis.p_display(table)
    assert shown.loc[0, "p값"] == "<.001"
    assert analysis.p_text(0.0412) == "0.041" and analysis.p_text(float("nan")) == ""
