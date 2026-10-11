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
    personas = load_personas(s["uuid"].head(5).tolist(), data_root)
    assert len(personas) == 5 and "persona" in personas


def test_demo_run_and_report(data_root):
    demo = load_demographics(data_root)
    s = stratified_sample(demo, 300)
    s = s.merge(load_personas(s["uuid"].tolist(), data_root), on="uuid", how="left")
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
