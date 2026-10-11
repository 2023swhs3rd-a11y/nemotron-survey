"""페르소나별 가상 응답 생성.

- llm 모드: Claude가 각 페르소나가 되어 리커트 5점 척도로 응답한다(ANTHROPIC_API_KEY 필요).
- demo 모드: API 없이 화면과 통계 흐름을 확인하는 규칙 기반 임의 응답. 분석 결과로 쓰면 안 된다.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd

from .questions import Question

DEFAULT_MODEL = "claude-opus-5-5"
MODEL_CHOICES = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
# 서버측 거절 대체(fallback)를 지원하는 모델
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}

SYSTEM_PROMPT = """당신은 설문조사 응답 시뮬레이션에서 한 명의 한국인 응답자를 연기합니다.
주어진 인구통계 정보와 페르소나 설명에 나타난 사람의 삶의 경험, 학력, 직업, 가족 상황, 가치관을 바탕으로
그 사람이 실제로 답했을 법한 응답을 고르세요.

원칙:
- 연구자나 AI의 관점이 아니라 그 사람 본인의 관점에서 답합니다.
- 사회적으로 바람직해 보이는 답이나 중간값(3)으로 몰지 말고, 그 사람의 배경에 비추어 솔직하게 고릅니다.
- 문항마다 독립적으로 판단하되, 한 사람의 일관된 가치관이 드러나게 합니다.
- 척도: 1=전혀 그렇지 않다, 2=그렇지 않다, 3=보통이다, 4=그렇다, 5=매우 그렇다."""


def persona_profile(row: pd.Series) -> str:
    fields = [
        ("성별", "sex"), ("나이", "age"), ("거주지", None), ("혼인 상태", "marital_status"),
        ("가구 유형", "family_type"), ("주거 형태", "housing_type"), ("학력", "education_level"),
        ("전공 계열", "bachelors_field"), ("직업", "occupation"),
    ]
    lines = []
    for label, col in fields:
        if col is None:
            lines.append(f"- 거주지: {row.get('province', '')} {row.get('district', '')}".rstrip())
        elif pd.notna(row.get(col)) and str(row.get(col)).strip():
            lines.append(f"- {label}: {row[col]}")
    texts = [
        ("요약", "persona"), ("직업 생활", "professional_persona"), ("가족", "family_persona"),
        ("성장 배경", "cultural_background"), ("커리어 목표", "career_goals_and_ambitions"),
    ]
    for label, col in texts:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            lines.append(f"- {label}: {row[col]}")
    return "\n".join(lines)


def _schema(questions: list[Question]) -> dict:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "object",
                "properties": {q.id: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for q in questions},
                "required": [q.id for q in questions],
                "additionalProperties": False,
            },
            "comment": {"type": "string", "description": "응답 이유를 한두 문장으로"},
        },
        "required": ["answers", "comment"],
        "additionalProperties": False,
    }


def _user_prompt(row: pd.Series, questions: list[Question]) -> str:
    items = "\n".join(f"{q.id}. {q.text}" for q in questions)
    return (
        f"[응답자 정보]\n{persona_profile(row)}\n\n"
        f"[설문: 대학교육과 진로교육에 대한 인식]\n{items}\n\n"
        "이 사람이 되어 각 문항에 1~5로 답하고, 전반적인 응답 이유를 comment에 짧게 적어 주세요."
    )


def ask_claude(client, row: pd.Series, questions: list[Question], model: str,
               effort: str = "low") -> dict:
    kwargs = dict(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(row, questions)}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": _schema(questions)}},
    )
    if model in FALLBACK_MODELS:
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    response = client.beta.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        raise RuntimeError("모델이 응답을 거절했습니다.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def demo_answers(row: pd.Series, questions: list[Question]) -> dict:
    """API 없이 쓰는 규칙 기반 임의 응답(파이프라인 점검용)."""
    seed = int(hashlib.md5(str(row["uuid"]).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    edu = str(row.get("education_level", ""))
    high_edu = any(k in edu for k in ["대학", "학사", "석사", "박사"]) and "중퇴" not in edu
    age = int(row.get("age", 40))
    answers = {}
    for q in questions:
        base = 3.0
        if q.category == "대학교육 가치":
            base += 0.5 if high_edu else -0.3
            base += 0.3 if age >= 60 else 0.0
            if q.reverse:
                base = 6 - base
        elif q.category == "진로교육 효과":
            base -= 0.3 if age < 35 else 0.0
        elif q.category == "교육 개선 요구":
            base += 0.6
        answers[q.id] = int(np.clip(np.rint(base + rng.normal(0, 0.9)), 1, 5))
    return {"answers": answers, "comment": "(데모 모드: 규칙 기반 임의 응답)"}


def run_survey(sample: pd.DataFrame, questions: list[Question], mode: str = "llm",
               model: str = DEFAULT_MODEL, effort: str = "low", workers: int = 8,
               progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """표본 전체에 설문을 돌려 응답자×문항 표(넓은 형태)를 돌려준다."""
    client = None
    if mode == "llm":
        import anthropic
        client = anthropic.Anthropic(max_retries=4)

    def one(idx: int) -> dict:
        row = sample.iloc[idx]
        try:
            result = (ask_claude(client, row, questions, model, effort) if mode == "llm"
                      else demo_answers(row, questions))
            out = {"uuid": row["uuid"], **result["answers"], "comment": result["comment"], "error": ""}
        except Exception as e:  # 한 명 실패가 전체를 멈추지 않도록 기록만 남긴다
            out = {"uuid": row["uuid"], "comment": "", "error": f"{type(e).__name__}: {e}"}
        return out

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, i) for i in range(len(sample))]
        for f in as_completed(futures):
            rows.append(f.result())
            done += 1
            if progress:
                progress(done, len(sample))

    responses = pd.DataFrame(rows)
    for q in questions:
        if q.id not in responses:
            responses[q.id] = np.nan
    demo_cols = [c for c in sample.columns if c not in responses.columns or c == "uuid"]
    merged = sample[demo_cols].merge(responses, on="uuid", how="left")
    return merged
