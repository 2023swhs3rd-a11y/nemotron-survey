"""설문 문항 관리: 추천 문항 10개를 기본으로 하고, 사용자 문항을 추가할 수 있다."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("questions.json")
CUSTOM_PATH = Path(__file__).resolve().parent.parent / "custom_questions.json"


@dataclass
class Question:
    id: str
    text: str
    category: str = "사용자 추가"
    reverse: bool = False


def load_scale(path: Path = DEFAULT_PATH) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["scale"].items()}


def load_default_questions(path: Path = DEFAULT_PATH) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**q) for q in data["questions"]]


def load_custom_questions(path: Path = CUSTOM_PATH) -> list[Question]:
    if not path.exists():
        return []
    return [Question(**q) for q in json.loads(path.read_text(encoding="utf-8"))]


def save_custom_questions(questions: list[Question], path: Path = CUSTOM_PATH) -> None:
    path.write_text(
        json.dumps([asdict(q) for q in questions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_all_questions() -> list[Question]:
    return load_default_questions() + load_custom_questions()


def next_custom_id(existing: list[Question]) -> str:
    used = {q.id for q in existing}
    i = 1
    while f"U{i:02d}" in used:
        i += 1
    return f"U{i:02d}"
