"""보고서에 남길 재현 정보: 코드북 파일 지문과 앱(코드) 버전."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def file_fingerprint(path: Path) -> str:
    """파일 내용의 짧은 지문. 내용이 한 글자라도 다르면 바뀐다."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]


def app_version() -> str:
    """git 커밋과 날짜. git이 없는 환경이면 '알 수 없음'."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=short"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "알 수 없음"
    except Exception:
        return "알 수 없음"
