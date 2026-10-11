"""실제 데이터와 같은 열 구조의 작은 가짜 데이터(테스트·오프라인 시연용)를 만든다.

사용: python tests/make_mock_data.py mock_data 5000
      NEMOTRON_DATA_DIR=mock_data streamlit run app.py
"""
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCES = {
    "서울": ["강남구", "마포구", "노원구"], "부산": ["해운대구", "사하구"], "경기": ["수원시", "성남시", "고양시"],
    "광주": ["서구", "북구"], "강원": ["춘천시", "강릉시"], "제주": ["제주시"],
}
EDUCATION = ["초등학교", "중학교", "고등학교", "2~3년제 대학", "4년제 대학교", "대학원(석사)", "대학원(박사)"]
FIELDS = ["인문", "사회", "교육", "공학", "자연", "의약", "예체능", ""]
JOBS = ["사무원", "교사", "간호사", "자영업자", "소프트웨어 개발자", "농업인", "판매원", "무직", "대학생", "공무원"]


def make(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prov = rng.choice(list(PROVINCES), n, p=[0.3, 0.12, 0.35, 0.08, 0.1, 0.05])
    edu = rng.choice(EDUCATION, n, p=[0.06, 0.08, 0.35, 0.15, 0.28, 0.06, 0.02])
    phrases = ["대학에서 전공 공부를 이어 가고 싶다.", "현장에서 손재주로 일해 왔다.", "자녀의 학원비와 입시가 걱정이다.",
               "생활비를 아끼며 알뜰하게 산다.", "문화센터 강좌에서 새로운 기술을 배우고 있다.",
               "승진과 자격증 취득으로 경력을 쌓고 싶다.", "평범한 일상을 보낸다."]
    pick = lambda: " ".join(rng.choice(phrases, 2, replace=False))
    return pd.DataFrame({
        "uuid": [uuid.UUID(int=int(rng.integers(0, 2**63))).hex for _ in range(n)],
        "professional_persona": [pick() for _ in range(n)],
        "sports_persona": "주말에 가볍게 산책한다.", "arts_persona": "가끔 영화를 본다.",
        "travel_persona": "국내 여행을 즐긴다.", "culinary_persona": "집밥을 좋아한다.",
        "family_persona": [pick() for _ in range(n)],
        "persona": "성실하고 현실적인 생활인.", "cultural_background": "지방 소도시에서 자랐다.",
        "skills_and_expertise": "문서 작성", "skills_and_expertise_list": "['문서 작성']",
        "hobbies_and_interests": "독서", "hobbies_and_interests_list": "['독서']",
        "career_goals_and_ambitions": "안정적인 노후를 준비하고 싶다.",
        "sex": rng.choice(["남자", "여자"], n),
        "age": rng.integers(19, 90, n),
        "marital_status": rng.choice(["미혼", "배우자있음", "사별", "이혼"], n, p=[0.35, 0.5, 0.08, 0.07]),
        "military_status": rng.choice(["비현역", "현역"], n, p=[0.97, 0.03]),
        "family_type": rng.choice(["1인 가구", "부부 가구", "부부와 미혼 자녀 가구", "한부모 가구"], n),
        "housing_type": rng.choice(["아파트", "단독주택", "다세대주택", "오피스텔"], n),
        "education_level": edu,
        "bachelors_field": rng.choice(FIELDS, n),
        "occupation": rng.choice(JOBS, n),
        "district": [rng.choice(PROVINCES[p]) for p in prov],
        "province": prov,
        "country": "대한민국",
    })


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "mock_data")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    out.mkdir(parents=True, exist_ok=True)
    make(n).to_parquet(out / "train-00000-of-00001.parquet", index=False)
    print(f"{out}/ 에 {n}행 생성")
