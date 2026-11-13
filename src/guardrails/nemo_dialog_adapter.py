
"""NeMo 입력 안전성 검사와 LangGraph 연결."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "nemo"


# 명확한 보안 검사 우회 명령 탐지
BYPASS_PATTERN = re.compile(
    r"(?:정책|보안|안전|가드레일|guardrails)"
    r".{0,20}?"
    r"(?:검사|규칙|필터)"
    r".{0,20}?"
    r"(?:건너뛰|우회|무시|꺼|비활성화)",
    re.IGNORECASE,
)

# 교육·설명·부정 목적의 질문은 단순 패턴만으로 차단하지 않음
EXPLANATION_PATTERN = re.compile(
    r"(?:어떻게|방법|이유|설명|예방|방지|"
    r"건너뛰지|우회하지|무시하지|"
    r"해서는\s*안|하면\s*안)",
    re.IGNORECASE,
)


def _is_explicit_bypass(question: str) -> bool:
    """명확한 보안 정책 우회 지시인지 확인한다."""

    match = BYPASS_PATTERN.search(question)

    if match is None:
        return False

    # 질문 전체에 설명 또는 부정 목적이 있으면
    # 여기서 차단하지 않고 NeMo의 맥락 판단에 맡긴다.
    if EXPLANATION_PATTERN.search(question):
        return False

    return True


@lru_cache(maxsize=1)
def _get_nemo_rails():
    """NeMo 설정과 모델 객체를 한 번만 생성한다."""

    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(str(CONFIG_DIR))
    return LLMRails(config)


def check_nemo_dialog(question: str) -> tuple[bool, str]:
    """규칙 기반 검사와 NeMo 입력 정책을 함께 실행한다."""

    # 1. 비어 있는 질문 차단
    if not isinstance(question, str) or not question.strip():
        return False, "검사할 사용자 질문이 없습니다."

    # 2. 명확한 보안 정책 우회 명령 우선 차단
    if _is_explicit_bypass(question):
        return False, "보안 정책 우회 명령이 탐지됐습니다."

    # 3. 나머지 질문은 NeMo로 검사
    try:
        rails = _get_nemo_rails()

        result = rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            options={"rails": ["input"]},
        )

        response = getattr(result, "response", result)

        if isinstance(response, list):
            if len(response) != 1:
                return False, "NeMo 응답 형식 오류"
            response = response[0]

        if not isinstance(response, dict):
            return False, "NeMo 응답 형식 오류"

        content = response.get("content")

        if content == question:
            return True, "NeMo 입력 정책 검사 통과"

        return False, "NeMo 입력 정책에서 요청을 차단하거나 변경했습니다."

    except Exception:
        return False, "NeMo 입력 정책 검사를 완료하지 못했습니다."
