"""Streamlit UI와 Supervisor 사이의 최소 답변 생성 경계입니다."""

from __future__ import annotations

from typing import Any, Protocol


FALLBACK_ANSWER = "답변을 생성하지 못했습니다. 다시 질문해 주세요."


class Supervisor(Protocol):
    """챗봇이 사용하는 Supervisor의 최소 인터페이스입니다."""

    def invoke(self, input_data: dict[str, str]) -> dict[str, Any]: ...


def generate_answer(supervisor: Supervisor, question: str) -> str:
    """질문을 Supervisor에 전달하고 화면에 표시할 답변을 반환합니다."""

    try:
        result = supervisor.invoke({"question": question})
    except Exception as exc:
        return f"답변 생성 중 오류가 발생했습니다: {exc}"

    answer = result.get("final_answer") if isinstance(result, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        return FALLBACK_ANSWER

    return answer.strip()

