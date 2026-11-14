from __future__ import annotations

import pytest

from src.streamlit.chatbot import generate_answer


class FakeSupervisor:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.result = result or {}
        self.inputs: list[dict[str, str]] = []

    def invoke(self, input_data: dict[str, str]) -> dict[str, object]:
        self.inputs.append(input_data)
        return self.result


def test_generate_answer_passes_question_and_returns_final_answer() -> None:
    supervisor = FakeSupervisor({"final_answer": "  마늘 재배 답변입니다.  "})

    answer = generate_answer(supervisor, "마늘 파종 시기는 언제인가요?")

    assert answer == "마늘 재배 답변입니다."
    assert supervisor.inputs == [{"question": "마늘 파종 시기는 언제인가요?"}]


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"final_answer": ""},
        {"final_answer": "   "},
        {"final_answer": None},
        {"final_answer": 123},
    ],
)
def test_generate_answer_uses_fallback_for_invalid_final_answer(
    result: dict[str, object],
) -> None:
    supervisor = FakeSupervisor(result)

    answer = generate_answer(supervisor, "질문")

    assert answer == "답변을 생성하지 못했습니다. 다시 질문해 주세요."


def test_generate_answer_converts_supervisor_exception_to_user_message() -> None:
    class BrokenSupervisor:
        def invoke(self, input_data: dict[str, str]) -> dict[str, object]:
            raise RuntimeError("Neo4j 연결 실패")

    answer = generate_answer(BrokenSupervisor(), "질문")

    assert answer == "답변 생성 중 오류가 발생했습니다: Neo4j 연결 실패"
