"""검색 근거를 바탕으로 답변을 생성하는 Answer Agent입니다."""

from __future__ import annotations

import os
from typing import Any

from src.retrievers.answer import (
    CitedAnswer,
    Evidence,
    render_evidence_block,
)
from src.retrievers.hybrid_retriever import build_context
from src.retrievers.search_result import SearchResult

ANSWER_MODEL = "gpt-5.6-luna"


class AnswerAgent:
    """하이브리드 검색 컨텍스트를 근거 기반 답변으로 변환합니다."""

    def __init__(self, model: Any | None = None):
        self.model = model

    @classmethod
    def from_env(cls) -> "AnswerAgent":
        """환경 설정이 있으면 LangChain ChatOpenAI를 만들고 없으면 fallback합니다."""
        if not os.getenv("OPENAI_API_KEY"):
            return cls()
        try:
            from langchain_openai import ChatOpenAI

            return cls(
                ChatOpenAI(
                    model=ANSWER_MODEL,
                    temperature=0,
                )
            )
        except Exception:
            return cls()

    def generate(self, question: str, results: list[SearchResult]) -> str:
        """검색 근거를 사용해 답변을 생성하고, 답변에 출처 정보를 포함합니다."""
        if not results:
            return "충분한 검색 근거를 확인하지 못해 답변을 보류합니다."
        context = build_context(results)
        if self.model is None:
            return self._fallback_answer(context)

        prompt = (
            "당신은 농업 지식 기반 QA 답변 생성기입니다.\n"
            "아래 검색 근거에서 확인되는 내용만 사용해 한국어로 답변하세요.\n"
            "근거에 없는 내용은 추측하지 말고 확인 불가라고 표시하세요.\n"
            "농약·비료 수치는 공식 근거와 함께 제시하고, 근거가 부족하면 사용을 권하지 말고 확인이 필요하다고 안내하세요.\n"
            "답변 마지막에 사용한 evidence_ids, source_ids, chunk_ids와 그래프 경로를 출처로 표시하세요.\n\n"
            f"질문: {question}\n\n검색 근거:\n{context}"
        )
        response = self.model.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content)

    def generate_cited(self, question: str, evidences: list[Evidence]) -> CitedAnswer:
        """Evidence를 이용해 주장별 인용이 연결된 답변을 생성합니다."""

        if not evidences:
            return CitedAnswer(
                answer="답변에 사용할 인용 가능한 근거가 없습니다.",
                claims=[],
            )

        if self.model is None:
            return CitedAnswer(
                answer="답변 생성 모델을 사용할 수 없어 답변을 보류합니다.",
                claims=[],
            )

        evidence_block = render_evidence_block(evidences)

        prompt = (
            "당신은 농업 지식 기반 QA 답변 생성기입니다.\n"
            "아래 검색 근거에서 확인되는 내용만 사용해 한국어로 답변하세요.\n\n"

            "다음 규칙을 반드시 지키세요.\n"
            "1. 주요 사실을 claims의 개별 Claim으로 분리하세요.\n"
            "2. 모든 주요 주장에는 하나 이상의 Citation을 연결하세요.\n"
            "3. Citation.evidence_id는 제공된 evidence_id만 사용하세요.\n"
            "4. Citation.quote는 해당 근거의 원문에서 연속된 구절을 "
            "그대로 복사하세요.\n"
            "5. quote를 요약하거나 표현을 수정하지 마세요.\n"
            "6. 검색 근거에 없는 사실은 추측하지 마세요.\n"
            "7. 농약·비료의 수치와 사용 방법은 해당 내용을 직접 포함한 "
            "근거에 연결하세요.\n"
            "8. 답변에 사용하지 않은 근거는 인용하지 마세요.\n\n"

            f"질문:\n{question}\n\n"
            f"검색 근거:\n{evidence_block}"
        )

        structured_model = self.model.with_structured_output(CitedAnswer)
        result = structured_model.invoke(prompt)

        if isinstance(result, CitedAnswer):
            return result

        return CitedAnswer.model_validate(result)

    @staticmethod
    def _fallback_answer(context: str) -> str:
        """LLM을 사용할 수 없을 때 근거 컨텍스트를 안전하게 반환합니다."""
        return f"LLM을 사용할 수 없어 검색 근거를 그대로 표시합니다.\n\n{context}"
