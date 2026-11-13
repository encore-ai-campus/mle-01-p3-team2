"""검색 근거를 바탕으로 답변을 생성하는 Answer Agent입니다."""

from __future__ import annotations

import os
from typing import Any

from retrievers.hybrid_retriever import build_context
from search_result import SearchResult

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

    @staticmethod
    def _fallback_answer(context: str) -> str:
        """LLM을 사용할 수 없을 때 근거 컨텍스트를 안전하게 반환합니다."""
        return f"LLM을 사용할 수 없어 검색 근거를 그대로 표시합니다.\n\n{context}"
