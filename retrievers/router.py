"""하이브리드 농업 GraphRAG 파이프라인의 질문 라우터입니다."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

try:
    from langchain_typesafe import Choice, TypeSafeClassifier
except ImportError:  # Optional until TYPESAFE_API_KEY is configured.
    Choice = None
    TypeSafeClassifier = None

Route = Literal["GRAPH", "VECTOR", "HYBRID", "REJECT"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    crop: str | None = None
    max_hops: int = 2
    warnings: list[str] = field(default_factory=list)


class QuestionRouter:
    """Vector DB 구현과 분리된 검색 경로 판단을 수행합니다."""

    GRAPH_TERMS = (
        "관계", "연결", "경로", "품종", "적합 지역", "파종 시기", "생육 단계",
        "병해", "병해충", "해충", "환경 조건", "멀티홉", "어떤 지역",
    )
    VECTOR_TERMS = (
        "방법", "뜻", "의미", "설명", "재배법", "관리법", "관수 방법",
        "시비 방법", "방제 방법", "예방", "무엇인가", "어떻게 재배",
    )
    OFFICIAL_NUMERIC_TERMS = (
        "농약", "비료", "희석", "배율", "사용량", "안전사용", "수확 전",
    )
    CROPS = ("상추", "양파", "마늘", "고추", "배추")

    def decide(self, question: str) -> RouteDecision:
        normalized = re.sub(r"\s+", " ", question.strip())
        if not normalized:
            return RouteDecision("REJECT", "질문이 비어 있습니다.", warnings=[])

        crop = next((crop for crop in self.CROPS if crop in normalized), None)
        has_graph = any(term in normalized for term in self.GRAPH_TERMS)
        has_vector = any(term in normalized for term in self.VECTOR_TERMS)
        requires_official = any(term in normalized for term in self.OFFICIAL_NUMERIC_TERMS)

        if requires_official and not has_graph and not has_vector:
            return RouteDecision(
                "GRAPH",
                "농약·비료 공식 근거를 그래프에서 먼저 확인합니다.",
                crop=crop,
                warnings=["최종 답변 전 공식 근거 검증이 필요합니다."],
            )
        if has_graph and has_vector:
            return RouteDecision("HYBRID", "관계 탐색과 원문 설명이 모두 필요합니다.", crop=crop)
        if has_graph:
            return RouteDecision("GRAPH", "품종·환경·병해충·지역 관계 검색이 적합합니다.", crop=crop)
        if has_vector:
            return RouteDecision("VECTOR", "설명·재배 방법 중심의 원문 검색이 적합합니다.", crop=crop)
        return RouteDecision(
            "HYBRID",
            "질문 유형이 불명확해 그래프와 원문 검색을 함께 시도합니다.",
            crop=crop,
            warnings=["검색 결과를 확인한 뒤 최종 답변을 검증합니다."],
        )

    def route_query(self, question: str) -> RouteDecision:
        """검색을 실행하지 않고 검색 경로 판단 결과만 반환합니다."""
        return self.decide(question)


class JevQuestionRouter(QuestionRouter):
    """JEV를 사용하고, 실패하면 규칙 기반 Router로 fallback합니다."""

    def __init__(self, classifier=None):
        self.classifier = classifier

    @classmethod
    def from_env(cls) -> "JevQuestionRouter":
        if TypeSafeClassifier is None:
            return cls()
        try:
            return cls(TypeSafeClassifier())
        except (TypeError, ValueError):
            return cls()

    def route_query(self, question: str) -> RouteDecision:
        if self.classifier is None or Choice is None:
            return super().route_query(question)

        try:
            response = self.classifier.invoke(
                {
                    "state": question,
                    "questions": {
                        "route": Choice(
                            instructions=(
                                "Choose the retrieval route. Use GRAPH for explicit "
                                "entity or relationship questions, VECTOR for textual "
                                "explanations, HYBRID for questions needing both, and "
                                "REJECT only for empty or out-of-scope questions."
                            ),
                            criteria={
                                "GRAPH": "Entity, relationship, crop, variety, region, environment, growth, or pest graph reasoning.",
                                "VECTOR": "Simple cultivation method, terminology, or document explanation.",
                                "HYBRID": "A combined condition, disease, prevention, management, and evidence question.",
                                "REJECT": "Empty, unrelated, or out-of-scope request.",
                            },
                        )
                    },
                }
            )
            route = response.choices["route"].choice
            if route not in {"GRAPH", "VECTOR", "HYBRID", "REJECT"}:
                raise ValueError(f"Unexpected Jev route: {route!r}")
            fallback = super().route_query(question)
            return RouteDecision(
                route=route,
                reason="Jev 구조화 라우팅 결과",
                crop=fallback.crop,
                max_hops=fallback.max_hops,
                warnings=fallback.warnings,
            )
        except Exception:
            fallback = super().route_query(question)
            return RouteDecision(
                route=fallback.route,
                reason=f"Jev 호출 실패로 규칙 기반 fallback: {fallback.reason}",
                crop=fallback.crop,
                max_hops=fallback.max_hops,
                warnings=fallback.warnings + ["Jev 라우팅을 사용하지 못했습니다."],
            )
