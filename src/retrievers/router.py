"""하이브리드 농업 GraphRAG 파이프라인의 질문 라우터입니다."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = None
    Field = None

try:
    from langchain_typesafe import Choice, TypeSafeClassifier
except ImportError:  # Optional until TYPESAFE_API_KEY is configured.
    Choice = None
    TypeSafeClassifier = None

Route = Literal["GRAPH", "VECTOR", "HYBRID", "REJECT"]

VALID_RELATION_TYPES = frozenset({
    "HAS_VARIETY", "SUITABLE_FOR_REGION",
    "HAS_SOWING_OBSERVATION", "HAS_SOWING_RECOMMENDATION", "USES_VARIETY",
    "HAS_ENVIRONMENT_REQUIREMENT", "HAS_ENVIRONMENT_OBSERVATION",
    "HAS_GROWTH_OBSERVATION", "AFFECTED_BY_DISEASE", "AFFECTED_BY_INSECT",
    "HAS_REGISTERED_PESTICIDE_USE", "CONTROLS_TARGET", "HAS_FERTILIZER_STANDARD",
})


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    crop: str | None = None
    max_hops: int = 2
    entities: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    needs_official_evidence: bool = False
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


if BaseModel is not None:
    class QueryAnalysis(BaseModel):
        crop: str | None = None
        entities: list[str] = Field(default_factory=list)
        relation_types: list[str] = Field(default_factory=list)
        max_hops: int = 2
        needs_official_evidence: bool = False
else:  # pragma: no cover
    QueryAnalysis = None


class LLMQueryAnalyzer:
    """GRAPH/HYBRID 질문에 필요한 검색 조건만 구조화합니다."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_env(cls):
        import os

        if not os.getenv("OPENAI_API_KEY") or QueryAnalysis is None:
            return None
        try:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            ).with_structured_output(QueryAnalysis)
            return cls(model)
        except Exception:
            return None

    def analyze(self, question: str) -> QueryAnalysis:
        return self.model.invoke(
            "질문에서 그래프 검색 조건만 추출하세요. "
            "답변하거나 안전 판단하지 마세요. "
            "작물, 품종·병해충·지역·환경 엔티티, 필요한 관계 타입, hop 수, "
            "공식 근거 필요 여부를 구조화합니다.\n질문: " + question
        )


class QuestionRouter:
    """Vector DB 구현과 분리된 검색 경로 판단을 수행합니다."""

    GRAPH_TERMS = (
        "관계", "연결", "경로", "품종", "적합 지역", "파종 시기", "생육 단계",
        "병해", "병해충", "해충", "환경 조건", "멀티홉", "어떤 지역",
        "어느 지역",
    )
    VECTOR_TERMS = (
        "방법", "뜻", "의미", "설명", "재배법", "관리법", "관수 방법",
        "시비 방법", "방제 방법", "예방", "무엇인가", "어떻게 재배",
    )
    OFFICIAL_NUMERIC_TERMS = (
        "농약", "비료", "희석", "배율", "사용량", "안전사용", "수확 전",
    )
    CROPS = ("상추", "양파", "마늘", "고추", "배추")

    # 글을 지어 달라는 요청. 작물 이름이 섞여 있어도 농업 지식 질의가 아닙니다.
    CREATIVE_REQUEST_PATTERN = re.compile(
        r"(?:시|글|노래|가사|편지|축사|축하\s*문구|동화|수필|에세이|랩)"
        r"\s*(?:를|을)?\s*(?:한\s*(?:편|곡|수)\s*)?"
        r"(?:써|지어|작성해|만들어|창작)"
        r"|작사|작곡|라임"
    )

    # 목록에 없는 병해충 이름을 도메인 신호로 인정하기 위한 접미사 규칙입니다.
    PEST_NAME_PATTERN = re.compile(
        r"[가-힣]{2,}(?:병|나방|진딧물|응애|총채벌레|잎벌레|굴파리|선충|균|곰팡이)"
    )

    # 작물 이름이 들어 있어도 이 말뭉치로 답할 수 없는 요청입니다.
    # 창작·투자·상품 추천·연예·법률처럼 농업 자료와 무관한 작업 의도를 가리킵니다.
    OUT_OF_SCOPE_INTENT_TERMS = (
        # 창작·마케팅 요청
        "게임 캐릭터", "캐릭터 설정", "캐릭터", "게임", "시놉시스", "영화", "드라마",
        "소설", "시나리오", "웹툰", "광고", "포스터", "문구", "카피", "슬로건", "브랜딩",
        # 투자·금융
        "주식", "종목", "증권", "코인", "비트코인", "투자", "수익률", "시세", "가격",
        # 상품·생활 추천
        "맛집", "데이트", "노트북", "스마트폰", "가전", "여행", "숙소", "레시피", "요리법",
        # 연예·스포츠
        "프로야구", "경기 결과", "스포츠", "연예인", "아이돌",
        # 법률·행정
        "법적 절차", "전세", "임대차", "소송", "세금 신고",
        # 일반 프로그래밍
        "파이썬", "자바스크립트", "알고리즘", "퀵정렬", "코드를 작성", "함수를 구현",
    )

    # 농업 도메인 신호입니다. 하나도 없으면 이 말뭉치의 범위 밖으로 봅니다.
    DOMAIN_TERMS = (
        # 작물·주체
        "상추", "양파", "마늘", "고추", "배추", "옥수수", "열무", "작물", "농작물",
        "원예작물", "농업", "농가", "농경지", "육묘장", "품종", "품종군",
        # 재배·생육
        "재배", "파종", "정식", "이식", "수확", "생육", "유묘", "추대", "꽃눈분화",
        "엇갈이", "채종", "관수", "포장", "하우스", "시설재배", "노지",
        "농사", "영농", "원예", "작황", "모종", "묘목", "멀칭", "양액", "수경",
        # 토양·경작지
        "밭", "논", "흙", "석회", "산도", "객토", "배수", "볏짚", "구비", "가축분",
        # 병해충·방제
        "병해", "병해충", "해충", "병원균", "병반", "발병", "방제", "예찰", "저항성",
        "유충", "애벌레", "나방", "진딧물", "응애", "총채벌레", "잎벌레", "흰나비",
        "노균병", "탄저병", "잎마름병", "풋마름병", "무름병", "뿌리혹병",
        # 농약·비료
        "농약", "약제", "살균제", "살충제", "제초제", "잡초", "희석", "살포",
        "안전사용", "비료", "시비", "밑거름", "웃거름", "질소", "인산", "칼리",
        "퇴비", "유기물", "탄질율", "토양",
        # 식물 부위·환경 자료
        "과실", "잎", "줄기", "뿌리", "꽃자루", "인경", "강우량", "ecocrop",
    )

    def scope_gate(self, question: str) -> RouteDecision | None:
        """범위 밖 질문이면 REJECT 판단을, 범위 안이면 None 을 돌려줍니다.

        검색을 실행하기 전에 먼저 걸러 내는 층입니다. 범위 안이지만 근거가 없는
        질문은 여기서 REJECT 하지 않고, 검색 뒤 답변 보류로 처리해야 합니다.
        """
        normalized = re.sub(r"\s+", " ", question.strip())
        if not normalized:
            return RouteDecision("REJECT", "질문이 비어 있습니다.", warnings=[])

        lowered = normalized.lower()

        # 작물·병해충 이름이 들어 있어도 글을 지어 달라는 요청은 농업 지식
        # 질의가 아닙니다. 단어 단위로는 '시비'·'시설재배'처럼 흔한 농업 용어와
        # 겹치므로 요청 형태를 패턴으로 봅니다.
        creative = self.CREATIVE_REQUEST_PATTERN.search(normalized)
        if creative is not None:
            return RouteDecision(
                "REJECT",
                f"농업 자료로 답할 수 없는 창작 요청입니다(‘{creative.group(0)}’).",
                warnings=[],
            )

        blocked = next(
            (term for term in self.OUT_OF_SCOPE_INTENT_TERMS if term in lowered), None
        )
        if blocked is not None:
            return RouteDecision(
                "REJECT",
                f"농업 자료로 답할 수 없는 요청입니다(‘{blocked}’).",
                warnings=[],
            )

        # 키워드 목록에 없는 병해충 이름도 도메인 신호로 봅니다. 목록에 없다는
        # 이유로 막으면 말뭉치에 근거가 있는 질문까지 검색 전에 버립니다.
        # 근거가 없는 경우는 검색 뒤 답변 보류로 처리합니다.
        has_domain_term = any(term in lowered for term in self.DOMAIN_TERMS)
        has_pest_name = self.PEST_NAME_PATTERN.search(normalized) is not None

        if not has_domain_term and not has_pest_name:
            return RouteDecision(
                "REJECT",
                "농업 도메인 신호가 없어 말뭉치 범위 밖으로 판단했습니다.",
                warnings=[],
            )
        return None

    def decide(self, question: str) -> RouteDecision:
        rejected = self.scope_gate(question)
        if rejected is not None:
            return rejected

        normalized = re.sub(r"\s+", " ", question.strip())
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

    def __init__(self, classifier=None, analyzer=None):
        self.classifier = classifier
        self.analyzer = analyzer

    @classmethod
    def from_env(cls) -> "JevQuestionRouter":
        if TypeSafeClassifier is None:
            return cls()
        try:
            classifier = TypeSafeClassifier(
                model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
                base_url=os.getenv(
                    "TYPESAFE_BASE_URL",
                    "https://api.typesafe.ai",
                ),
            )
            return cls(classifier, LLMQueryAnalyzer.from_env())
        except (TypeError, ValueError):
            return cls()

    def route_query(self, question: str) -> RouteDecision:
        # 범위 밖 질문은 분류기를 호출하지 않고 먼저 차단합니다.
        rejected = self.scope_gate(question)
        if rejected is not None:
            return rejected
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
            analysis = None
            warnings = list(fallback.warnings)
            if route in {"GRAPH", "HYBRID"} and self.analyzer is not None:
                try:
                    analysis = self.analyzer.analyze(question)
                except Exception:
                    warnings.append("그래프 검색 조건 LLM 분석에 실패했습니다.")
            valid_relations = [
                relation
                for relation in (analysis.relation_types if analysis else [])
                if relation in VALID_RELATION_TYPES
            ]
            if analysis and len(valid_relations) != len(analysis.relation_types):
                warnings.append("LLM이 반환한 일부 관계 타입을 제외했습니다.")
            return RouteDecision(
                route=route,
                reason="Jev 구조화 라우팅 결과",
                crop=(analysis.crop if analysis and analysis.crop else fallback.crop),
                max_hops=(analysis.max_hops if analysis else fallback.max_hops),
                entities=(analysis.entities if analysis else []),
                relation_types=valid_relations,
                needs_official_evidence=(analysis.needs_official_evidence if analysis else False),
                confidence=None,
                warnings=warnings,
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
