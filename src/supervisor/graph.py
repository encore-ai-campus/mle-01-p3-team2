"""검색·답변 생성·인용 검증 흐름을 조정하는 LangGraph Supervisor입니다."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.retrievers.answer import (
    CitedAnswer,
    Evidence,
    build_evidence,
    render_cited_answer,
)
from src.retrievers.hybrid_retriever import hybrid_search
from src.retrievers.router import QuestionRouter
from src.retrievers.search_result import SearchResult

from .answer_agent import AnswerAgent
from .citation_verifier import SupportChecker, verify_citations
from .state import GraphState


GraphSearch = Callable[..., list[SearchResult]]
VectorSearch = Callable[..., list[SearchResult]]

EvidenceLookup = Callable[
    [list[str]],
    dict[str, dict],
]

CitedAnswerGenerator = Callable[
    [str, list[Evidence]],
    CitedAnswer,
]


INSUFFICIENT_EVIDENCE_WARNING = (
    "답변에 사용할 공식 근거가 부족합니다."
)


def _append_warning(
    existing: list[str],
    message: str,
) -> list[str]:
    """같은 경고를 중복해서 쌓지 않습니다."""

    if message in existing:
        return existing

    return [*existing, message]

def _has_usable_evidence(
    result: SearchResult,
) -> bool:
    """공통 Evidence로 변환할 수 있는 검색 결과인지 확인합니다."""

    if result.metadata.get("retrieval_eligible") is False:
        return False

    # build_evidence()는 evidence_ids를 기준으로 원문을 조회합니다.
    if not result.evidence_ids:
        return False

    if result.source == "graph":
        edges = result.path_edges or []

        if (
            result.metadata.get("has_complete_evidence") is not True
            or not edges
        ):
            return False

        if not all(
            edge.get("original_chunks")
            and all(
                chunk.get("text", "").strip()
                for chunk in edge["original_chunks"]
            )
            for edge in edges
        ):
            return False

    return True


def _failed_claim_texts(
    state: GraphState,
) -> list[str]:
    """인용 검증에 실패한 주장 텍스트를 반환합니다."""

    cited_answer = state.get("cited_answer")

    if cited_answer is None:
        return []

    failure_indexes = {
        failure.get("claim_index")
        for failure in state.get("citation_failures", [])
        if isinstance(failure.get("claim_index"), int)
        and failure.get("claim_index", -1) >= 0
    }

    return [
        claim.text
        for index, claim in enumerate(cited_answer.claims)
        if index in failure_indexes
    ]


def _build_retry_query(
    state: GraphState,
) -> str:
    """원래 질문에 검증 실패 주장을 추가합니다."""

    # search_query를 재사용하면 재시도 문구가 중복될 수 있으므로
    # 항상 원래 사용자 질문에서 시작합니다.
    question = state.get("question", "")
    failed_claims = _failed_claim_texts(state)

    if not failed_claims:
        return question

    claim_block = "\n".join(
        f"- {claim}"
        for claim in failed_claims
    )

    return (
        f"{question}\n\n"
        "다음 사실을 확인할 수 있는 공식 근거를 우선 검색:\n"
        f"{claim_block}"
    )


def _build_withhold_reason(
    state: GraphState,
) -> str:
    """충돌·인용 실패·원문 조회 실패를 보류 사유로 정리합니다."""

    parts: list[str] = []

    for conflict in state.get("citation_conflicts", []):
        evidence_ids = ", ".join(
            conflict.get("evidence_ids", [])
        )
        reason = conflict.get(
            "reason",
            "근거 사이에 충돌이 있습니다.",
        )

        parts.append(
            f"CONFLICT [{evidence_ids}]: {reason}"
        )

    for failure in state.get("citation_failures", []):
        code = failure.get("code", "UNKNOWN")
        reason = failure.get(
            "reason",
            "인용 검증에 실패했습니다.",
        )

        parts.append(f"{code}: {reason}")

    for dropped in state.get("dropped_evidences", []):
        evidence_id = dropped.get("evidence_id", "")
        reason = dropped.get("reason", "UNKNOWN")

        parts.append(
            f"{reason} [{evidence_id}]"
        )

    if not parts:
        return "충분한 공식 근거를 확인하지 못했습니다."

    # 중복 사유를 제거하면서 기존 순서는 유지합니다.
    return "; ".join(dict.fromkeys(parts))


def build_supervisor(
    *,
    graph_search_fn: GraphSearch,
    vector_search_fn: VectorSearch,
    evidence_lookup: EvidenceLookup,
    hybrid_search_fn: Callable[
        ...,
        list[SearchResult],
    ] | None = None,
    router: QuestionRouter | None = None,
    cited_answer_generator: CitedAnswerGenerator | None = None,
    citation_support_checker: SupportChecker | None = None,
    max_retries: int = 1,
    result_limit: int = 10,
):
    """검색부터 인용 검증과 최종 출력까지 연결합니다."""

    if max_retries < 0:
        raise ValueError(
            "max_retries는 0 이상이어야 합니다."
        )

    if result_limit < 1:
        raise ValueError(
            "result_limit은 양수여야 합니다."
        )

    selected_router = router or QuestionRouter()
    selected_hybrid = hybrid_search_fn or hybrid_search
    selected_answer_agent = AnswerAgent.from_env()

    generate_cited_answer = (
        cited_answer_generator
        or selected_answer_agent.generate_cited
    )

    def route_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """최초 검색 경로를 결정합니다."""

        question = state.get("question", "")
        decision = selected_router.route_query(question)

        return {
            "route": decision.route,
            "initial_route": decision.route,
            "route_path": [decision.route],
            "route_reason": decision.reason,
            "initial_route_reason": decision.reason,
            "crop": decision.crop,
            "max_hops": decision.max_hops,
            "warnings": list(decision.warnings),

            # 최초 검색은 사용자 질문 그대로 실행
            "search_query": question,

            "retry_count": state.get(
                "retry_count",
                0,
            ),
            "max_retries": state.get(
                "max_retries",
                max_retries,
            ),

            "citation_status": "",
            "citation_failures": [],
            "citation_conflicts": [],
            "dropped_evidences": [],
        }

    def retrieve_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """선택된 경로로 검색합니다."""

        route = state.get("route", "REJECT")
        question = (
            state.get("search_query")
            or state.get("question", "")
        )
        crop = state.get("crop")
        max_hops = state.get("max_hops", 2)

        if route == "GRAPH":
            results = graph_search_fn(
                question,
                crop=crop,
                max_hops=max_hops,
                limit=result_limit,
            )

        elif route == "VECTOR":
            results = vector_search_fn(
                question,
                k=result_limit,
            )

        elif route == "HYBRID":
            results = selected_hybrid(
                question,
                graph_search_fn=graph_search_fn,
                vector_search_fn=vector_search_fn,
                crop=crop,
                max_hops=max_hops,
                limit=result_limit,
            )

        else:
            results = []

        attempts = [
            *state.get("attemps", []),
            {
                "route": route,
                "query": question,
                "max_hops": max_hops,
                "limit": result_limit,
                "found": len(results),
            },
        ]

        return {
            "results": results,

            # 평가용 검색 원본은 재시도 후에도 누적합니다.
            "retrieved_results": [
                *state.get("retrieved_results", []),
                *results,
            ],

            # 기존 필드명과의 호환을 위해 attemps를 유지합니다.
            "attemps": attempts,
        }

    def verify_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """검색 결과가 공통 Evidence로 변환 가능한지 확인합니다."""

        results = state.get("results", [])

        usable = [
            result
            for result in results
            if _has_usable_evidence(result)
        ]

        rejected = [
            result
            for result in results
            if not _has_usable_evidence(result)
        ]

        accumulated_rejected = [
            *state.get("rejected_results", []),
            *rejected,
        ]

        if usable:
            return {
                "results": usable,
                "rejected_results": accumulated_rejected,
                "evidence_status": "PASS",
            }

        return {
            "results": [],
            "rejected_results": accumulated_rejected,
            "evidence_status": "INSUFFICIENT",
            "warnings": _append_warning(
                state.get("warnings", []),
                INSUFFICIENT_EVIDENCE_WARNING,
            ),
        }

    def prepare_evidence_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """SearchResult를 공통 Evidence 형식으로 변환합니다."""

        evidences, dropped = build_evidence(
            state.get("results", []),
            evidence_lookup,
        )

        if not evidences:
            return {
                "evidences": [],
                "dropped_evidences": dropped,
                "evidence_status": "INSUFFICIENT",
                "warnings": _append_warning(
                    state.get("warnings", []),
                    INSUFFICIENT_EVIDENCE_WARNING,
                ),
            }

        return {
            "evidences": evidences,
            "dropped_evidences": dropped,
            "evidence_status": "PASS",
        }

    def answer_cited_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """Evidence를 이용해 구조화된 인용 답변을 생성합니다."""

        evidences = state.get("evidences", [])

        if not evidences:
            return {
                "cited_answer": None,
                "answer": "",
                "citation_status": "FAIL",
                "citation_failures": [
                    {
                        "code": "NO_CLAIMS",
                        "claim_index": -1,
                        "citation_index": None,
                        "evidence_id": None,
                        "reason": (
                            "인용 가능한 근거가 없습니다."
                        ),
                    }
                ],
                "citation_conflicts": [],
            }

        cited_answer = generate_cited_answer(
            state.get("question", ""),
            evidences,
        )

        return {
            "cited_answer": cited_answer,
            "answer": cited_answer.answer,

            # 검증 노드가 새 값을 설정하도록 초기화
            "citation_status": "",
            "citation_failures": [],
            "citation_conflicts": [],
        }

    def verify_citations_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """구조화 답변의 ID·인용문·주장 지원 여부를 검사합니다."""

        cited_answer = state.get("cited_answer")
        evidences = state.get("evidences", [])

        if cited_answer is None:
            return {
                "citation_status": "FAIL",
                "citation_failures": [
                    {
                        "code": "NO_CLAIMS",
                        "claim_index": -1,
                        "citation_index": None,
                        "evidence_id": None,
                        "reason": (
                            "구조화 답변이 생성되지 않았습니다."
                        ),
                    }
                ],
                "citation_conflicts": [],
                "checked_claims": 0,
                "supported_claims": 0,
            }

        if citation_support_checker is None:
            return {
                "citation_status": "FAIL",
                "citation_failures": [
                    {
                        "code": "NOT_SUPPORTED",
                        "claim_index": -1,
                        "citation_index": None,
                        "evidence_id": None,
                        "reason": (
                            "주장 지원 여부 판정기가 "
                            "연결되지 않았습니다."
                        ),
                    }
                ],
                "citation_conflicts": [],
                "checked_claims": len(
                    cited_answer.claims
                ),
                "supported_claims": 0,
            }

        result = verify_citations(
            answer=cited_answer,
            evidences=evidences,
            support_checker=citation_support_checker,
        )

        return {
            "citation_status": result.status,
            "citation_failures": [
                failure.model_dump()
                for failure in result.failures
            ],
            "citation_conflicts": [
                conflict.model_dump()
                for conflict in result.conflicts
            ],
            "checked_claims": result.checked_claims,
            "supported_claims": result.supported_claims,
        }

    def retry_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """구조 근거 또는 인용 근거 부족 시 HYBRID로 재검색합니다."""

        path = state.get("route_path", [])
        citation_status = state.get("citation_status")

        if citation_status == "PARTIAL":
            retry_reason = (
                "일부 주장에 대한 인용 근거가 부족해 "
                "하이브리드 검색으로 재시도합니다."
            )
            search_query = _build_retry_query(state)

        else:
            retry_reason = (
                "검색 결과의 구조적 근거가 부족해 "
                "하이브리드 검색으로 재시도합니다."
            )
            search_query = state.get("question", "")

        return {
            "route": "HYBRID",
            "route_path": (
                path
                if path and path[-1] == "HYBRID"
                else [*path, "HYBRID"]
            ),
            "route_reason": retry_reason,
            "retry_reason": retry_reason,
            "search_query": search_query,
            "retry_count": (
                state.get("retry_count", 0) + 1
            ),

            # 이전 시도 결과 초기화
            "results": [],
            "evidences": [],
            "dropped_evidences": [],
            "cited_answer": None,
            "answer": "",
            "citation_status": "",
            "citation_failures": [],
            "citation_conflicts": [],
            "checked_claims": 0,
            "supported_claims": 0,
        }

    def finalize_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """검증을 통과한 답변에 실제 출처 정보를 붙입니다."""

        cited_answer = state.get("cited_answer")
        evidences = state.get("evidences", [])

        if cited_answer is None:
            return {
                "final_status": "WITHHELD",
                "withhold_reason": (
                    "구조화 답변이 없습니다."
                ),
                "final_answer": (
                    "답변을 보류합니다."
                ),
            }

        final_answer = render_cited_answer(
            cited_answer,
            evidences,
        )

        return {
            "final_status": "ANSWERED",
            "withhold_reason": "",
            "final_answer": final_answer,
        }

    def withhold_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """검증 실패 원인을 포함해 답변을 보류합니다."""

        reason = _build_withhold_reason(state)

        return {
            "final_status": "WITHHELD",
            "withhold_reason": reason,
            "final_answer": (
                "검색 근거만으로 안전하고 정확한 답변을 "
                "확정할 수 없어 답변을 보류합니다.\n"
                f"사유: {reason}"
            ),
        }

    def reject_node(
        state: GraphState,
    ) -> dict[str, Any]:
        """범위 밖 질문은 검색 없이 거절합니다."""

        reason = (
            state.get("route_reason")
            or "질문이 서비스 범위를 벗어났습니다."
        )

        return {
            "results": [],
            "evidence_status": "NOT_APPLICABLE",
            "final_status": "REJECTED",
            "withhold_reason": "",
            "final_answer": (
                f"{reason} "
                "농업 재배·병해충·농약·비료 관련 "
                "질문을 입력해 주세요."
            ),
        }

    def after_route(
        state: GraphState,
    ) -> str:
        if state.get("route") == "REJECT":
            return "reject"

        return "retrieve"

    def after_verify(
        state: GraphState,
    ) -> str:
        if state.get("evidence_status") == "PASS":
            return "prepare_evidence"

        retry_count = state.get("retry_count", 0)
        retry_limit = state.get(
            "max_retries",
            max_retries,
        )

        if retry_count < retry_limit:
            return "retry"

        return "withhold"

    def after_prepare_evidence(
        state: GraphState,
    ) -> str:
        if state.get("evidences"):
            return "answer_cited"

        retry_count = state.get("retry_count", 0)
        retry_limit = state.get(
            "max_retries",
            max_retries,
        )

        if retry_count < retry_limit:
            return "retry"

        return "withhold"

    def after_citations(
        state: GraphState,
    ) -> str:
        """인용 검증 결과에 따라 최종 행동을 결정합니다."""

        # 유효한 근거끼리 충돌하면 PASS보다 우선하여 보류
        if state.get("citation_conflicts"):
            return "withhold"

        status = state.get("citation_status")
        retry_count = state.get("retry_count", 0)
        retry_limit = state.get(
            "max_retries",
            max_retries,
        )

        if status == "PASS":
            return "finalize"

        if (
            status == "PARTIAL"
            and retry_count < retry_limit
        ):
            return "retry"

        return "withhold"

    workflow = StateGraph(GraphState)

    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node(
        "prepare_evidence",
        prepare_evidence_node,
    )
    workflow.add_node(
        "answer_cited",
        answer_cited_node,
    )
    workflow.add_node(
        "verify_citations",
        verify_citations_node,
    )
    workflow.add_node("retry", retry_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("withhold", withhold_node)
    workflow.add_node("reject", reject_node)

    workflow.add_edge(START, "route")

    workflow.add_conditional_edges(
        "route",
        after_route,
        {
            "retrieve": "retrieve",
            "reject": "reject",
        },
    )

    workflow.add_edge("retrieve", "verify")

    workflow.add_conditional_edges(
        "verify",
        after_verify,
        {
            "prepare_evidence": "prepare_evidence",
            "retry": "retry",
            "withhold": "withhold",
        },
    )

    workflow.add_conditional_edges(
        "prepare_evidence",
        after_prepare_evidence,
        {
            "answer_cited": "answer_cited",
            "retry": "retry",
            "withhold": "withhold",
        },
    )

    workflow.add_edge(
        "answer_cited",
        "verify_citations",
    )

    workflow.add_conditional_edges(
        "verify_citations",
        after_citations,
        {
            "finalize": "finalize",
            "retry": "retry",
            "withhold": "withhold",
        },
    )

    workflow.add_edge("retry", "retrieve")
    workflow.add_edge("finalize", END)
    workflow.add_edge("withhold", END)
    workflow.add_edge("reject", END)

    return workflow.compile()