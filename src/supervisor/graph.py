"""검색·검증 흐름을 조정하는 LangGraph Supervisor입니다."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.retrievers.hybrid_retriever import hybrid_search
from src.retrievers.router import QuestionRouter
from src.retrievers.search_result import SearchResult

from .answer_agent import AnswerAgent
from .state import GraphState
from src.guardrails.supervisor_adapter import guardrails_node
from src.guardrails.input_rail import input_safety_node, after_input_safety
from src.guardrails.supervisor_adapter import guardrails_node
from src.guardrails.dialog_rail import dialog_node, after_dialog
from src.guardrails.retrieval_rail import filter_retrieval_results
from src.guardrails.execution_adapter import execution_node, after_execution
from src.guardrails.neo4j_pesticide_lookup import make_neo4j_pesticide_lookup

GraphSearch = Callable[..., list[SearchResult]]
VectorSearch = Callable[..., list[SearchResult]]
AnswerGenerator = Callable[[str, list[SearchResult]], str]


def _default_answer(question: str, results: list[SearchResult]) -> str:
    """LLM 연결 전에도 검색 경로를 확인할 수 있는 임시 답변을 만듭니다."""
    del question
    return "\n".join(result.text for result in results)

INSUFFICIENT_EVIDENCE_WARNING = "답변에 사용할 공식 근거가 부족합니다."


def _append_warning(existing: list[str], message: str) -> list[str]:
    """같은 경고를 재시도마다 반복해서 쌓지 않습니다."""
    return existing if message in existing else [*existing, message]

def _has_usable_evidence(result: SearchResult) -> bool:
    """검색 결과가 답변 근거로 사용할 수 있는지 확인합니다."""
    if result.metadata.get("retrieval_eligible") is False:
        return False
    return bool(result.evidence_ids or result.source_ids or result.chunk_ids)


def build_supervisor(
    *,
    graph_search_fn: GraphSearch,
    vector_search_fn: VectorSearch,
    pesticide_lookup_fn=None,
    current_verifier_fn=None,
    hybrid_search_fn: Callable[..., list[SearchResult]] | None = None,
    router: QuestionRouter | None = None,
    answer_generator: AnswerGenerator | None = None,
    max_retries: int = 1,
    result_limit: int = 10,
):
    """검색 Worker와 검증 흐름을 연결한 실행 가능한 LangGraph를 만듭니다.

    Router는 경로만 결정하고, 실제 검색 함수 호출은 Supervisor 노드가
    담당합니다. ``answer_generator``를 주입하지 않으면 검색 결과를
    줄바꿈으로 보여주는 검사용 기본 답변을 사용합니다.
    """
    if max_retries < 0:
        raise ValueError("max_retries는 0 이상이어야 합니다.")
    if result_limit < 1:
        raise ValueError("result_limit은 양수여야 합니다.")

    selected_router = router or QuestionRouter()
    selected_pesticide_lookup = (
        pesticide_lookup_fn
        if pesticide_lookup_fn is not None
        else make_neo4j_pesticide_lookup()
    )
    selected_hybrid = hybrid_search_fn or hybrid_search
    selected_answer_agent = AnswerAgent.from_env()
    generate_answer = answer_generator or selected_answer_agent.generate

    def route_node(state: GraphState) -> dict[str, Any]:
        decision = selected_router.route_query(state.get("question", ""))
        return {
            "route": decision.route,
            # 최초 판단은 fallback이 덮어쓰지 않도록 따로 보존합니다.
            "initial_route": decision.route,
            "route_path": [decision.route],
            "route_reason": decision.reason,
            "crop": decision.crop,
            "max_hops": decision.max_hops,
            "warnings": list(decision.warnings),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", max_retries),
            'route_reason': decision.reason,
            'initial_route_reason': decision.reason,
        }

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        route = state.get("route", "REJECT")
        question = state.get("question", "")
        crop = state.get("crop")
        max_hops = state.get("max_hops", 2)
        if route == "GRAPH":
            results = graph_search_fn(
                question, crop=crop, max_hops=max_hops, limit=result_limit
            )
        elif route == "VECTOR":
            results = vector_search_fn(question, k=result_limit)
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
        return {"results": results,
                # 재시도가 일어나도 이전 시도의 검색 결과르 잃지 않도록 누적
                "retrieved_results": [*state.get('retrieved_results', []), *results],
                'attemps': [*state.get('attemps', []), {'route': route, 'max_hops': max_hops, 'limit': result_limit, 'found': len(results)}],
                }


    def verify_node(state: GraphState) -> dict[str, Any]:
        """검색 근거 검증 및 평가용 원본 보존."""

        results = state.get("results", [])

        checked = filter_retrieval_results(
            results,
            pesticide_use_requested=(
                state.get("pesticide_use_requested") is True
            ),
        )

        usable = [
            result
            for result in checked.accepted
            if _has_usable_evidence(result)
        ]

        rejected = [
            result
            for result in results
            if not any(result is item for item in usable)
        ]

        warnings = list(dict.fromkeys([
            *state.get("warnings", []),
            *checked.warnings,
        ]))

        if not usable:
            warnings = _append_warning(
                warnings,
                INSUFFICIENT_EVIDENCE_WARNING,
            )

        return {
            "results": usable,
            "rejected_results": [
                *state.get("rejected_results", []),
                *rejected,
            ],
            "retrieval_rejections": checked.rejected,
            "evidence_status": (
                "PASS" if usable else "INSUFFICIENT"
            ),
            "warnings": warnings,
        }


        # 2. 기존 Supervisor 근거 검사 유지
        usable = [
            result
            for result in checked.accepted
            if _has_usable_evidence(result)
        ]

        # 3. 기존 경고와 신규 경고 병합
        warnings = list(
            dict.fromkeys([
                *state.get("warnings", []),
                *checked.warnings,
            ])
        )

        if usable:
            return {
                "results": usable,
                "evidence_status": "PASS",
                "warnings": warnings,
                "retrieval_rejections": checked.rejected,
            }

        # 4. 근거가 모두 제외되면 기존 재검색 흐름 사용
        return {
            "results": [],
            "evidence_status": "INSUFFICIENT",
            "warnings": list(dict.fromkeys([
                *warnings,
                "답변에 사용할 검증 가능한 근거가 부족합니다.",
            ])),
            "retrieval_rejections": checked.rejected,
            'rejected_results': [r for r in results if not _has_usable_evidence(r)],
            "warnings": _append_warning(state.get("warnings", []), INSUFFICIENT_EVIDENCE_WARNING),
        }





    def answer_node(state: GraphState) -> dict[str, Any]:
        """검색 근거가 충분할 때 답변 초안만 생성합니다."""

        if state.get("evidence_status") != "PASS":
            return {
                "answer": "",
            }

        answer = generate_answer(
            state.get("question", ""),
            state.get("results", []),
        )

        # 최종 답변은 Guardrails 검증 후 결정
        return {
            "answer": answer,
        }


    def reject_node(state: GraphState) -> dict[str, Any]:
        """범위 밖 질문을 검색 없이 거절"""
        reason = state.get("route_reason") or "질문이 이 서비스의 범위를 벗어났습니다."
        return {
            "results": [],
            # 검증을 수행하지 않았다는 의미
            # INSUFFICIENT (검색은 했지만 근거가 부족함)와 구별해야 함
            "evidence_status": "NOT_APPLICABLE",
            "final_status": "REJECTED",
            "final_answer": (
                f"{reason} 농업 재배·병해충·농약·비료 관련 질문을 입력해 주세요."
            )
        }

    def reject_node(state: GraphState) -> dict[str, Any]:
        """범위 밖 질문을 검색 없이 거절"""
        reason = state.get("route_reason") or "질문이 이 서비스의 범위를 벗어났습니다."
        return {
            "results": [],
            # 검증을 수행하지 않았다는 의미
            # INSUFFICIENT (검색은 했지만 근거가 부족함)와 구별해야 함
            "evidence_status": "NOT_APPLICABLE",
            "final_status": "REJECTED",
            "final_answer": (
                f"{reason} 농업 재배·병해충·농약·비료 관련 질문을 입력해 주세요."
            )
        }

    def retry_node(state: GraphState) -> dict[str, Any]:
        # route만 HYBRID로 바꾸고, initial_route는 건드리지 않습니다.
        # route_path는 '경로가 바뀐 이력', 같은 경로로 재시도할 때는
        # 덧붙이지 않음. 재시도 횟수는 retry_count()가 따로 셈
        path = state.get('route_path', [])
        return {
            "route": "HYBRID",
            "route_path": path if path and path[-1] == "HYBRID" else [*path, 'HYBRID'],
            "route_reason": "기존 검색 결과의 근거가 부족해 하이브리드 검색으로 재시도합니다.",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def after_route(state: GraphState) -> str:
        return "retrieve" if state.get("route") != "REJECT" else "reject"

    def after_verify(state: GraphState) -> str:
        if state.get("evidence_status") == "PASS":
            return "answer"
        if state.get("retry_count", 0) < state.get("max_retries", max_retries):
            return "retry"
        return "answer"



    workflow = StateGraph(GraphState)

    # =====================================
    # 1. Supervisor 노드 등록
    # =====================================
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("reject", reject_node)

    # =====================================
    # 2. Guardrails 노드 등록
    # =====================================
    workflow.add_node("input_safety", input_safety_node)
    workflow.add_node("dialog", dialog_node)
    workflow.add_node("guardrails", guardrails_node)

    # Execution Rail
    workflow.add_node(
        "execution",
        lambda state: execution_node(
            state,
            lookup_fn=selected_pesticide_lookup,
            current_verifier=current_verifier_fn,
        ),
    )

    # =====================================
    # 3. Input Rail → Dialog Rail
    # =====================================
    workflow.add_edge(START, "input_safety")
    workflow.add_edge("input_safety", "dialog")

    workflow.add_conditional_edges(
        "dialog",
        after_dialog,
        {
            "route": "route",
            "end": END,
        },
    )

    # =====================================
    # 4. Router 분기
    # =====================================
    workflow.add_conditional_edges(
        "route",
        after_route,
        {
            "retrieve": "retrieve",
            "reject": "reject",
        },
    )

    # =====================================
    # 5. 검색 → Retrieval Rail 검증
    # =====================================
    workflow.add_edge("retrieve", "verify")

    workflow.add_conditional_edges(
        "verify",
        after_verify,
        {
            "retry": "retry",
            "answer": "execution",
        },
    )

    # 근거 부족 시 기존 재검색 유지
    workflow.add_edge("retry", "retrieve")

    # =====================================
    # 6. Execution Rail 분기
    # =====================================
    workflow.add_conditional_edges(
        "execution",
        after_execution,
        {
            "answer": "answer",
            "end": END,
        },
    )

    # =====================================
    # 7. Answer → Output Guardrails
    # =====================================
    workflow.add_edge("answer", "guardrails")
    workflow.add_edge("guardrails", END)

    # 범위 밖 질문은 별도 종료
    workflow.add_edge("reject", END)

    return workflow.compile()


