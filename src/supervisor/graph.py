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


GraphSearch = Callable[..., list[SearchResult]]
VectorSearch = Callable[..., list[SearchResult]]
AnswerGenerator = Callable[[str, list[SearchResult]], str]


def _default_answer(question: str, results: list[SearchResult]) -> str:
    """LLM 연결 전에도 검색 경로를 확인할 수 있는 임시 답변을 만듭니다."""
    del question
    return "\n".join(result.text for result in results)


def _has_usable_evidence(result: SearchResult) -> bool:
    """검색 결과가 답변 근거로 사용할 수 있는지 확인합니다."""
    if result.metadata.get("retrieval_eligible") is False:
        return False
    return bool(result.evidence_ids or result.source_ids or result.chunk_ids)


def build_supervisor(
    *,
    graph_search_fn: GraphSearch,
    vector_search_fn: VectorSearch,
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
        return {"results": results}

    def verify_node(state: GraphState) -> dict[str, Any]:
        results = state.get("results", [])
        usable = [result for result in results if _has_usable_evidence(result)]
        if usable:
            return {"results": usable, "evidence_status": "PASS"}
        return {
            "results": [],
            "evidence_status": "INSUFFICIENT",
            "warnings": [*state.get("warnings", []), "답변에 사용할 공식 근거가 부족합니다."],
        }

    def answer_node(state: GraphState) -> dict[str, Any]:
        if state.get("evidence_status") != "PASS":
            return {
                "final_status": "WITHHELD",
                "final_answer": "충분한 공식 근거를 확인하지 못해 답변을 보류합니다.",
            }
        answer = generate_answer(state.get("question", ""), state.get("results", []))
        return {"answer": answer, "final_answer": answer, "final_status": "ANSWERED"}

    def retry_node(state: GraphState) -> dict[str, Any]:
        # route만 HYBRID로 바꾸고, initial_route는 건드리지 않습니다.
        # route_path에는 이동 이력을 덧붙여 어디서 넘어왔는지 남깁니다.
        return {
            "route": "HYBRID",
            "route_path": [*state.get("route_path", []), "HYBRID"],
            "route_reason": "기존 검색 결과의 근거가 부족해 하이브리드 검색으로 재시도합니다.",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def after_route(state: GraphState) -> str:
        return "retrieve" if state.get("route") != "REJECT" else "answer"

    def after_verify(state: GraphState) -> str:
        if state.get("evidence_status") == "PASS":
            return "answer"
        if state.get("retry_count", 0) < state.get("max_retries", max_retries):
            return "retry"
        return "answer"

    workflow = StateGraph(GraphState)
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("answer", answer_node)
    workflow.add_edge(START, "route")
    workflow.add_conditional_edges("route", after_route, {"retrieve": "retrieve", "answer": "answer"})
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges("verify", after_verify, {"retry": "retry", "answer": "answer"})
    workflow.add_edge("retry", "retrieve")
    workflow.add_edge("answer", END)
    return workflow.compile()
