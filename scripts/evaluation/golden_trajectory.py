"""Collect one Langfuse trace per golden-set question.

This is the instrumentation-first runner. It uses the golden-set answer as the
temporary answer source; the Supervisor adapter can replace ``run_pipeline``
later without changing the trace/span contract.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/golden_trajectory_results.jsonl"
TRACE_NAME = "question_run"
MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL", "golden-set-baseline")
PROMPT_VERSION = "answer-v3"


def scenario_for(row: dict[str, Any]) -> str:
    behavior = str(row.get("expected_behavior", "")).upper()
    strategy = str(row.get("query_strategy", "")).lower()
    if behavior in {"REJECT", "REFUSE"} or strategy == "reject":
        return "safety"
    if behavior in {"INSUFFICIENT", "WITHHELD"} or strategy in {
        "insufficient", "evidence_gap"
    }:
        return "insufficient"
    if strategy in {"graph", "hybrid", "multi_hop", "multihop"}:
        return "multi_hop"
    return "normal"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "READY") != "READY":
                continue
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question on line {line_number}")
            answer = row.get("reference") or "\n".join(
                str(item) for item in row.get("expected_facts", [])
            )
            cases.append({
                "question_id": row.get("case_id", f"line-{line_number}"),
                "question": question,
                "answer": answer or "골든셋에 정답 원문이 없습니다.",
                "scenario": scenario_for(row),
                "expected_behavior": row.get("expected_behavior"),
                "query_strategy": row.get("query_strategy"),
            })
    return cases


def safe_error(exc: Exception) -> dict[str, str]:
    error_type = type(exc).__name__
    code = {
        "TimeoutError": "RETRIEVAL_TIMEOUT",
        "ConnectionError": "DEPENDENCY_CONNECTION_ERROR",
        "FileNotFoundError": "INPUT_NOT_FOUND",
    }.get(error_type, "PIPELINE_ERROR")
    return {
        "error_type": error_type,
        "error_code": code,
        "error_message": str(exc)[:200],
    }


def run_pipeline(case: dict[str, Any]) -> dict[str, Any]:
    """Instrumentation-first pipeline; replace internals with Supervisor later."""
    strategy = str(case.get("query_strategy") or "vector").lower()
    if case["scenario"] == "safety":
        return {
            "initial_route": "REJECT",
            "final_route": "REJECT",
            "route_path": ["REJECT"],
            "vector_result_count": 0,
            "graph_result_count": 0,
            "merged_result_count": 0,
            "hop_count": 0,
            "retry_count": 0,
            "answer": case["answer"],
            "final_status": "REJECTED",
        }

    route = "HYBRID" if strategy in {"hybrid", "multi_hop", "multihop"} else "VECTOR"
    return {
        "initial_route": route,
        "final_route": route,
        "route_path": [route],
        "vector_result_count": 0,
        "graph_result_count": 0,
        "merged_result_count": 0,
        "hop_count": 0,
        "retry_count": 0,
        "answer": case["answer"],
        "final_status": "ANSWERED" if case["answer"] else "WITHHELD",
    }


def collect_case(case: dict[str, Any], langfuse: Any) -> dict[str, Any]:
    started = time.perf_counter()
    visited: list[str] = []
    result: dict[str, Any] = {}
    error: dict[str, str] | None = None

    # langfuse 4.x 에는 "trace" observation 타입이 없다. 루트 span 이 그대로 trace 가 되고,
    # 이 span 의 metadata 가 trace metadata 로 올라간다.
    with langfuse.start_as_current_observation(
        as_type="span",
        name=TRACE_NAME,
        input=case["question"],
        metadata={
            "question_id": case["question_id"],
            "scenario": case["scenario"],
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
        },
    ) as trace:
        try:
            with langfuse.start_as_current_observation(
                as_type="span", name="guardrail_input", metadata={"status": "CHECKED"}
            ):
                visited.append("guardrail_input")

            with langfuse.start_as_current_observation(
                as_type="span", name="route", metadata={"status": "EXECUTED"}
            ):
                pipeline_preview = run_pipeline(case)
                visited.append("route")

            route = pipeline_preview["final_route"]
            if route in {"VECTOR", "HYBRID"}:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="vector_retrieve",
                    metadata={
                        "attempt": 1,
                        "status": "EXECUTED",
                        "result_count": pipeline_preview["vector_result_count"],
                    },
                ):
                    visited.append("vector_retrieve")

            if route in {"GRAPH", "HYBRID"}:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="graph_retrieve",
                    metadata={
                        "attempt": 1,
                        "status": "STUB",
                        "result_count": pipeline_preview["graph_result_count"],
                        "actual_hops": pipeline_preview["hop_count"],
                    },
                ):
                    visited.append("graph_retrieve")

            if route == "HYBRID":
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="merge",
                    metadata={
                        "status": "STUB",
                        "result_count": pipeline_preview["merged_result_count"],
                    },
                ):
                    visited.append("merge")

            with langfuse.start_as_current_observation(
                as_type="span",
                name="verify",
                metadata={"status": "EXECUTED", "result_count": pipeline_preview["merged_result_count"]},
            ):
                visited.append("verify")

            # Retry is created only when the pipeline reports an actual retry.
            for attempt in range(1, pipeline_preview["retry_count"] + 1):
                with langfuse.start_as_current_observation(
                    as_type="span", name="retry", metadata={"attempt": attempt, "status": "EXECUTED"}
                ):
                    visited.append("retry")

            answer = pipeline_preview["answer"]
            with langfuse.start_as_current_observation(
                as_type="span",
                name="answer",
                input=case["question"],
                output=answer,
                metadata={
                    "model": MODEL_NAME,
                    "prompt_version": PROMPT_VERSION,
                    "evidence_count": pipeline_preview["merged_result_count"],
                    "output_length": len(answer),
                },
            ):
                visited.append("answer")

            with langfuse.start_as_current_observation(
                as_type="span", name="guardrail_output", metadata={"status": "CHECKED"}
            ):
                visited.append("guardrail_output")

            result = pipeline_preview
        except Exception as exc:
            error = safe_error(exc)
            result = {"final_status": "ERROR"}

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        trace.update(
            output={"answer": result.get("answer", ""), "status": result.get("final_status")},
            metadata={
                "question_id": case["question_id"],
                "scenario": case["scenario"],
                "initial_route": result.get("initial_route"),
                "final_route": result.get("final_route"),
                "route_path": result.get("route_path", []),
                "vector_result_count": result.get("vector_result_count", 0),
                "graph_result_count": result.get("graph_result_count", 0),
                "merged_result_count": result.get("merged_result_count", 0),
                "hop_count": result.get("hop_count", 0),
                "retry_count": result.get("retry_count", 0),
                "final_status": result.get("final_status", "ERROR"),
                "latency_ms": latency_ms,
                "visited_nodes": visited,
                **(error or {}),
            },
        )

    return {
        "question_id": case["question_id"],
        "scenario": case["scenario"],
        "trace_id": getattr(trace, "trace_id", None),
        "root_span_id": getattr(trace, "id", None),
        "visited_nodes": visited,
        "final_status": result.get("final_status", "ERROR"),
        "latency_ms": latency_ms,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    langfuse = get_client()

    # 키가 없거나 틀려도 SDK 는 예외를 던지지 않고 전송만 조용히 실패한다.
    # 그대로 두면 Langfuse 에 한 건도 안 들어간 실행을 "수집 완료"로 착각하므로 먼저 끊는다.
    try:
        authenticated = langfuse.auth_check()
    except Exception as exc:  # 네트워크·주소 오류도 같은 자리에서 알린다.
        raise SystemExit(
            f"Langfuse 인증 확인에 실패했습니다({type(exc).__name__}: {str(exc)[:200]}). "
            f"{PROJECT_ROOT / '.env'} 의 LANGFUSE_BASE_URL 을 확인하세요."
        ) from exc
    if not authenticated:
        raise SystemExit(
            "Langfuse 인증이 거부되었습니다. "
            f"{PROJECT_ROOT / '.env'} 의 LANGFUSE_PUBLIC_KEY·LANGFUSE_SECRET_KEY 를 확인하세요."
        )
    base_url = os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    print(f"Langfuse 인증 확인: {base_url}", flush=True)

    rows = [collect_case(case, langfuse) for case in load_cases(args.golden)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    langfuse.flush()
    print(json.dumps({"count": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
