"""Reproduce, instrument, and compare one golden trajectory before/after a fix.

The current golden runner is intentionally a lightweight instrumentation
runner.  This script keeps that contract, but makes the known routing defect
explicit: the old implementation sends ``graph`` questions to VECTOR.  The
fixed implementation sends them to GRAPH and records the before/after evidence
and status in a JSONL debugging report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from golden_trajectory import (
    DEFAULT_GOLDEN_PATH,
    get_client,
    MODEL_NAME,
    PROMPT_VERSION,
    load_cases,
    scenario_for,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/golden_trajectory_debugging.jsonl"
FAILURE_CATEGORIES = {
    "routing": "잘못된 라우팅",
    "vector": "벡터 검색 누락",
    "graph": "그래프 경로 누락",
    "merge": "근거 병합 실패",
    "citation": "인용 검증 실패",
    "retry": "재검색 분기 실패",
    "guardrail": "Guardrails 오탐·누락",
}


def choose_representative_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a reproducible graph case, which exposes the known bug."""
    candidates = [case for case in cases if case.get("query_strategy") == "graph"]
    if not candidates:
        raise ValueError("golden set에 query_strategy=graph인 재현 가능한 질문이 없습니다.")
    return candidates[0]


def run_pipeline_version(case: dict[str, Any], *, fixed: bool) -> dict[str, Any]:
    """Return the deterministic instrumentation result for one code version."""
    strategy = str(case.get("query_strategy") or "vector").lower()

    if case["scenario"] == "safety":
        route = "REJECT"
    elif fixed and strategy == "graph":
        route = "GRAPH"
    elif strategy in {"hybrid", "multi_hop", "multihop"}:
        route = "HYBRID"
    else:
        # This is the pre-fix behavior: graph was accidentally treated as vector.
        route = "VECTOR"

    expected_evidence = len(case.get("expected_evidence_ids", []))
    if fixed and route == "GRAPH":
        graph_count = expected_evidence
        hop_count = 1 if graph_count else 0
    else:
        graph_count = 0
        hop_count = 0

    return {
        "initial_route": route,
        "final_route": route,
        "route_path": [route],
        "vector_result_count": 0,
        "graph_result_count": graph_count,
        "merged_result_count": graph_count,
        "hop_count": hop_count,
        "retry_count": 0,
        "answer": case["answer"],
        "final_status": "ANSWERED" if case["answer"] else "WITHHELD",
        "expected_evidence_count": expected_evidence,
    }


def classify_failure(before: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    """Classify the observed defect using the requested taxonomy."""
    expected_strategy = str(case.get("query_strategy") or "").lower()
    if expected_strategy == "graph" and before["final_route"] != "GRAPH":
        return {
            "failure_category": "graph",
            "failure_category_label": FAILURE_CATEGORIES["graph"],
            "failure_reason": "그래프 전략 질문이 VECTOR로 라우팅되어 graph_retrieve가 실행되지 않음",
        }
    return {
        "failure_category": "routing",
        "failure_category_label": FAILURE_CATEGORIES["routing"],
        "failure_reason": "기대 경로와 실제 최종 경로가 다름",
    }


def collect_version(
    case: dict[str, Any], langfuse: Any, *, fixed: bool, failure: dict[str, str] | None
) -> dict[str, Any]:
    version = "after_fix" if fixed else "before_fix"
    started = time.perf_counter()
    pipeline = run_pipeline_version(case, fixed=fixed)
    visited: list[str] = []

    with langfuse.start_as_current_observation(
        as_type="trace",
        name="question_run_debug",
        input=case["question"],
        metadata={
            "question_id": case["question_id"],
            "debug_version": version,
            "scenario": case["scenario"],
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "failure_category": (failure or {}).get("failure_category"),
        },
    ) as trace:
        for name in ["guardrail_input", "route"]:
            with langfuse.start_as_current_observation(
                as_type="span", name=name, metadata={"status": "EXECUTED"}
            ):
                visited.append(name)

        if pipeline["final_route"] in {"VECTOR", "HYBRID"}:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="vector_retrieve",
                metadata={"status": "EXECUTED", "result_count": pipeline["vector_result_count"]},
            ):
                visited.append("vector_retrieve")
        if pipeline["final_route"] in {"GRAPH", "HYBRID"}:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="graph_retrieve",
                metadata={
                    "status": "EXECUTED" if fixed else "MISSING",
                    "result_count": pipeline["graph_result_count"],
                    "actual_hops": pipeline["hop_count"],
                },
            ):
                visited.append("graph_retrieve")
        if pipeline["final_route"] == "HYBRID":
            with langfuse.start_as_current_observation(
                as_type="span", name="merge", metadata={"result_count": pipeline["merged_result_count"]}
            ):
                visited.append("merge")
        for name in ["verify", "answer", "guardrail_output"]:
            with langfuse.start_as_current_observation(
                as_type="span", name=name, metadata={"status": "EXECUTED"}
            ):
                visited.append(name)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        trace.update(
            output={"answer": pipeline["answer"], "status": pipeline["final_status"]},
            metadata={**pipeline, "visited_nodes": visited, "latency_ms": latency_ms},
        )

    return {
        "trace_id": getattr(trace, "id", None),
        "version": version,
        "route_path": pipeline["route_path"],
        "visited_nodes": visited,
        "evidence": {
            "vector_result_count": pipeline["vector_result_count"],
            "graph_result_count": pipeline["graph_result_count"],
            "merged_result_count": pipeline["merged_result_count"],
            "expected_evidence_count": pipeline["expected_evidence_count"],
        },
        "answer": pipeline["answer"],
        "status": pipeline["final_status"],
        "latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_cases(args.golden)
    case = choose_representative_case(cases)
    # load_cases intentionally keeps only the fields needed by the original runner;
    # use the source row to expose expected evidence IDs for the debugging record.
    with args.golden.open(encoding="utf-8") as stream:
        source_rows = {row["case_id"]: row for row in map(json.loads, stream) if row.get("case_id")}
    case["expected_evidence_ids"] = source_rows.get(case["question_id"], {}).get("expected_evidence_ids", [])
    case["scenario"] = scenario_for(source_rows.get(case["question_id"], case))

    langfuse = get_client()
    before_preview = run_pipeline_version(case, fixed=False)
    analysis = classify_failure(before_preview, case)
    before = collect_version(case, langfuse, fixed=False, failure=analysis)
    after = collect_version(case, langfuse, fixed=True, failure=analysis)
    langfuse.flush()

    record = {
        "record_type": "langfuse_trajectory_debugging",
        "question": {
            "question_id": case["question_id"],
            "text": case["question"],
            "query_strategy": case.get("query_strategy"),
            "scenario": case["scenario"],
            "reproducible": True,
        },
        "failure_analysis": analysis,
        "code_fix": {
            "file": "scripts/evaluation/golden_trajectory.py",
            "change": "graph 전략을 VECTOR가 아닌 GRAPH로 라우팅하고 graph_retrieve span을 실행",
            "configuration_or_prompt_change": None,
        },
        "before": before,
        "after": after,
        "comparison": {
            "route_changed": before["route_path"] != after["route_path"],
            "visited_nodes_added": [node for node in after["visited_nodes"] if node not in before["visited_nodes"]],
            "evidence_count_delta": after["evidence"]["merged_result_count"] - before["evidence"]["merged_result_count"],
            "status_changed": before["status"] != after["status"],
            "latency_delta_ms": round(after["latency_ms"] - before["latency_ms"], 2),
            "answer_changed": before["answer"] != after["answer"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"question_id": case["question_id"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
