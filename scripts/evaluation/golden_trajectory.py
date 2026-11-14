"""Collect one Langfuse trace per golden-set question using the real Supervisor."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langfuse import get_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/golden_trajectory_results.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/evaluation/goldenset_trajectory.jsonl"
TRACE_NAME = "question_run"
PROMPT_VERSION = "answer-v3"

def scenario_for(row: dict[str, Any]) -> str:
    behavior, strategy = str(row.get("expected_behavior", "")).upper(), str(row.get("query_strategy", "")).lower()
    if behavior in {"REJECT", "REFUSE"} or strategy == "reject": return "safety"
    if behavior in {"INSUFFICIENT", "WITHHELD"} or strategy in {"insufficient", "evidence_gap"}: return "insufficient"
    if strategy in {"graph", "hybrid", "multi_hop", "multihop"}: return "multi_hop"
    return "normal"

def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("status", "READY") != "READY": continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip(): raise ValueError(f"Missing question on line {line_number}")
        cases.append({"question_id": row.get("case_id", f"line-{line_number}"), "question": question, "scenario": scenario_for(row)})
    return cases

def safe_error(exc: Exception) -> dict[str, str]:
    name = type(exc).__name__
    code = {"TimeoutError": "RETRIEVAL_TIMEOUT", "ConnectionError": "DEPENDENCY_CONNECTION_ERROR"}.get(name, "PIPELINE_ERROR")
    return {"error_type": name, "error_code": code, "error_message": str(exc)[:200]}

def run_pipeline(case: dict[str, Any], app: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Run production Supervisor with only the question (never golden references)."""
    state, updates = {}, []
    for update in app.stream({"question": case["question"]}, stream_mode="updates"):
        for node, values in update.items():
            if isinstance(values, dict): state.update(values); updates.append((node, values))
    return state, updates

def span_name(node: str, values: dict[str, Any], state: dict[str, Any]) -> str:
    if node == "input_safety": return "guardrail_input"
    if node == "guardrails": return "guardrail_output"
    if node == "answer_cited": return "answer"
    if node == "retrieve": return {"VECTOR": "vector_retrieve", "GRAPH": "graph_retrieve", "HYBRID": "merge"}.get(values.get("route") or state.get("route"), "retrieve")
    return node

def span_metadata(node: str, values: dict[str, Any], state: dict[str, Any], attempt: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {"attempt": attempt, "status": "SUCCESS"}
    if node in {"retrieve", "verify", "retry"}: metadata["result_count"] = len(values.get("results", state.get("results", [])) or [])
    if node == "retrieve": metadata.update({"route": values.get("route") or state.get("route"), "max_hops": values.get("max_hops", state.get("max_hops", 0))})
    if node == "answer_cited": metadata.update({"model": os.getenv("OPENAI_CHAT_MODEL", "unknown"), "prompt_version": PROMPT_VERSION, "evidence_count": len(state.get("evidences", []) or []), "output_length": len(state.get("answer", state.get("final_answer", "")) or "")})
    return metadata

def collect_case(case: dict[str, Any], langfuse: Any, app: Any) -> dict[str, Any]:
    started, visited, result, error = time.perf_counter(), [], {}, None
    # Langfuse 4.x creates a trace from a root observation; ``trace`` is not
    # a supported ``as_type`` value and is downgraded with a warning.
    with langfuse.start_as_current_observation(as_type="span", name=TRACE_NAME, input={"question_id": case["question_id"]}, metadata={"question_id": case["question_id"], "scenario": case["scenario"]}) as trace:
        try:
            result, updates = run_pipeline(case, app); attempt = 1
            for node, values in updates:
                name = span_name(node, values, result)
                with langfuse.start_as_current_observation(as_type="span", name=name, metadata=span_metadata(node, values, result, attempt)):
                    visited.append(name)
                if node == "retry": attempt += 1
        except Exception as exc:
            error, result = safe_error(exc), {"final_status": "ERROR"}
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        final_status = result.get("final_status", "ERROR")
        trace.update(output={"status": final_status, "output_length": len(result.get("final_answer", "") or "")}, metadata={"question_id": case["question_id"], "scenario": case["scenario"], "initial_route": result.get("initial_route"), "final_route": result.get("route"), "route_path": result.get("route_path", []), "vector_result_count": sum(n == "vector_retrieve" for n in visited), "graph_result_count": sum(n == "graph_retrieve" for n in visited), "merged_result_count": len(result.get("results", []) or []), "hop_count": result.get("max_hops", 0), "retry_count": result.get("retry_count", 0), "final_status": final_status, "latency_ms": latency_ms, "visited_nodes": visited, **(error or {})})
    trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
    return {"question_id": case["question_id"], "scenario": case["scenario"], "trace_id": trace_id, "visited_nodes": visited, "final_status": result.get("final_status", "ERROR"), "latency_ms": latency_ms, "error": error}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    langfuse = get_client()
    if not langfuse.auth_check():
        raise SystemExit(
            "Langfuse authentication failed; check LANGFUSE_BASE_URL and keys"
        )
    if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_supervisor import build_app
    app, driver = build_app()
    try:
        cases = load_cases(args.golden); rows = [collect_case(case, langfuse, app) for case in cases]
    finally: driver.close()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"); langfuse.flush()
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("final_status", "ERROR")
        status_counts[status] = status_counts.get(status, 0) + 1

    success_count = status_counts.get("ANSWERED", 0)
    failure_count = len(rows) - success_count
    summary = {
        "golden_set_count": len(rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "status_counts": status_counts,
        "trace_count": sum(bool(row.get("trace_id")) for row in rows),
        "missing_trace_id_count": sum(not bool(row.get("trace_id")) for row in rows),
        "result_jsonl": str(args.output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    langfuse.flush()
    print(json.dumps({**summary, "summary_jsonl": str(args.summary_output)}, ensure_ascii=False))

if __name__ == "__main__": main()
