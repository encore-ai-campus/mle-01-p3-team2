"""Evaluate the real Supervisor pipeline with RAGAS and ID-based checks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `python -m scripts.evaluation...`과 `python scripts/evaluation/...`을
# 모두 지원하려면 루트가 import 경로에 있어야 한다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
    "context_relevance",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _result_ids(results: Sequence[Any], field: str, top_k: int) -> list[str]:
    return _unique(
        identifier
        for result in list(results)[:top_k]
        for identifier in (getattr(result, field, None) or [])
    )


def _evidence_contexts(evidences: Sequence[Any]) -> list[str]:
    return _unique(
        getattr(evidence, "quote", "")
        for evidence in evidences
        if isinstance(getattr(evidence, "quote", ""), str)
    )


def _hit_at_k(expected: Sequence[str], retrieved: Sequence[str]) -> float | None:
    expected_set = set(expected or [])
    if not expected_set:
        return None
    return float(bool(expected_set.intersection(retrieved)))


def state_to_record(
    case: dict[str, Any],
    state: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Convert one Supervisor terminal state into an evaluation record."""
    results = list(state.get("retrieved_results") or state.get("results") or [])
    chunk_ids = _result_ids(results, "chunk_ids", top_k)
    evidence_ids = _result_ids(results, "evidence_ids", top_k)
    route_path = list(state.get("route_path") or [])
    initial_route = state.get("initial_route") or (route_path[0] if route_path else None)
    actual_route = state.get("route") or (route_path[-1] if route_path else None)
    expected_behavior = case.get("expected_behavior")
    final_status = state.get("final_status") or "UNKNOWN"
    is_reject = expected_behavior == "REJECT"

    structural_pass = (
        final_status == "REJECTED" if is_reject else final_status == "ANSWERED"
    )

    return {
        "case_id": case.get("case_id"),
        "question": case.get("question", ""),
        "question_type": case.get("question_type"),
        "difficulty": case.get("difficulty"),
        "expected_behavior": expected_behavior,
        "run_status": "OK",
        "error_type": None,
        "error_message": None,
        "initial_route": initial_route,
        "actual_route": actual_route,
        "route_path": route_path,
        "fallback": len(route_path) > 1,
        "retry_count": state.get("retry_count", 0),
        "final_status": final_status,
        "response": state.get("final_answer") or "",

        # 보류·차단된 답변의 원문. 어떤 주장이 걸렸는지 재실행 없이
        # 확인하기 위해 남긴다.
        "draft_answer": state.get("answer") or "",
        "citation_status": state.get("citation_status"),
        "evidence_status": state.get("evidence_status"),
        # 보류·차단이 어느 레일에서 났는지 결과만 보고 구분하기 위한 필드
        "guardrail_decision": state.get("guardrail_decision"),
        "guardrail_reason": state.get("guardrail_reason"),
        "judge_severity": (
            (state.get("guardrail_details") or {}).get("judge_severity")
        ),
        # 판정 사유가 어느 검증 담당인지. 범위 밖 분류는 보류로 쓰지 않는다.
        "judge_category": (
            (state.get("guardrail_details") or {}).get("judge_category")
        ),
        "execution_decision": state.get("execution_decision"),
        "execution_reason": state.get("execution_reason"),
        "execution_record_count": len(
            state.get("execution_records") or []
        ),
        "warnings": list(state.get("warnings") or []),
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_evidence_ids": evidence_ids,
        "retrieved_contexts": _evidence_contexts(state.get("evidences") or []),
        "chunk_hit_at_k": _hit_at_k(
            case.get("expected_source_chunk_ids") or [], chunk_ids
        ),
        "evidence_hit_at_k": _hit_at_k(
            case.get("expected_evidence_ids") or [], evidence_ids
        ),
        "structural_pass": structural_pass,
        "ragas_eligible": not is_reject,
    }


def _error_record(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "question": case.get("question", ""),
        "question_type": case.get("question_type"),
        "difficulty": case.get("difficulty"),
        "expected_behavior": case.get("expected_behavior"),
        "run_status": "ERROR",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "initial_route": None,
        "actual_route": None,
        "route_path": [],
        "fallback": False,
        "retry_count": 0,
        "final_status": "ERROR",
        "response": "",
        "draft_answer": "",
        "citation_status": None,
        "evidence_status": None,
        "guardrail_decision": None,
        "guardrail_reason": None,
        "judge_severity": None,
        "judge_category": None,
        "execution_decision": None,
        "execution_reason": None,
        "execution_record_count": 0,
        "warnings": [],
        "retrieved_chunk_ids": [],
        "retrieved_evidence_ids": [],
        "retrieved_contexts": [],
        "chunk_hit_at_k": 0.0,
        "evidence_hit_at_k": 0.0,
        "structural_pass": False,
        "ragas_eligible": False,
    }


def run_supervisor_cases(
    app: Any,
    cases: Sequence[dict[str, Any]],
    *,
    top_k: int = 5,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run cases without ever passing golden references into the Supervisor."""
    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            state = app.invoke({"question": case.get("question", "")})
            record = state_to_record(case, state, top_k=top_k)
        except Exception as exc:  # Preserve the remaining expensive evaluation run.
            record = _error_record(case, exc)
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def _context_text(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("clean_text")
        or chunk.get("raw_text")
        or chunk.get("embedding_text")
        or ""
    )


def build_ragas_rows(
    records: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join references after Supervisor execution, preventing evaluation leakage."""
    cases_by_id = {case.get("case_id"): case for case in cases}
    chunks_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
    rows: list[dict[str, Any]] = []

    for record in records:
        if not record.get("ragas_eligible") or record.get("run_status", "OK") != "OK":
            continue
        case = cases_by_id.get(record.get("case_id"), {})
        reference_ids = case.get("expected_source_chunk_ids") or []
        rows.append(
            {
                "case_id": record.get("case_id"),
                "user_input": record.get("question", ""),
                "response": record.get("response", ""),
                "retrieved_contexts": list(record.get("retrieved_contexts") or []),
                "reference": " ".join(case.get("expected_facts") or []),
                "reference_contexts": [
                    _context_text(chunks_by_id[chunk_id])
                    for chunk_id in reference_ids
                    if chunk_id in chunks_by_id
                ],
            }
        )
    return rows


def evaluate_ragas_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    embedding_model: str,
    timeout: int,
    max_workers: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    """Run the six RAGAS metrics and return scores keyed by case_id."""
    if not rows:
        return []

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
    )
    from ragas.run_config import RunConfig

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            f"{PROJECT_ROOT / '.env'}에 OPENAI_API_KEY=... 를 넣거나 환경변수로 export 하세요."
        )

    payload = [{key: value for key, value in row.items() if key != "case_id"} for row in rows]
    dataset = Dataset.from_list(payload)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=embedding_model)
    )
    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=max_retries,
        log_tenacity=True,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
            ContextPrecision(),
            ContextRecall(),
            ContextRelevance(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        # A partial RAGAS run must not be reported as a successful evaluation.
        # With the default False, failed metric jobs are silently converted to NaN.
        raise_exceptions=True,
    )
    frame = result.to_pandas()
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        values = frame.iloc[index].to_dict()
        values["context_relevance"] = values.pop(
            "nv_context_relevance", values.get("context_relevance")
        )
        scores = {name: _finite_float(values.get(name)) for name in RAGAS_METRICS}
        scored.append({"case_id": row["case_id"], **scores})
    _require_complete_scores(scored)
    return scored


def _require_complete_scores(scores: Sequence[dict[str, Any]]) -> None:
    """Reject partial RAGAS output instead of persisting misleading averages."""
    missing = {
        str(row.get("case_id")): [
            name for name in RAGAS_METRICS if _finite_float(row.get(name)) is None
        ]
        for row in scores
    }
    missing = {case_id: metrics for case_id, metrics in missing.items() if metrics}
    if missing:
        preview = ", ".join(
            f"{case_id}={metrics}" for case_id, metrics in list(missing.items())[:10]
        )
        raise RuntimeError(
            "RAGAS 평가에 누락된 지표가 있어 결과를 저장하지 않습니다: "
            f"{preview}"
        )


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def merge_ragas_scores(
    records: Sequence[dict[str, Any]],
    scores: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores_by_id = {row["case_id"]: row for row in scores}
    return [
        {
            **record,
            "ragas": {
                name: scores_by_id.get(record.get("case_id"), {}).get(name)
                for name in RAGAS_METRICS
            },
        }
        for record in records
    ]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def average(values: Iterable[Any]) -> float | None:
        numbers = [number for value in values if (number := _finite_float(value)) is not None]
        return sum(numbers) / len(numbers) if numbers else None

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("question_type"))].append(record)

    # 답변하지 못한 사례를 레일별로 모아 보류 원인을 바로 볼 수 있게 한다.
    withheld = [
        record
        for record in records
        if record.get("final_status") not in ("ANSWERED", "REJECTED")
    ]

    return {
        "total_cases": len(records),
        "successful_runs": sum(record.get("run_status") == "OK" for record in records),
        "withheld_cases": len(withheld),
        "withheld_by_execution_decision": dict(
            Counter(
                str(record.get("execution_decision"))
                for record in withheld
            )
        ),
        "withheld_by_execution_reason": dict(
            Counter(
                str(record.get("execution_reason"))
                for record in withheld
                if record.get("execution_reason")
            )
        ),
        "withheld_by_guardrail_reason": dict(
            Counter(
                str(record.get("guardrail_reason"))
                for record in withheld
                if record.get("guardrail_reason")
            )
        ),
        "error_counts": dict(Counter(record.get("error_type") for record in records if record.get("error_type"))),
        "ragas_evaluated_cases": sum(
            any(value is not None for value in (record.get("ragas") or {}).values())
            for record in records
        ),
        "ragas_metrics": {
            name: average((record.get("ragas") or {}).get(name) for record in records)
            for name in RAGAS_METRICS
        },
        "chunk_hit_at_k": average(record.get("chunk_hit_at_k") for record in records),
        "evidence_hit_at_k": average(record.get("evidence_hit_at_k") for record in records),
        "structural_pass_rate": average(float(record.get("structural_pass", False)) for record in records),
        "by_question_type": {
            key: {
                "count": len(group),
                "structural_pass_rate": average(float(row.get("structural_pass", False)) for row in group),
                "answer_correctness": average((row.get("ragas") or {}).get("answer_correctness") for row in group),
            }
            for key, group in sorted(by_type.items())
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--golden-set", type=Path, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "collect", "score"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-ids",
        default="",
        help=(
            "쉼표로 구분한 케이스 ID만 실행한다. 수정한 레일에 걸린 "
            "케이스만 다시 돌려 LLM 호출 비용을 줄일 때 쓴다."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1"))
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=900,
        help="RAGAS metric job timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=4,
        help="Maximum concurrent RAGAS metric jobs (default: 4)",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=10,
        help="Maximum retries for transient RAGAS/API errors (default: 10)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("ragas_timeout", "ragas_max_workers", "ragas_max_retries"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    root = args.root.resolve()
    golden_path = args.golden_set or root / "data/evaluation/golden_set_final.jsonl"
    chunk_path = args.chunks or root / "data/processed/chunks.jsonl"
    output_dir = args.output_dir or root / "data/evaluation/ragas_result"
    raw_path = output_dir / "supervisor_runs.jsonl"
    scored_path = output_dir / "ragas_results.jsonl"
    summary_path = output_dir / "ragas_summary.json"

    cases = [case for case in read_jsonl(golden_path) if case.get("status") == "READY"]

    wanted_ids = {
        case_id.strip()
        for case_id in args.case_ids.split(",")
        if case_id.strip()
    }

    if wanted_ids:
        cases = [case for case in cases if case.get("case_id") in wanted_ids]

        missing = wanted_ids - {case.get("case_id") for case in cases}

        if missing:
            raise ValueError(
                "Golden Set에 없는 케이스 ID: "
                + ", ".join(sorted(missing))
            )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        cases = cases[: args.limit]

    if args.stage in {"all", "collect"}:
        from scripts.run_supervisor import build_app

        output_dir.mkdir(parents=True, exist_ok=True)
        app, driver = build_app()
        try:
            with raw_path.open("w", encoding="utf-8") as checkpoint:
                def save_checkpoint(record: dict[str, Any]) -> None:
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()

                records = run_supervisor_cases(
                    app, cases, top_k=args.top_k, on_record=save_checkpoint
                )
        finally:
            driver.close()
    else:
        if not raw_path.exists():
            raise FileNotFoundError(f"Supervisor 실행 결과가 없습니다: {raw_path}")
        records = read_jsonl(raw_path)

    if args.stage == "collect":
        print(f"saved: {raw_path} ({len(records)} cases)")
        return 0

    chunks = read_jsonl(chunk_path)
    ragas_rows = build_ragas_rows(records, cases, chunks)
    scores = evaluate_ragas_rows(
        ragas_rows,
        model_name=args.model,
        embedding_model=args.embedding_model,
        timeout=args.ragas_timeout,
        max_workers=args.ragas_max_workers,
        max_retries=args.ragas_max_retries,
    )
    merged = merge_ragas_scores(records, scores)
    write_jsonl(scored_path, merged)
    summary = summarize(merged)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {scored_path}")
    print(f"saved: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
