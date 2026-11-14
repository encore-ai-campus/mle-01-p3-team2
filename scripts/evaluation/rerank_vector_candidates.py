"""Offline reranker experiment over the recorded GS056 vector Top-50."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from scripts.evaluation.evaluate_multihop import DEFAULT_GOLDEN_PATH, load_golden_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
DEFAULT_DIAGNOSTIC = REPORT_DIR / "vector_failure_diagnostic_v3.json"
DEFAULT_BASELINE = REPORT_DIR / "top10_results.jsonl"
DEFAULT_OUTPUT = REPORT_DIR / "gs056_reranker_top50.json"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CASE_ID = "GS056"
EXPECTED_CHUNK_ID = "CHK_44B1292296567DF6"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_baseline_top10(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    row = next((row for row in rows if row.get("case_id") == CASE_ID), None)
    if row is None or row.get("route") != "VECTOR" or row.get("k") != 10:
        raise ValueError(f"Missing cached VECTOR Top-10 baseline for {CASE_ID}: {path}")
    return row.get("hits") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--baseline-top10", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=RERANKER_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")

    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    case = next(
        (row for row in diagnostic.get("cases", []) if row.get("case_id") == CASE_ID),
        None,
    )
    if case is None or case.get("expected_chunk_id") != EXPECTED_CHUNK_ID:
        raise ValueError(f"Diagnostic report is missing the expected {CASE_ID} case")
    candidates = case.get("searches_by_k", {}).get("50") or []
    if not candidates:
        raise ValueError("Diagnostic report has no Top-50 candidates; run check_vector first")

    cases = {
        row["case_id"]: row
        for row in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if row.get("case_id") == CASE_ID
    }
    question = cases[CASE_ID]["question"]
    candidate_texts = case.get("candidate_texts_by_chunk_id", {})
    pairs = []
    passages = {}
    for hit in candidates:
        chunk_id = hit.get("chunk_id")
        metadata = candidate_texts.get(chunk_id) or {}
        passage_source = next(
            (
                field
                for field in ("raw_text", "clean_text", "embedding_text")
                if isinstance(metadata.get(field), str) and metadata[field].strip()
            ),
            None,
        )
        if passage_source is None:
            raise ValueError(f"No passage text for Top-50 chunk {chunk_id}")
        passage = metadata[passage_source]
        pairs.append((question, passage))
        passages[chunk_id] = {"text_field_used": passage_source, "text": passage}

    # This process loads only the reranker. Run check_vector.py to completion
    # first so the BGE-M3 embedder process has already exited.
    model = CrossEncoder(
        args.model,
        device="cpu",
        max_length=args.max_length,
        local_files_only=True,
    )
    scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=False,
    )
    reranked = [
        {
            "rerank_score": float(score),
            **hit,
            "passage_text_field": passages[hit["chunk_id"]]["text_field_used"],
        }
        for score, hit in zip(scores, candidates, strict=True)
    ]
    reranked.sort(key=lambda row: (-row["rerank_score"], row["rank"]))
    for rerank_rank, row in enumerate(reranked, 1):
        row["rerank_rank"] = rerank_rank
        row["passage_text"] = passages[row["chunk_id"]]["text"]

    baseline = _load_baseline_top10(args.baseline_top10)
    expected_original = next(
        (row for row in candidates if row.get("chunk_id") == EXPECTED_CHUNK_ID), None
    )
    expected_reranked = next(
        (row for row in reranked if row.get("chunk_id") == EXPECTED_CHUNK_ID), None
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": CASE_ID,
        "question": question,
        "expected_chunk_id": EXPECTED_CHUNK_ID,
        "candidate_source": str(args.diagnostic.resolve()),
        "baseline_source": str(args.baseline_top10.resolve()),
        "baseline_top10_expected_found": any(
            EXPECTED_CHUNK_ID in (hit.get("chunk_ids") or []) for hit in baseline
        ),
        "reranker_model": args.model,
        "reranker_settings": {
            "device": "cpu",
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "candidate_count": len(candidates),
        },
        "expected_chunk_original_rank": (
            expected_original.get("rank") if expected_original else None
        ),
        "expected_chunk_original_vector_score": (
            expected_original.get("score") if expected_original else None
        ),
        "expected_chunk_reranked_rank": (
            expected_reranked.get("rerank_rank") if expected_reranked else None
        ),
        "expected_chunk_reranker_score": (
            expected_reranked.get("rerank_score") if expected_reranked else None
        ),
        "expected_in_top10_after_rerank": bool(
            expected_reranked and expected_reranked["rerank_rank"] <= 10
        ),
        "top50_reranked": reranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{CASE_ID}: original rank={result['expected_chunk_original_rank']}, "
        f"reranked rank={result['expected_chunk_reranked_rank']}, "
        f"in Top-10={result['expected_in_top10_after_rerank']}"
    )
    print(f"Reranker report written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
