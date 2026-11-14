"""Compare the historical VECTOR Top-10 baseline with a Top-50 reranker run.

The input candidate file is produced separately by collect_vector_top50.py.
This process loads only the cross-encoder, not the BGE-M3 embedder or Neo4j.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from scripts.evaluation.evaluate_multihop import (
    DEFAULT_GOLDEN_PATH,
    expected_route_of,
    load_golden_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
DEFAULT_CANDIDATES = REPORT_DIR / "vector_reranker_top50_candidates_v1.json"
DEFAULT_BASELINE_TOP10 = REPORT_DIR / "top10_results.jsonl"
DEFAULT_ANN_DIAGNOSTIC = REPORT_DIR / "vector_failure_diagnostic_v3.json"
DEFAULT_OUTPUT = REPORT_DIR / "vector_reranker_19case_cost_v2.json"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"
RERANKER_MAX_LENGTH = 384
RERANKER_BATCH_SIZE = 4
EXPECTED_VECTOR_CASE_COUNT = 19
METRIC_KS = (1, 3, 5, 10)
SPECIAL_TRACKS = {
    "GS045": "separate_metadata_embedding_input_experiment",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_ready_vector_cases() -> dict[str, dict[str, Any]]:
    cases = {
        row["case_id"]: row
        for row in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if expected_route_of(row) == "VECTOR"
    }
    if len(cases) != EXPECTED_VECTOR_CASE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_CASE_COUNT} READY VECTOR cases, found {len(cases)}"
        )
    return cases


def _load_baseline(path: Path, cases: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    baseline = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id not in cases:
            continue
        if row.get("route") != "VECTOR" or row.get("k") != 10:
            raise ValueError(f"Baseline route/K mismatch for {case_id}")
        baseline[case_id] = row.get("hits") or []
    missing = cases.keys() - baseline.keys()
    if missing:
        raise ValueError(f"Historical Top-10 results are missing cases: {sorted(missing)}")
    return baseline


def _expected_match(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    expected_evidence = set(case.get("expected_evidence_ids") or [])
    expected_chunks = set(case.get("expected_source_chunk_ids") or [])
    hit_evidence = set(hit.get("evidence_ids") or [])
    hit_chunks = set(hit.get("chunk_ids") or [])
    if hit.get("chunk_id"):
        hit_chunks.add(hit["chunk_id"])
    return bool(expected_evidence & hit_evidence or expected_chunks & hit_chunks)


def _rank_for(hits: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any] | None:
    return next((hit for hit in hits if _expected_match(hit, case)), None)


def _exact_chunk_rank(hits: list[dict[str, Any]], expected_chunk_ids: set[str]) -> int | None:
    return next(
        (hit.get("rank") for hit in hits if expected_chunk_ids & set(hit.get("chunk_ids") or [hit.get("chunk_id")])),
        None,
    )


def _metric(rows: dict[str, dict[str, Any]], rank_key: str, k: int) -> dict[str, Any]:
    hits = sum(
        row[rank_key] is not None and row[rank_key] <= k
        for row in rows.values()
    )
    total = len(rows)
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def _rank_movement(before: int | None, after: int | None) -> dict[str, Any]:
    if before is None or after is None:
        return {"delta_positions": None, "direction": "not_comparable"}
    delta = before - after
    direction = "improved" if delta > 0 else "dropped" if delta < 0 else "unchanged"
    return {"delta_positions": delta, "direction": direction}


def _load_ann_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = {}
    for row in report.get("cases", []):
        if row.get("case_id") not in {"GS039", "GS065", "GS045"}:
            continue
        selected[row["case_id"]] = {
            "diagnosis_signal": row.get("diagnosis_signal"),
            "k10_expected_rank": (row.get("expected_chunk_rank_and_score_by_k") or {}).get("10"),
            "k100_expected_rank": (row.get("expected_chunk_rank_and_score_by_k") or {}).get("100"),
            "direct_similarity": row.get("direct_similarity"),
            "baseline_evaluation_comparison": row.get("baseline_evaluation_comparison"),
            "key_term_audit": row.get("key_term_audit"),
            "expected_embedding_text": (row.get("expected_chunk") or {}).get("embedding_text"),
            "expected_raw_text": (row.get("expected_chunk") or {}).get("raw_text"),
            "expected_title": (row.get("expected_chunk") or {}).get("title"),
        }
    return selected


def _load_candidates(path: Path, cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("k") != 50:
        raise ValueError(f"Candidate report must use K=50, got {report.get('k')}")
    if report.get("model") != "BAAI/bge-m3" or report.get("index", {}).get("name") != "chunk_embedding":
        raise ValueError("Candidate report does not match the VECTOR baseline model/index")
    by_id = {row["case_id"]: row for row in report.get("cases", [])}
    missing = cases.keys() - by_id.keys()
    extra = by_id.keys() - cases.keys()
    if missing or extra:
        raise ValueError(f"Candidate case mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    for case_id, case in cases.items():
        row = by_id[case_id]
        if row.get("question") != case["question"]:
            raise ValueError(f"Question mismatch for {case_id}")
        hits = row.get("vector_hits") or []
        if len(hits) > 50 or any(hit.get("rank") != rank for rank, hit in enumerate(hits, 1)):
            raise ValueError(f"Invalid Top-50 ranks for {case_id}")
    return {"generated_at_utc": report.get("generated_at_utc"), "cases": by_id, "index": report.get("index")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--baseline-top10", type=Path, default=DEFAULT_BASELINE_TOP10)
    parser.add_argument("--ann-diagnostic", type=Path, default=DEFAULT_ANN_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = _load_ready_vector_cases()
    baseline = _load_baseline(args.baseline_top10, cases)
    candidate_report = _load_candidates(args.candidates, cases)
    ann_diagnostics = _load_ann_diagnostics(args.ann_diagnostic)

    scored_cases: dict[str, dict[str, Any]] = {}
    process = psutil.Process(os.getpid())
    memory_samples: list[int] = []
    sampling = threading.Event()

    def sample_memory() -> None:
        while not sampling.is_set():
            try:
                memory_samples.append(process.memory_info().rss)
            except psutil.Error:
                pass
            sampling.wait(0.05)

    rss_before_import = process.memory_info().rss
    cpu_before_load = process.cpu_times()
    load_started = time.perf_counter()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    # Import inside the measured window so the reported peak includes model stack setup.
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=RERANKER_DEVICE,
        max_length=RERANKER_MAX_LENGTH,
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    rss_after_load = process.memory_info().rss
    cpu_after_load = process.cpu_times()
    inference_wall_seconds = 0.0
    inference_cpu_seconds = 0.0
    inference_case_seconds: dict[str, float] = {}
    total_pairs = 0
    for case_id, case in cases.items():
        candidate_case = candidate_report["cases"][case_id]
        vector_hits = candidate_case.get("vector_hits") or []
        text_by_id = candidate_case.get("candidate_texts_by_chunk_id") or {}
        pairs = []
        passage_field_by_id = {}
        for hit in vector_hits:
            chunk_id = hit["chunk_id"]
            metadata = text_by_id.get(chunk_id) or {}
            passage_field = next(
                (
                    field
                    for field in ("raw_text", "clean_text", "embedding_text")
                    if isinstance(metadata.get(field), str) and metadata[field].strip()
                ),
                None,
            )
            if passage_field is None:
                raise ValueError(f"No text for {case_id} candidate {chunk_id}")
            pairs.append((case["question"], metadata[passage_field]))
            passage_field_by_id[chunk_id] = passage_field

        predict_wall_started = time.perf_counter()
        predict_cpu_before = process.cpu_times()
        scores = reranker.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        ) if pairs else []
        predict_wall = time.perf_counter() - predict_wall_started
        predict_cpu_after = process.cpu_times()
        predict_cpu = (
            predict_cpu_after.user + predict_cpu_after.system
            - predict_cpu_before.user - predict_cpu_before.system
        )
        inference_case_seconds[case_id] = predict_wall
        inference_wall_seconds += predict_wall
        inference_cpu_seconds += predict_cpu
        total_pairs += len(pairs)
        reranked = [
            {
                **hit,
                "vector_rank": hit["rank"],
                "vector_score": hit.get("score"),
                "rerank_score": float(score),
                "passage_text_field": passage_field_by_id[hit["chunk_id"]],
                "passage_text": text_by_id[hit["chunk_id"]][passage_field_by_id[hit["chunk_id"]]],
            }
            for hit, score in zip(vector_hits, scores, strict=True)
        ]
        reranked.sort(key=lambda row: (-row["rerank_score"], row["vector_rank"]))
        for rerank_rank, hit in enumerate(reranked, 1):
            hit["rank"] = rerank_rank

        baseline_hits = baseline[case_id]
        before = _rank_for(vector_hits, case)
        after = _rank_for(reranked, case)
        historic = _rank_for(baseline_hits, case)
        expected_chunk_ids = set(case.get("expected_source_chunk_ids") or [])
        scored_cases[case_id] = {
            "case_id": case_id,
            "question": case["question"],
            "expected_evidence_ids": case.get("expected_evidence_ids") or [],
            "expected_chunk_ids": sorted(expected_chunk_ids),
            "historic_top10_relevant_rank": historic.get("rank") if historic else None,
            "current_top50_vector_relevant_rank": before.get("rank") if before else None,
            "reranked_top50_relevant_rank": after.get("rank") if after else None,
            "current_top50_expected_chunk_rank": _exact_chunk_rank(vector_hits, expected_chunk_ids),
            "reranked_top50_expected_chunk_rank": _exact_chunk_rank(reranked, expected_chunk_ids),
            "rank_movement_current_vector_to_reranked": _rank_movement(
                before.get("rank") if before else None,
                after.get("rank") if after else None,
            ),
            "historic_top10_hit": historic is not None and historic.get("rank", 999) <= 10,
            "current_top50_vector_hit_at": {
                str(k): before is not None and before.get("rank", 999) <= k
                for k in METRIC_KS
            },
            "reranked_top50_hit_at": {
                str(k): after is not None and after.get("rank", 999) <= k
                for k in METRIC_KS
            },
            "candidate_pool_contains_expected": _exact_chunk_rank(vector_hits, expected_chunk_ids) is not None,
            "follow_up_track": SPECIAL_TRACKS.get(case_id),
            "ann_diagnostic": ann_diagnostics.get(case_id),
            "reranked_hits": reranked,
        }
        print(
            f"[{len(scored_cases)}/{len(cases)}] {case_id}: "
            f"{len(pairs)} pairs in {predict_wall:.2f}s",
            flush=True,
        )

    baseline_rank_rows = {
        case_id: {"rank": row["historic_top10_relevant_rank"]}
        for case_id, row in scored_cases.items()
    }
    vector_rank_rows = {
        case_id: {"rank": row["current_top50_vector_relevant_rank"]}
        for case_id, row in scored_cases.items()
    }
    reranked_rank_rows = {
        case_id: {"rank": row["reranked_top50_relevant_rank"]}
        for case_id, row in scored_cases.items()
    }
    summary = {
        "question_count": len(scored_cases),
        "historical_vector_top10": {
            f"hit_at_{k}": _metric(baseline_rank_rows, "rank", k)
            for k in METRIC_KS
        },
        "current_vector_top50": {
            f"hit_at_{k}": _metric(vector_rank_rows, "rank", k)
            for k in METRIC_KS
        },
        "reranked_top50": {
            f"hit_at_{k}": _metric(reranked_rank_rows, "rank", k)
            for k in METRIC_KS
        },
        "expected_chunk_candidate_recall_at_50": sum(
            row["candidate_pool_contains_expected"] for row in scored_cases.values()
        ),
        "rank_movement_counts": {
            direction: sum(
                row["rank_movement_current_vector_to_reranked"]["direction"] == direction
                for row in scored_cases.values()
            )
            for direction in ("improved", "dropped", "unchanged", "not_comparable")
        },
    }
    sampling.set()
    sampler.join(timeout=1)
    memory_samples.append(process.memory_info().rss)
    peak_rss = max(memory_samples, default=process.memory_info().rss)
    cpu_after_all = process.cpu_times()
    load_cpu_seconds = (
        cpu_after_load.user + cpu_after_load.system
        - cpu_before_load.user - cpu_before_load.system
    )
    postload_rss_peak = max(memory_samples, default=rss_after_load)
    cost = {
        "scope": "offline_reranker_only_no_neo4j_or_embedding_search",
        "process_id": os.getpid(),
        "model_load_wall_seconds": model_load_seconds,
        "model_load_process_cpu_seconds": load_cpu_seconds,
        "inference_wall_seconds_total": inference_wall_seconds,
        "inference_process_cpu_seconds_total": inference_cpu_seconds,
        "inference_wall_seconds_per_case": inference_case_seconds,
        "case_count": len(cases),
        "candidate_pair_count": total_pairs,
        "pairs_per_second": total_pairs / inference_wall_seconds if inference_wall_seconds else None,
        "rss_bytes_before_model_import": rss_before_import,
        "rss_bytes_after_model_load": rss_after_load,
        "rss_bytes_peak_during_model_load_and_inference": peak_rss,
        "rss_bytes_peak_delta_from_before_import": peak_rss - rss_before_import,
        "rss_bytes_inference_peak_delta_from_after_model_load": max(0, postload_rss_peak - rss_after_load),
        "rss_sampling_interval_ms": 50,
        "process_cpu_seconds_total_at_end": cpu_after_all.user + cpu_after_all.system,
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_top10_file": str(args.baseline_top10.resolve()),
        "candidate_file": str(args.candidates.resolve()),
        "candidate_generated_at_utc": candidate_report["generated_at_utc"],
        "ann_diagnostic_file": str(args.ann_diagnostic.resolve()) if args.ann_diagnostic.is_file() else None,
        "reranker": {
            "model": RERANKER_MODEL,
            "device": RERANKER_DEVICE,
            "max_length": RERANKER_MAX_LENGTH,
            "batch_size": RERANKER_BATCH_SIZE,
            "local_files_only": True,
        },
        "comparison_note": (
            "Primary before/after comparison uses the same current Top-50 candidate pool. "
            "Historical Top-10 is reported separately and is not treated as the same run."
        ),
        "summary": summary,
        "cost_measurement": cost,
        "cases": list(scored_cases.values()),
        "separate_follow_up": {
            "GS045": "Metadata-enriched embedding input is tracked separately; no text, vector, or index was changed in this reranker experiment."
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Reranker report written: {args.output.resolve()}")
    for label, key in (
        ("Historical Top-10", "historical_vector_top10"),
        ("Current vector Top-50", "current_vector_top50"),
        ("Reranked Top-50", "reranked_top50"),
    ):
        cells = ", ".join(
            f"Hit@{k}={summary[key][f'hit_at_{k}']['hits']}/{summary[key][f'hit_at_{k}']['total']}"
            for k in METRIC_KS
        )
        print(f"{label}: {cells}")
    print(f"Expected chunk in Top-50: {summary['expected_chunk_candidate_recall_at_50']}/{len(scored_cases)}")
    print(f"Rank movement: {summary['rank_movement_counts']}")
    print(
        "Cost: "
        f"load={cost['model_load_wall_seconds']:.2f}s, "
        f"inference={cost['inference_wall_seconds_total']:.2f}s/"
        f"{cost['candidate_pair_count']} pairs, "
        f"peak RSS={cost['rss_bytes_peak_during_model_load_and_inference'] / (1024 ** 2):.1f} MiB"
    )


if __name__ == "__main__":
    main()
