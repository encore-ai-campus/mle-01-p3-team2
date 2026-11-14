"""Benchmark VECTOR reranking at candidate-pool sizes 10/20/30/50.

Uses the frozen Top-50 candidate file and golden set. It never connects to
Neo4j or changes vectors, indexes, retrieval code, or evaluation data.
The reranker is loaded once for the whole process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
DEFAULT_CANDIDATES = REPORT_DIR / "vector_reranker_top50_candidates_v1.json"
DEFAULT_OUTPUT = REPORT_DIR / "vector_reranker_candidate_sizes_v1.json"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
K_VALUES = (10, 20, 30, 50)
METRIC_KS = (1, 3, 5, 10)
TEXT_NORMALIZER = re.compile(r"[^0-9a-z가-힣]+", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hit_expected(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    return bool(
        set(case.get("expected_evidence_ids") or []) & set(hit.get("evidence_ids") or [])
        or set(case.get("expected_source_chunk_ids") or [])
        & set(hit.get("chunk_ids") or [hit.get("chunk_id")])
    )


def _metric(rows: dict[str, int | None], k: int) -> dict[str, Any]:
    hits = sum(rank is not None and rank <= k for rank in rows.values())
    return {"hits": hits, "total": len(rows), "rate": hits / len(rows) if rows else None}


def _rank(hits: list[dict[str, Any]], case: dict[str, Any]) -> int | None:
    for pos, hit in enumerate(hits, 1):
        if _hit_expected(hit, case):
            return pos
    return None


def _candidate_data(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(path)
    if report.get("k") != 50 or report.get("model") != "BAAI/bge-m3":
        raise ValueError("Expected the frozen BAAI/bge-m3 Top-50 candidate report")
    cases = {
        row["case_id"]: row
        for row in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if expected_route_of(row) == "VECTOR"
    }
    by_id = {row["case_id"]: row for row in report.get("cases", [])}
    if len(cases) != 19 or cases.keys() != by_id.keys():
        raise ValueError("Candidate report must match all 19 READY VECTOR cases")
    for case_id, case in cases.items():
        if by_id[case_id].get("question") != case["question"]:
            raise ValueError(f"Question mismatch for {case_id}")
    return cases, report


def _rerank_order(hits: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    if strategy == "score_then_vector":
        return sorted(hits, key=lambda h: (-h["rerank_score"], h["vector_rank"]))
    digits = int(strategy.removeprefix("score_bucket_"))
    return sorted(
        hits,
        key=lambda h: (-round(h["rerank_score"], digits), h["vector_rank"]),
    )


def _deduplicate_same_evidence(hits: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse only identical text for the same document and evidence record."""
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for hit in hits:
        meta = metadata.get(hit["chunk_id"]) or {}
        text = meta.get("raw_text") or meta.get("clean_text") or ""
        normalized = TEXT_NORMALIZER.sub("", text.lower())
        evidence_ids = sorted(hit.get("evidence_ids") or [])
        key = (
            str(meta.get("document_id") or ""),
            ",".join(evidence_ids),
            normalized,
            str(hit.get("source") or "vector"),
        )
        current = best.get(key)
        if current is None or (-hit["rerank_score"], hit["vector_rank"]) < (
            -current["rerank_score"], current["vector_rank"]
        ):
            best[key] = hit
    return sorted(best.values(), key=lambda h: (-h["rerank_score"], h["vector_rank"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases, candidate_report = _candidate_data(args.candidates)
    process = psutil.Process(os.getpid())
    samples: list[tuple[float, int]] = []
    stop_sampler = threading.Event()

    def sample_rss() -> None:
        while not stop_sampler.is_set():
            try:
                samples.append((time.perf_counter(), process.memory_info().rss))
            except psutil.Error:
                pass
            stop_sampler.wait(0.05)

    rss_before_model = process.memory_info().rss
    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    load_started = time.perf_counter()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        max_length=384,
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    rss_after_model = process.memory_info().rss

    scored_by_k: dict[str, dict[str, list[dict[str, Any]]]] = {}
    timing_by_k: dict[str, dict[str, float]] = {}
    for k in K_VALUES:
        k_key = str(k)
        scored_by_k[k_key] = {}
        timing_by_k[k_key] = {}
        phase_started = time.perf_counter()
        for case_index, (case_id, case) in enumerate(cases.items(), 1):
            candidate_case = candidate_report["cases"]
            candidate_case = next(row for row in candidate_case if row["case_id"] == case_id)
            vector_hits = candidate_case["vector_hits"][:k]
            text_by_id = candidate_case["candidate_texts_by_chunk_id"]
            pairs = []
            for hit in vector_hits:
                metadata = text_by_id[hit["chunk_id"]]
                passage = next(
                    (metadata.get(field) for field in ("raw_text", "clean_text", "embedding_text")
                     if isinstance(metadata.get(field), str) and metadata[field].strip()),
                    None,
                )
                if passage is None:
                    raise ValueError(f"No reranker text for {case_id}:{hit['chunk_id']}")
                pairs.append((case["question"], passage))
            started = time.perf_counter()
            scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
            elapsed = time.perf_counter() - started
            timing_by_k[k_key][case_id] = elapsed
            scored = [
                {
                    **hit,
                    "vector_rank": hit["rank"],
                    "rerank_score": float(score),
                    "passage_text": pairs[i][1],
                }
                for i, (hit, score) in enumerate(zip(vector_hits, scores, strict=True))
            ]
            scored.sort(key=lambda row: (-row["rerank_score"], row["vector_rank"]))
            for rank, hit in enumerate(scored, 1):
                hit["rank"] = rank
            scored_by_k[k_key][case_id] = scored
            print(f"Top{k} [{case_index}/19] {case_id}: {elapsed:.2f}s", flush=True)
        print(f"Top{k} phase complete in {time.perf_counter() - phase_started:.2f}s", flush=True)

    stop_sampler.set()
    sampler.join(timeout=1)
    samples.append((time.perf_counter(), process.memory_info().rss))
    all_peak = max(value for _, value in samples)

    size_summaries = {}
    per_case: dict[str, Any] = {}
    for k in K_VALUES:
        key = str(k)
        ranks = {case_id: _rank(scored_by_k[key][case_id], case) for case_id, case in cases.items()}
        case_latencies = list(timing_by_k[key].values())
        phase_end = time.perf_counter()
        # Per-pool figures include the loaded model; RSS is reported conservatively
        # as the maximum process RSS observed during the full benchmark.
        size_summaries[key] = {
            "candidate_count": k,
            "candidate_pairs": sum(len(row) for row in scored_by_k[key].values()),
            "hit_at": {str(m): _metric(ranks, m) for m in METRIC_KS},
            "mean_inference_seconds_per_question": sum(case_latencies) / len(case_latencies),
            "p95_inference_seconds_per_question": sorted(case_latencies)[max(0, math.ceil(0.95 * len(case_latencies)) - 1)],
            "total_inference_seconds": sum(case_latencies),
            "max_rss_mib_conservative_full_run": all_peak / (1024 ** 2),
        }
        per_case[key] = {
            case_id: {
                "expected_rank": ranks[case_id],
                "expected_candidate_recall": any(_hit_expected(hit, cases[case_id]) for hit in scored_by_k[key][case_id]),
                "inference_seconds": timing_by_k[key][case_id],
            }
            for case_id in cases
        }

    # Rank-adjustment experiments reuse the measured Top-50 scores: these are
    # ranking-only comparisons, not additional model inference runs.
    variant_results: dict[str, Any] = {}
    top50_case_by_id = {row["case_id"]: row for row in candidate_report["cases"]}
    for strategy in ("score_then_vector", "score_bucket_3", "score_bucket_4", "score_bucket_5", "same_evidence_dedup"):
        variant_ranks: dict[str, int | None] = {}
        gs_details = {}
        for case_id, case in cases.items():
            hits = scored_by_k["50"][case_id]
            if strategy == "same_evidence_dedup":
                hits = _deduplicate_same_evidence(
                    hits, top50_case_by_id[case_id]["candidate_texts_by_chunk_id"]
                )
            else:
                hits = _rerank_order(hits, strategy)
            rank = _rank(hits, case)
            variant_ranks[case_id] = rank
            if case_id in {"GS034", "GS038", "GS056"}:
                gs_details[case_id] = {
                    "rank": rank,
                    "candidate_count_after_strategy": len(hits),
                    "top_10_chunk_ids": [hit["chunk_id"] for hit in hits[:10]],
                }
        variant_results[strategy] = {
            "hit_at": {str(m): _metric(variant_ranks, m) for m in METRIC_KS},
            "focus_cases": gs_details,
            "inference_rerun": False,
        }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(args.candidates.resolve()),
        "candidate_generated_at_utc": candidate_report.get("generated_at_utc"),
        "reranker": {"model": RERANKER_MODEL, "device": "cpu", "max_length": 384, "batch_size": 4, "model_loaded_once": True},
        "model_load_seconds": model_load_seconds,
        "rss_before_model_mib": rss_before_model / (1024 ** 2),
        "rss_after_model_mib": rss_after_model / (1024 ** 2),
        "max_rss_mib_conservative_full_run": all_peak / (1024 ** 2),
        "pool_size_summaries": size_summaries,
        "per_case_by_pool_size": per_case,
        "ranking_strategy_experiments_on_top50_scores": variant_results,
        "limitations": [
            "ANN and embedding-text experiments are separate read-only local PowerShell diagnostics.",
            "Tie and dedup variants reuse the same Top-50 scores; they do not include title-aware reranker inference.",
            "Maximum RSS is a conservative process peak across the entire run, not isolated per pool-size peak.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Candidate-size report written: {args.output.resolve()}")
    for k in K_VALUES:
        row = size_summaries[str(k)]
        print(
            f"Top{k}: " + ", ".join(
                f"Hit@{m}={row['hit_at'][str(m)]['hits']}/19" for m in METRIC_KS
            ) + f", mean={row['mean_inference_seconds_per_question']:.2f}s, "
            f"P95={row['p95_inference_seconds_per_question']:.2f}s"
        )


if __name__ == "__main__":
    main()
