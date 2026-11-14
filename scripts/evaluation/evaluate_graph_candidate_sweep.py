"""Compare GRAPH candidate pool sizes on fixed GRAPH/Multi-hop gold cases."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import (
    PROJECT_ROOT,
    _hit_row,
    evaluate_multihop_paths,
)
from src.retrievers.router import QuestionRouter
from src.retrievers.graph_retriever import GraphRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30, 50])
    parser.add_argument("--tag", default="p0")
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        raise ValueError("candidate sizes must be positive")
    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    graph_cases = [c for c in cases if expected_route_of(c) == "GRAPH"]
    multi_cases = [c for c in cases if c.get("question_type") == "multi_hop"
                   and expected_route_of(c) in ("GRAPH", "HYBRID")
                   and c.get("expected_relation_ids")]
    selected = {c["case_id"]: c for c in graph_cases + multi_cases}
    router = QuestionRouter()
    graph = GraphRetriever.from_env()
    all_rows = []
    summaries = []
    try:
        for size in args.sizes:
            forced = {}
            latencies: list[float] = []
            case_rows = []
            for case in selected.values():
                decision = router.route_query(case["question"])
                # Force GRAPH by calling the graph retriever directly. Router-derived
                # crop/hop hints are unchanged; no expected IDs/hops enter the query.
                started = time.perf_counter()
                results = graph.search(
                    case["question"], crop=decision.crop,
                    max_hops=decision.max_hops,
                    limit=size,
                    candidate_pool_size=size,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies.append(elapsed_ms)
                diag = dict(graph.last_search_diagnostics)
                top = [_hit_row(i, r) for i, r in enumerate(results[:10], 1)]
                candidates = [_hit_row(i, r) for i, r in enumerate(results, 1)]
                forced[case["case_id"]] = {"case_id": case["case_id"], "route": "GRAPH",
                                            "k": 10, "result_count": len(top), "hits": top}
                gold_e = set(case.get("expected_evidence_ids") or [])
                gold_c = set(case.get("expected_source_chunk_ids") or [])
                hit10 = any(gold_e.intersection(h["evidence_ids"] or []) or
                            gold_c.intersection(h["chunk_ids"] or []) for h in top)
                hitn = any(gold_e.intersection(h["evidence_ids"] or []) or
                           gold_c.intersection(h["chunk_ids"] or []) for h in candidates)
                case_rows.append({"case_id": case["case_id"], "expected_route": expected_route_of(case),
                                  "question_type": case.get("question_type"), "elapsed_ms": round(elapsed_ms, 3),
                                  "router_route": decision.route,
                                  "entity_match_count": diag.get("entity_match_count", 0),
                                  "anchored_start_count": diag.get("anchored_start_count", 0),
                                  "matched_start_entity_ids": diag.get("matched_start_entity_ids", []),
                                  "anchor_strategy": diag.get("anchor_strategy"),
                                  "path_search_attempted": diag.get("path_search_attempted", False),
                                  "failure_stage": diag.get("failure_stage"),
                                  "candidate_count": len(results), "hit_at_10": bool(hit10),
                                  "candidate_evidence_recall": bool(hitn),
                                  "best_rank": next((i for i, h in enumerate(candidates, 1)
                                                     if gold_e.intersection(h["evidence_ids"] or []) or
                                                     gold_c.intersection(h["chunk_ids"] or [])), None),
                                  "paths": candidates})
            mh = [c for c in multi_cases]
            route_details = [{"case_id": c["case_id"], "actual_route": expected_route_of(c)} for c in mh]
            # Evaluate complete-path candidate recall across the entire N pool.
            # Score top-10 path success with unchanged criterion.
            top10_details, top10_summary = evaluate_multihop_paths(
                mh, forced, route_details, PROJECT_ROOT / "data/processed/relations.jsonl")
            # The evaluator sees N results here to measure whether a complete path
            # exists anywhere in the candidate pool rather than in the final top 10.
            candidate_docs = {}
            for row in case_rows:
                candidate_docs[row["case_id"]] = {"case_id": row["case_id"], "route": "GRAPH",
                                                   "k": size, "result_count": size,
                                                   "hits": row["paths"]}
            _, candidate_summary = evaluate_multihop_paths(
                mh, candidate_docs, route_details, PROJECT_ROOT / "data/processed/relations.jsonl")
            actual_routes = [{"case_id": c["case_id"],
                              "actual_route": next(r["router_route"] for r in case_rows if r["case_id"] == c["case_id"])}
                             for c in mh]
            _, integrated_top10_summary = evaluate_multihop_paths(
                mh, forced, actual_routes, PROJECT_ROOT / "data/processed/relations.jsonl")
            _, integrated_candidate_summary = evaluate_multihop_paths(
                mh, candidate_docs, actual_routes, PROJECT_ROOT / "data/processed/relations.jsonl")
            graph_group = [r for r in case_rows if r["expected_route"] == "GRAPH"]
            summary = {
                "candidate_pool_size": size,
                "graph_questions": len(graph_group),
                "graph_hit_at_10": sum(r["hit_at_10"] for r in graph_group),
                "graph_candidate_evidence_recall": sum(r["candidate_evidence_recall"] for r in graph_group),
                "multihop_questions": len(mh),
                "entity_anchor_match_count": sum(r["entity_match_count"] > 0 for r in case_rows),
                "entity_anchor_match_rate": round(sum(r["entity_match_count"] > 0 for r in case_rows) / len(case_rows), 4),
                "anchored_start_count": sum(r["anchored_start_count"] > 0 for r in case_rows),
                "anchor_fallback_count": sum(r["anchor_strategy"] == "all_crop_fallback" for r in case_rows),
                "path_candidate_found_count": sum(r["candidate_count"] > 0 for r in case_rows),
                "path_candidate_found_rate": round(sum(r["candidate_count"] > 0 for r in case_rows) / len(case_rows), 4),
                "multihop_top10_full_success": top10_summary["full_success"],
                "multihop_integrated_top10_full_success": integrated_top10_summary["full_success"],
                "multihop_candidate_full_path_recall": candidate_summary["full_success"],
                "multihop_integrated_candidate_full_path_recall": integrated_candidate_summary["full_success"],
                "multihop_top10_components": top10_summary["component_pass_counts"],
                "multihop_candidate_components": candidate_summary["component_pass_counts"],
                "mean_search_ms": round(sum(latencies) / len(latencies), 3),
                "mean_graph_ms": round(sum(r["elapsed_ms"] for r in case_rows if r["expected_route"] == "GRAPH") / len(graph_group), 3),
                "mean_multihop_ms": None,
            }
            # Patch the group timing without re-reading questions.
            mh_ids = {c["case_id"] for c in mh}
            summary["mean_multihop_ms"] = round(sum(r["elapsed_ms"] for r in case_rows if r["case_id"] in mh_ids) / len(mh), 3)
            summaries.append(summary)
            all_rows.extend({**r, "candidate_pool_size": size} for r in case_rows)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        graph.close()

    out = PROJECT_ROOT / "data/evaluation/reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"graph_candidate_sweep_{args.tag}.json").write_text(
        json.dumps({"mode": "current", "summaries": summaries,
                    "cases": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"graph_candidate_sweep_{args.tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["candidate_pool_size", "case_id", "expected_route", "router_route", "question_type", "elapsed_ms",
                "entity_match_count", "anchored_start_count", "anchor_strategy", "path_search_attempted", "failure_stage",
                "candidate_count", "hit_at_10", "candidate_evidence_recall", "best_rank"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in cols} for row in all_rows)
    print(f"Reports written to {out}")


if __name__ == "__main__":
    main()
