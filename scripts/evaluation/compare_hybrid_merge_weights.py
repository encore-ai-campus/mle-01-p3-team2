"""Offline RRF-weight sweep using the same stored GRAPH/VECTOR candidates."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import _hit_row, evaluate_multihop_paths, PROJECT_ROOT
from src.retrievers.hybrid_retriever import merge_results
from src.retrievers.search_result import SearchResult

REPORTS = PROJECT_ROOT / "data/evaluation/reports"
GRAPH_WEIGHTS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)


def load_results(record: dict, source_key: str) -> list[SearchResult]:
    return [SearchResult(**hit["raw_result"]) for hit in record[source_key]]


def hit_at_10(case: dict, rows: list[dict]) -> bool:
    evidence = set(case.get("expected_evidence_ids") or [])
    chunks = set(case.get("expected_source_chunk_ids") or [])
    return any(evidence.intersection(row.get("evidence_ids") or [])
               or chunks.intersection(row.get("chunk_ids") or []) for row in rows[:10])


def path_key(row: dict) -> tuple:
    return (tuple(node.get("entity_id") for node in row.get("path_nodes") or []),
            tuple(row.get("relation_ids") or []))


def main() -> None:
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    hybrid_cases = [case for case in cases if expected_route_of(case) == "HYBRID"]
    multihop = [case for case in hybrid_cases if case.get("question_type") == "multi_hop"
                and case.get("expected_relation_ids")]
    fixed = json.loads((REPORTS / "hybrid_merge_fixed_candidates_after.json").read_text(encoding="utf-8"))
    before_fixed = json.loads(
        (REPORTS / "hybrid_merge_fixed_candidates_before.json").read_text(encoding="utf-8")
    )
    records = fixed["cases"]["HYBRID"]
    live_after = fixed
    live_rows = list(fixed["cases"]["HYBRID"].values())
    live_latencies = [row["elapsed_ms"] for row in live_rows]
    live_hit_count = fixed["summary"]["HYBRID"]["hit_at_10"]
    summaries = []
    case_ranks: list[dict] = []
    for graph_weight in GRAPH_WEIGHTS:
        docs: dict[str, dict] = {}
        merged_rows: dict[str, list[dict]] = {}
        hit_count = 0
        preserved = 0
        graph_path_total = 0
        start = time.perf_counter()
        for case in hybrid_cases:
            record = records[case["case_id"]]
            graph_results = load_results(record, "graph_hits")
            vector_results = load_results(record, "vector_hits")
            results = merge_results(
                graph_results, vector_results, limit=10,
                graph_weight=graph_weight, vector_weight=1.0,
                query=case["question"],
            )
            rows = [_hit_row(rank, result) for rank, result in enumerate(results, 1)]
            merged_rows[case["case_id"]] = rows
            docs[case["case_id"]] = {
                "case_id": case["case_id"], "route": "HYBRID", "k": 10,
                "result_count": len(rows), "hits": rows,
            }
            hit_count += int(hit_at_10(case, rows))
            graph_keys = {path_key(row) for row in record["graph_hits"] if row.get("path_edges")}
            merged_keys = {path_key(row) for row in rows if row.get("source") == "graph"}
            graph_path_total += len(graph_keys)
            preserved += len(graph_keys & merged_keys)
        merge_ms_per_case = (time.perf_counter() - start) * 1000 / max(len(hybrid_cases), 1)
        routes = [{"case_id": case["case_id"], "actual_route": "HYBRID"} for case in multihop]
        details, mh_summary = evaluate_multihop_paths(
            multihop, docs, routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        detail_by_id = {row["case_id"]: row for row in details}
        graph_only_success = 0
        for case in multihop:
            record = records[case["case_id"]]
            graph_doc = {"case_id": case["case_id"], "route": "GRAPH", "k": 10,
                         "result_count": len(record["graph_hits"]), "hits": record["graph_hits"]}
            graph_detail, _ = evaluate_multihop_paths(
                [case], {case["case_id"]: graph_doc},
                [{"case_id": case["case_id"], "actual_route": expected_route_of(case)}],
                PROJECT_ROOT / "data/processed/relations.jsonl",
            )
            graph_success = graph_detail[0]["full_success"]
            graph_only_success += int(graph_success)
            case_ranks.append({
                "graph_weight": graph_weight, "case_id": case["case_id"],
                "graph_only_success": graph_success,
                "graph_only_path_rank": graph_detail[0].get("actual_path_rank"),
                "hybrid_success": detail_by_id[case["case_id"]]["full_success"],
                "hybrid_path_rank": detail_by_id[case["case_id"]].get("actual_path_rank"),
                "hybrid_hit_at_10": hit_at_10(case, merged_rows[case["case_id"]]),
                "vector_candidate_hit_at_10": hit_at_10(case, records[case["case_id"]]["vector_hits"]),
                "vector_hit_lost_after_merge": (
                    hit_at_10(case, records[case["case_id"]]["vector_hits"])
                    and not hit_at_10(case, merged_rows[case["case_id"]])
                ),
                "hybrid_graph_count": sum(row.get("source") == "graph" for row in merged_rows[case["case_id"]]),
                "hybrid_vector_count": sum(row.get("source") == "vector" for row in merged_rows[case["case_id"]]),
                "graph_gold_path_lost": graph_success and not detail_by_id[case["case_id"]]["full_success"],
                "graph_path_preservation": sum(
                    path_key(row) in {path_key(hit) for hit in records[case["case_id"]]["graph_hits"]
                                      if hit.get("path_edges")}
                    for row in merged_rows[case["case_id"]] if row.get("source") == "graph"
                ),
            })
        summaries.append({
            "graph_weight": graph_weight, "vector_weight": 1.0,
            "hybrid_hit_at_10": hit_count, "hybrid_questions": len(hybrid_cases),
            "multihop_full_success_at_10": mh_summary["full_success"],
            "multihop_questions": len(multihop),
            "graph_only_multihop_success_at_10": graph_only_success,
            "gold_graph_paths_preserved": sum(
                row["graph_only_success"] and row["hybrid_success"]
                for row in case_ranks if row["graph_weight"] == graph_weight
            ),
            "vector_gold_hits_lost_after_merge": sum(
                row["vector_hit_lost_after_merge"]
                for row in case_ranks if row["graph_weight"] == graph_weight
            ),
            "all_graph_top10_paths_preserved": preserved,
            "all_graph_top10_path_count": graph_path_total,
            "path_preservation_rate": round(preserved / graph_path_total, 4) if graph_path_total else None,
            "merge_ms_per_hybrid_case": round(merge_ms_per_case, 3),
        })

    output = {
        "candidate_inputs_identical": True,
        "candidate_source": "hybrid_merge_fixed_candidates_after.json (same captured inputs for all weights)",
        "gold_ids_used_by_merge": False,
        "before_live": before_fixed["summary"],
        "after_live": fixed["summary"],
        "after_live_multihop": {
            "hybrid_hit_at_10": live_hit_count,
            "multi_hop_full_success_at_10": live_after["multihop_integrated"]["full_success"],
            "questions": len(live_rows),
            "mean_search_ms": round(sum(live_latencies) / len(live_latencies), 3),
            "median_search_ms": sorted(live_latencies)[len(live_latencies) // 2],
            "min_search_ms": min(live_latencies),
            "max_search_ms": max(live_latencies),
        },
        "weight_sweep": summaries,
        "case_rank_changes": case_ranks,
    }
    target = REPORTS / "hybrid_merge_weight_comparison_after.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_target = REPORTS / "hybrid_merge_weight_case_ranks_after.csv"
    with csv_target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_ranks[0]))
        writer.writeheader()
        writer.writerows(case_ranks)
    print(json.dumps({"baseline_live": fixed["summary"], "weight_sweep": summaries},
                     ensure_ascii=False, indent=2))
    print(f"Reports: {target}, {csv_target}")


if __name__ == "__main__":
    main()
