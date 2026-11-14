"""Diagnose HYBRID merge losses and compare route-specific retrieval hits."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import evaluate_multihop_paths, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "data/evaluation/reports"


def docs_from(records: dict, key: str) -> dict:
    return {
        case_id: {"case_id": case_id, "route": "GRAPH" if key == "graph_hits" else "HYBRID",
                  "k": 10, "result_count": len(row[key]), "hits": row[key]}
        for case_id, row in records.items()
    }


def path_key(hit: dict) -> tuple:
    nodes = tuple(node.get("entity_id") for node in hit.get("path_nodes") or [])
    relations = tuple(hit.get("relation_ids") or [])
    return nodes, relations


def main() -> None:
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    mh = [case for case in cases if case.get("question_type") == "multi_hop"
          and expected_route_of(case) in {"GRAPH", "HYBRID"}
          and case.get("expected_relation_ids")]
    fixed = json.loads((REPORTS / "hybrid_merge_fixed_candidates_after.json").read_text(encoding="utf-8"))
    hybrid = fixed["cases"]["HYBRID"]
    graph_docs = docs_from(hybrid, "graph_hits")
    merged_docs = docs_from(hybrid, "merged_hits")
    expected_routes = [{"case_id": case["case_id"], "actual_route": expected_route_of(case)} for case in mh]
    graph_details, graph_summary = evaluate_multihop_paths(
        mh, graph_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl")
    merged_details, merged_summary = evaluate_multihop_paths(
        mh, merged_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl")
    graph_by_id = {row["case_id"]: row for row in graph_details}
    merged_by_id = {row["case_id"]: row for row in merged_details}

    sweep = json.loads((REPORTS / "graph_candidate_sweep_ranking-final.json").read_text(encoding="utf-8"))
    pool30 = {row["case_id"]: row for row in sweep["cases"] if row["candidate_pool_size"] == 30}
    pool_docs = {
        case["case_id"]: {"case_id": case["case_id"], "route": "GRAPH", "k": 30,
                          "result_count": len(pool30[case["case_id"]]["paths"]),
                          "hits": pool30[case["case_id"]]["paths"]}
        for case in mh
    }
    _, pool_summary = evaluate_multihop_paths(
        mh, pool_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl")

    rows = []
    preserved_path_count = 0
    source_graph_path_count = 0
    for case in mh:
        case_id = case["case_id"]
        record = hybrid[case_id]
        g_hits = record["graph_hits"]
        m_hits = record["merged_hits"]
        graph_keys = {path_key(hit) for hit in g_hits if hit.get("path_edges")}
        merged_keys = {path_key(hit) for hit in m_hits if hit.get("source") == "graph"}
        retained = len(graph_keys & merged_keys)
        source_graph_path_count += len(graph_keys)
        preserved_path_count += retained
        graph_success = graph_by_id[case_id]["full_success"]
        merged_success = merged_by_id[case_id]["full_success"]
        rows.append({
            "case_id": case_id,
            "graph_only_full_success": graph_success,
            "graph_only_gold_path_rank": graph_by_id[case_id].get("actual_path_rank"),
            "hybrid_full_success": merged_success,
            "hybrid_gold_path_rank": merged_by_id[case_id].get("actual_path_rank"),
            "graph_path_count_before_merge": len(g_hits),
            "graph_path_count_after_merge_top10": sum(hit.get("source") == "graph" for hit in m_hits),
            "vector_result_count_after_merge_top10": sum(hit.get("source") == "vector" for hit in m_hits),
            "graph_paths_preserved": retained,
            "graph_paths_removed_by_top10": max(0, len(graph_keys) - retained),
            "merge_loss": graph_success and not merged_success,
            "candidate30_full_path_recall": pool30[case_id].get("candidate_evidence_recall"),
        })

    # Aggregate route-level hit metrics from the same live capture.
    route_hits = {}
    for route in ("GRAPH", "VECTOR", "HYBRID"):
        records = fixed["cases"][route]
        hit_key = {"GRAPH": "graph_hit_at_10", "VECTOR": "vector_hit_at_10",
                   "HYBRID": "hybrid_hit_at_10"}[route]
        route_hits[route] = {"hits": sum(bool(row[hit_key]) for row in records.values()),
                             "total": len(records)}

    output = {
        "same_call_route_hits": route_hits,
        "graph_only_multihop_full_success_at_10": graph_summary["full_success"],
        "hybrid_multihop_full_success_at_10": merged_summary["full_success"],
        "graph_candidate_pool30_multihop_full_path_recall": pool_summary["full_success"],
        "hybrid_graph_path_preservation_rate": round(
            preserved_path_count / source_graph_path_count, 4) if source_graph_path_count else None,
        "graph_top10_path_count": source_graph_path_count,
        "hybrid_top10_retained_graph_path_count": preserved_path_count,
        "lost_successful_paths": [row for row in rows if row["merge_loss"]],
        "cases": rows,
    }
    target = REPORTS / "hybrid_merge_diagnostics_after.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "cases"},
                     ensure_ascii=False, indent=2))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
