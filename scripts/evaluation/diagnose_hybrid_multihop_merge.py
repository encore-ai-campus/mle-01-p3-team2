"""Compare forced GRAPH and actual HYBRID Top-10 results for multi-hop cases."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import evaluate_multihop_paths, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "data/evaluation/reports"


def main() -> None:
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    cases = [case for case in cases if case.get("question_type") == "multi_hop"
             and expected_route_of(case) in {"GRAPH", "HYBRID"}
             and case.get("expected_relation_ids")]
    graph_sweep = json.loads((REPORTS / "graph_candidate_sweep_ranking-final.json").read_text(encoding="utf-8"))
    hybrid_run = json.loads((REPORTS / "router_p0_multihop_actual.json").read_text(encoding="utf-8"))
    graph_rows = {row["case_id"]: row for row in graph_sweep["cases"]
                  if row["candidate_pool_size"] == 30}
    hybrid_rows = {row["case_id"]: row for row in hybrid_run["retrieval_rows"]}
    graph_top10 = {
        case["case_id"]: {"case_id": case["case_id"], "route": "GRAPH", "k": 10,
                          "result_count": min(10, len(graph_rows[case["case_id"]]["paths"])),
                          "hits": graph_rows[case["case_id"]]["paths"][:10]}
        for case in cases
    }
    hybrid_top10 = {
        case["case_id"]: {"case_id": case["case_id"], "route": "HYBRID", "k": 10,
                          "result_count": len(hybrid_rows[case["case_id"]]["hits"]),
                          "hits": hybrid_rows[case["case_id"]]["hits"]}
        for case in cases
    }
    expected = [{"case_id": case["case_id"], "actual_route": expected_route_of(case)}
                for case in cases]
    graph_details, graph_summary = evaluate_multihop_paths(
        cases, graph_top10, expected, PROJECT_ROOT / "data/processed/relations.jsonl")
    hybrid_details, hybrid_summary = evaluate_multihop_paths(
        cases, hybrid_top10, expected, PROJECT_ROOT / "data/processed/relations.jsonl")
    graph_by_id = {row["case_id"]: row for row in graph_details}
    hybrid_by_id = {row["case_id"]: row for row in hybrid_details}

    rows = []
    for case in cases:
        case_id = case["case_id"]
        graph_detail = graph_by_id[case_id]
        hybrid_detail = hybrid_by_id[case_id]
        results = hybrid_rows[case_id]["hits"]
        graph_hits = sum(hit["source"] == "graph" for hit in results)
        vector_hits = sum(hit["source"] == "vector" for hit in results)
        expected_relations = set(case.get("expected_relation_ids") or [])
        rank_any_expected_relation = next((
            hit["rank"] for hit in results
            if expected_relations.intersection(hit.get("relation_ids") or [])
        ), None)
        exact_path_present = any(
            expected_relations.issubset(set(hit.get("relation_ids") or []))
            and len(hit.get("relation_ids") or []) == len(expected_relations)
            for hit in results if hit["source"] == "graph"
        )
        candidate_recall = graph_rows[case_id].get("full_path_candidate_recall", False)
        rows.append({
            "case_id": case_id,
            "router_route": hybrid_rows[case_id]["actual_route"],
            "graph_only_success": graph_detail["full_success"],
            "graph_only_expected_path_rank": graph_detail.get("actual_path_rank"),
            "graph_candidate_pool_recall": candidate_recall,
            "hybrid_success": hybrid_detail["full_success"],
            "hybrid_expected_path_rank": hybrid_detail.get("actual_path_rank"),
            "hybrid_graph_result_count": graph_hits,
            "hybrid_vector_result_count": vector_hits,
            "hybrid_first_expected_relation_rank": rank_any_expected_relation,
            "hybrid_exact_expected_path_present": exact_path_present,
            "hybrid_top10_graph_fraction": round(graph_hits / max(len(results), 1), 3),
            "merge_loss": graph_detail["full_success"] and not hybrid_detail["full_success"],
            "loss_stage": (
                "final_limit_or_ranking" if graph_detail["full_success"] and not exact_path_present
                else "path_or_evidence_mismatch" if graph_detail["full_success"] and not hybrid_detail["full_success"]
                else "none"
            ),
        })

    output = {
        "graph_only_full_success_at_10": graph_summary["full_success"],
        "hybrid_full_success_at_10": hybrid_summary["full_success"],
        "graph_candidate_full_path_recall_at_30": sum(
            bool(row.get("full_path_candidate_recall")) for row in graph_rows.values()
            if row["case_id"] in {case["case_id"] for case in cases}
        ),
        "graph_top10_to_hybrid_top10_path_preservation": sum(
            row["graph_only_success"] and row["hybrid_success"] for row in rows
        ),
        "graph_top10_paths_lost_in_hybrid": [row for row in rows if row["merge_loss"]],
        "cases": rows,
    }
    path = REPORTS / "hybrid_multihop_merge_diagnostics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = REPORTS / "hybrid_multihop_case_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in output.items() if key != "cases"},
                     ensure_ascii=False, indent=2))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Reports: {path}, {csv_path}")


if __name__ == "__main__":
    main()
