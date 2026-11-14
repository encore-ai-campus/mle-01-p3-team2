"""Compare router P0 metrics and route-gated multi-hop results."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.evaluation.evaluate_multihop import (
    expected_route_of,
    load_golden_cases,
)
from scripts.evaluation.evaluate_retrieval import evaluate_multihop_paths
from src.retrievers.router import QuestionRouter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "data/evaluation/reports"
ROUTES = ("GRAPH", "HYBRID", "VECTOR", "REJECT")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def route_metrics(details: list[dict]) -> dict:
    matrix = {expected: {actual: 0 for actual in ROUTES} for expected in ROUTES}
    for row in details:
        matrix[row["expected_route"]][row["actual_route"]] += 1
    by_route = {}
    for route in ROUTES:
        tp = matrix[route][route]
        fp = sum(matrix[other][route] for other in ROUTES if other != route)
        fn = sum(matrix[route][other] for other in ROUTES if other != route)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_route[route] = {
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    correct = sum(matrix[route][route] for route in ROUTES)
    total = sum(sum(row.values()) for row in matrix.values())
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4),
            "confusion_matrix": matrix, "by_route": by_route}


def main() -> None:
    before = read_jsonl(REPORTS / "routing_details_router-p0-before.jsonl")
    after = read_jsonl(REPORTS / "routing_details_router-p0-final.jsonl")
    before_by_id = {row["case_id"]: row for row in before}
    after_by_id = {row["case_id"]: row for row in after}
    cases = load_golden_cases(ROOT / "data/evaluation/golden_set_final.jsonl")
    multihop = [
        case for case in cases
        if case.get("question_type") == "multi_hop"
        and expected_route_of(case) in {"GRAPH", "HYBRID"}
        and case.get("expected_relation_ids")
    ]
    before_mh = [before_by_id[case["case_id"]] for case in multihop]
    after_mh = [after_by_id[case["case_id"]] for case in multihop]

    # Reuse the persisted candidate-30 GRAPH pool. Only the route gate changes;
    # no answer IDs or expected paths enter the router.
    sweep = json.loads((REPORTS / "graph_candidate_sweep_ranking-final.json").read_text(encoding="utf-8"))
    pool_rows = [row for row in sweep["cases"] if row["candidate_pool_size"] == 30]
    pool_by_id = {row["case_id"]: row for row in pool_rows}
    docs = {
        case["case_id"]: {
            "case_id": case["case_id"], "route": "GRAPH", "k": 10,
            "result_count": min(10, len(pool_by_id[case["case_id"]]["paths"])),
            "hits": pool_by_id[case["case_id"]]["paths"][:10],
        }
        for case in multihop
    }
    old_routes = [{"case_id": case["case_id"],
                   "actual_route": pool_by_id[case["case_id"]]["router_route"]}
                  for case in multihop]
    current_router = QuestionRouter()
    new_routes = [{"case_id": case["case_id"],
                   "actual_route": current_router.decide(case["question"]).route}
                  for case in multihop]
    old_integrated = evaluate_multihop_paths(
        multihop, docs, old_routes, ROOT / "data/processed/relations.jsonl")[1]
    new_integrated = evaluate_multihop_paths(
        multihop, docs, new_routes, ROOT / "data/processed/relations.jsonl")[1]
    actual_multihop = json.loads(
        (REPORTS / "router_p0_multihop_actual.json").read_text(encoding="utf-8")
    )
    path_only = evaluate_multihop_paths(
        multihop, docs,
        [{"case_id": case["case_id"], "actual_route": expected_route_of(case)} for case in multihop],
        ROOT / "data/processed/relations.jsonl")[1]

    # Measure retrieval hit retention independently from the routing decision.
    golden_by_id = {case["case_id"]: case for case in cases}
    old_top = {row["case_id"]: row for row in read_jsonl(REPORTS / "top10_results.jsonl")}
    graph_rows = [row for row in pool_rows if row["expected_route"] == "GRAPH"]
    graph_retrieval = {
        "forced_graph_hit_at_10": sum(bool(row["hit_at_10"]) for row in graph_rows),
        "questions": len(graph_rows),
        "successful_graph_cases_routed_away": [
            row["case_id"] for row in graph_rows
            if row["hit_at_10"] and after_by_id[row["case_id"]]["actual_route"] == "VECTOR"
        ],
    }
    vector_cases = [case for case in cases if expected_route_of(case) == "VECTOR"]
    vector_hits = {}
    for case in vector_cases:
        evidence = set(case.get("expected_evidence_ids") or [])
        chunks = set(case.get("expected_source_chunk_ids") or [])
        hits = old_top[case["case_id"]]["hits"][:10]
        vector_hits[case["case_id"]] = any(
            evidence.intersection(hit.get("evidence_ids") or [])
            or chunks.intersection(hit.get("chunk_ids") or []) for hit in hits
        )
    vector_retrieval = {
        "forced_vector_hit_at_10": sum(vector_hits.values()),
        "questions": len(vector_cases),
        "successful_vector_cases_routed_away": [
            case_id for case_id, hit in vector_hits.items()
            if hit and after_by_id[case_id]["actual_route"] != "VECTOR"
        ],
    }

    output = {
        "before": route_metrics(before),
        "after": route_metrics(after),
        "multi_hop_count": len(multihop),
        "multi_hop_route_accuracy_before": sum(r["route_correct"] for r in before_mh),
        "multi_hop_route_accuracy_after": sum(r["route_correct"] for r in after_mh),
        "multi_hop_integrated_full_success_before": old_integrated["full_success"],
        "multi_hop_route_gated_graph_pool_success_after": new_integrated["full_success"],
        "multi_hop_integrated_retrieval_full_success_after": actual_multihop["summary"]["full_success"],
        "multi_hop_path_only_full_success_at_10": path_only["full_success"],
        "graph_retrieval": graph_retrieval,
        "vector_retrieval": vector_retrieval,
        "route_errors": [
            {"case_id": row["case_id"], "expected": row["expected_route"],
             "actual": row["actual_route"], "question_type": row["question_type"],
             "question": golden_by_id[row["case_id"]]["question"]}
            for row in after if not row["route_correct"]
        ],
    }
    target = REPORTS / "router_p0_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
