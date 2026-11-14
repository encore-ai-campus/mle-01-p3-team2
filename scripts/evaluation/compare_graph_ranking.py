"""Compare relevance scoring on identical, previously retrieved GRAPH pools."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT, evaluate_multihop_paths
from src.retrievers.graph_retriever import _path_relevance_score
from src.retrievers.router import QuestionRouter


REPORT_DIR = PROJECT_ROOT / "data/evaluation/reports"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    cases_by_id = {case["case_id"]: case for case in cases}
    report = json.loads((REPORT_DIR / "graph_p0_comparison.json").read_text(encoding="utf-8"))
    pool_rows = report["cases"]
    entities = {row["entity_id"]: row for row in _load_jsonl(PROJECT_ROOT / "data/processed/entities.jsonl")}
    chunks = {row["chunk_id"]: row for row in _load_jsonl(PROJECT_ROOT / "data/processed/chunks.jsonl")}
    router = QuestionRouter()
    multi_cases = [
        case for case in cases
        if case.get("question_type") == "multi_hop"
        and expected_route_of(case) in ("GRAPH", "HYBRID")
        and case.get("expected_relation_ids")
    ]
    summaries: list[dict[str, Any]] = []
    case_report: list[dict[str, Any]] = []

    for size in (10, 20, 30, 50):
        old_docs: dict[str, dict[str, Any]] = {}
        new_docs: dict[str, dict[str, Any]] = {}
        old_full_times: list[float] = []
        new_full_times: list[float] = []
        rows = [row for row in pool_rows if row["candidate_pool_size"] == size]

        for row in rows:
            case = cases_by_id[row["case_id"]]
            # A stable copy of the exact same candidate pool is used for both rankings.
            old_hits = [dict(hit) for hit in row["paths"]]
            for rank, hit in enumerate(old_hits, 1):
                hit["rank"] = rank
            started = time.perf_counter()
            scored = []
            for old_rank, hit in enumerate(old_hits, 1):
                nodes = hit.get("path_nodes") or []
                edges = [dict(edge) for edge in (hit.get("path_edges") or [])]
                for edge in edges:
                    for chunk in edge.get("original_chunks") or []:
                        source_chunk = chunks.get(chunk.get("chunk_id"), {})
                        chunk["text"] = source_chunk.get("raw_text") or source_chunk.get("clean_text") or ""
                        chunk["evidence_id"] = chunk.get("evidence_id") or source_chunk.get("evidence_id")
                        chunk["has_text"] = bool(chunk["text"].strip())
                attributes = [
                    (entities.get(node.get("entity_id")) or {}).get("attributes") or {}
                    for node in nodes
                ]
                value = _path_relevance_score(nodes, edges, case["question"], attributes)
                scored.append((value, old_rank, hit))
            new_full_times.append((time.perf_counter() - started) * 1000)
            new_hits = [dict(item[2]) for item in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)]
            for rank, hit in enumerate(new_hits, 1):
                hit["rank"] = rank
            old_docs[case["case_id"]] = {"hits": old_hits}
            new_docs[case["case_id"]] = {"hits": new_hits}

        expected_routes = [{"case_id": case["case_id"], "actual_route": expected_route_of(case)} for case in multi_cases]
        actual_routes = [{"case_id": case["case_id"], "actual_route": router.route_query(case["question"]).route} for case in multi_cases]
        old_top_details, old_top = evaluate_multihop_paths(
            multi_cases, {key: {"hits": value["hits"][:10]} for key, value in old_docs.items()},
            expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        new_top_details, new_top = evaluate_multihop_paths(
            multi_cases, {key: {"hits": value["hits"][:10]} for key, value in new_docs.items()},
            expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        _, old_pool = evaluate_multihop_paths(
            multi_cases, old_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        _, new_pool = evaluate_multihop_paths(
            multi_cases, new_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        _, old_integrated = evaluate_multihop_paths(
            multi_cases, {key: {"hits": value["hits"][:10]} for key, value in old_docs.items()},
            actual_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        _, new_integrated = evaluate_multihop_paths(
            multi_cases, {key: {"hits": value["hits"][:10]} for key, value in new_docs.items()},
            actual_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )

        graph_cases = [case for case in cases if expected_route_of(case) == "GRAPH"]
        old_graph_hit = new_graph_hit = 0
        graph_candidate_recall = 0
        for case in graph_cases:
            old_hits = old_docs[case["case_id"]]["hits"]
            new_hits = new_docs[case["case_id"]]["hits"]
            evidence = set(case.get("expected_evidence_ids") or [])
            chunk_ids = set(case.get("expected_source_chunk_ids") or [])
            old_graph_hit += int(any(evidence.intersection(hit.get("evidence_ids") or []) or
                                     chunk_ids.intersection(hit.get("chunk_ids") or []) for hit in old_hits[:10]))
            new_graph_hit += int(any(evidence.intersection(hit.get("evidence_ids") or []) or
                                     chunk_ids.intersection(hit.get("chunk_ids") or []) for hit in new_hits[:10]))
            graph_candidate_recall += int(any(evidence.intersection(hit.get("evidence_ids") or []) or
                                              chunk_ids.intersection(hit.get("chunk_ids") or []) for hit in new_hits))

        old_success = {detail["case_id"] for detail in old_top_details if detail["full_success"]}
        new_success = {detail["case_id"] for detail in new_top_details if detail["full_success"]}
        # Detailed ranks for all candidates, plus the old/new Top-10 verdict.
        old_all_details, _ = evaluate_multihop_paths(
            multi_cases, old_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        new_all_details, _ = evaluate_multihop_paths(
            multi_cases, new_docs, expected_routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        old_rank = {detail["case_id"]: detail.get("actual_path_rank") for detail in old_all_details}
        new_rank = {detail["case_id"]: detail.get("actual_path_rank") for detail in new_all_details}
        regressions = sorted(old_success - new_success)
        improvements = sorted(new_success - old_success)
        summary = {
            "candidate_pool_size": size,
            "candidate_set_identical": True,
            "graph_hit_at_10_before": old_graph_hit,
            "graph_hit_at_10_after": new_graph_hit,
            "graph_candidate_evidence_recall": graph_candidate_recall,
            "multihop_success_at_10_before": old_top["full_success"],
            "multihop_success_at_10_after": new_top["full_success"],
            "multihop_candidate_full_path_recall_before": old_pool["full_success"],
            "multihop_candidate_full_path_recall_after": new_pool["full_success"],
            "multihop_integrated_success_before": old_integrated["full_success"],
            "multihop_integrated_success_after": new_integrated["full_success"],
            "rerank_ms_per_case": round(sum(new_full_times) / max(len(new_full_times), 1), 3),
            "regressions": regressions,
            "improvements": improvements,
            "gs061_rank_before": old_rank.get("GS061"),
            "gs061_rank_after": new_rank.get("GS061"),
        }
        summaries.append(summary)
        for case in multi_cases:
            before = old_rank.get(case["case_id"])
            after = new_rank.get(case["case_id"])
            case_report.append({
                "candidate_pool_size": size,
                "case_id": case["case_id"],
                "question": case["question"],
                "best_path_rank_before": before,
                "best_path_rank_after": after,
                "top10_success_before": case["case_id"] in old_success,
                "top10_success_after": case["case_id"] in new_success,
                "regression": case["case_id"] in regressions,
                "improvement": case["case_id"] in improvements,
            })
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "candidate_set_source": "graph_p0_comparison.json; exact persisted candidate pool per case and size",
        "gold_ids_used_for_scoring": False,
        "scorer": "_path_relevance_score",
        "summaries": summaries,
        "case_rank_changes": case_report,
    }
    (REPORT_DIR / "graph_ranking_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (REPORT_DIR / "graph_ranking_case_changes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["candidate_pool_size", "case_id", "best_path_rank_before", "best_path_rank_after",
                   "top10_success_before", "top10_success_after", "regression", "improvement"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in case_report)
    print(f"Reports: {REPORT_DIR / 'graph_ranking_comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
