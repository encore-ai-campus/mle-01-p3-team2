"""Collect fixed GRAPH/VECTOR inputs and current HYBRID output for route evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT, _hit_row, evaluate_multihop_paths
from src.retrievers.graph_retriever import GraphRetriever
from src.retrievers.hybrid_retriever import merge_results
from src.retrievers.router import QuestionRouter
from src.retrievers.vector_retriever import VectorIndex


def score_hits(case: dict, hits: list[dict]) -> bool:
    evidence = set(case.get("expected_evidence_ids") or [])
    chunks = set(case.get("expected_source_chunk_ids") or [])
    return any(evidence.intersection(hit.get("evidence_ids") or [])
               or chunks.intersection(hit.get("chunk_ids") or []) for hit in hits[:10])


def encode_results(results: list, raw: bool = False) -> list[dict]:
    encoded = []
    for rank, result in enumerate(results, 1):
        row = _hit_row(rank, result)
        row["hybrid_score"] = result.metadata.get("hybrid_score")
        row["graph_rrf_score"] = result.metadata.get("graph_rrf_score")
        row["vector_rrf_score"] = result.metadata.get("vector_rrf_score")
        row["matched_vector_hit_count"] = len(result.metadata.get("matched_vector_hits") or [])
        if raw:
            row["raw_result"] = asdict(result)
        encoded.append(row)
    return encoded


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    graph_cases = [case for case in cases if expected_route_of(case) == "GRAPH"]
    vector_cases = [case for case in cases if expected_route_of(case) == "VECTOR"]
    hybrid_cases = [case for case in cases if expected_route_of(case) == "HYBRID"]
    router = QuestionRouter()
    graph = GraphRetriever.from_env()
    vector = VectorIndex(graph.driver, SentenceTransformer("BAAI/bge-m3"), database=graph.database)
    vector.verify_ready()
    records = {"GRAPH": {}, "VECTOR": {}, "HYBRID": {}}
    timings: dict[str, list[float]] = {key: [] for key in records}
    try:
        for route, route_cases in (("GRAPH", graph_cases), ("VECTOR", vector_cases), ("HYBRID", hybrid_cases)):
            for case in route_cases:
                decision = router.decide(case["question"])
                started = time.perf_counter()
                graph_results = []
                vector_results = []
                if route in {"GRAPH", "HYBRID"}:
                    graph_results = graph.search(
                        case["question"], crop=decision.crop, max_hops=decision.max_hops,
                        limit=10,
                    )
                if route in {"VECTOR", "HYBRID"}:
                    vector_results = vector.search(case["question"], k=10)
                if route == "GRAPH":
                    merged_results = graph_results
                elif route == "VECTOR":
                    merged_results = vector_results
                else:
                    merged_results = merge_results(graph_results, vector_results, limit=10)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                timings[route].append(elapsed_ms)
                record = {
                    "case_id": case["case_id"], "expected_route": route,
                    "router_route": decision.route, "crop": decision.crop,
                    "max_hops": decision.max_hops, "elapsed_ms": elapsed_ms,
                    "graph_result_count": len(graph_results),
                    "vector_result_count": len(vector_results),
                    "graph_hits": encode_results(graph_results, raw=True),
                    "vector_hits": encode_results(vector_results, raw=True),
                    "merged_hits": encode_results(merged_results, raw=True),
                    "graph_hit_at_10": score_hits(case, encode_results(graph_results)),
                    "vector_hit_at_10": score_hits(case, encode_results(vector_results)),
                    "hybrid_hit_at_10": score_hits(case, encode_results(merged_results)),
                }
                records[route][case["case_id"]] = record
                print(f"{route} {case['case_id']} graph={len(graph_results)} "
                      f"vector={len(vector_results)} result={len(merged_results)} "
                      f"elapsed_ms={elapsed_ms}", flush=True)
        mh = [case for case in hybrid_cases if case.get("question_type") == "multi_hop"
              and case.get("expected_relation_ids")]
        docs = {
            case["case_id"]: {
                "case_id": case["case_id"], "route": "HYBRID", "k": 10,
                "result_count": len(records["HYBRID"][case["case_id"]]["merged_hits"]),
                "hits": records["HYBRID"][case["case_id"]]["merged_hits"],
            }
            for case in mh
        }
        routes = [{"case_id": case["case_id"], "actual_route": "HYBRID"} for case in mh]
        _, multi_summary = evaluate_multihop_paths(
            mh, docs, routes, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        output = {
            "mode": "same-call fixed GRAPH/VECTOR inputs with default merge_results",
            "route_counts": {route: len(records[route]) for route in records},
            "summary": {
                route: {
                    "hit_at_10": sum(row["hybrid_hit_at_10"] if route == "HYBRID"
                                      else row[f"{route.lower()}_hit_at_10"]
                                      for row in records[route].values()),
                    "questions": len(records[route]),
                    "mean_elapsed_ms": round(sum(timings[route]) / len(timings[route]), 3),
                }
                for route in records
            },
            "multihop_integrated": multi_summary,
            "cases": records,
        }
        target = PROJECT_ROOT / "data/evaluation/reports/hybrid_merge_fixed_candidates_after.json"
        target.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in output.items() if key != "cases"},
                         ensure_ascii=False, indent=2))
        print(f"Report: {target}")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
