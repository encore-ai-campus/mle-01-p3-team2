"""Run the current router and retrieval stack on the 17 Graph-family multi-hop cases."""
from __future__ import annotations

import json
import time

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from scripts.evaluation.evaluate_multihop import expected_route_of, load_golden_cases
from scripts.evaluation.evaluate_retrieval import PROJECT_ROOT, _hit_row, evaluate_multihop_paths
from src.retrievers.graph_retriever import GraphRetriever
from src.retrievers.hybrid_retriever import hybrid_search
from src.retrievers.router import QuestionRouter
from src.retrievers.vector_retriever import VectorIndex


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_golden_cases(PROJECT_ROOT / "data/evaluation/golden_set_final.jsonl")
    cases = [case for case in cases if case.get("question_type") == "multi_hop"
             and expected_route_of(case) in {"GRAPH", "HYBRID"}
             and case.get("expected_relation_ids")]
    router = QuestionRouter()
    graph = GraphRetriever.from_env()
    vector = VectorIndex(
        graph.driver, SentenceTransformer("BAAI/bge-m3"), database=graph.database,
    )
    vector.verify_ready()
    docs: dict[str, dict] = {}
    route_details: list[dict] = []
    rows: list[dict] = []
    try:
        for case in cases:
            question = case["question"]
            decision = router.decide(question)
            started = time.perf_counter()
            if decision.route == "GRAPH":
                results = graph.search(question, crop=decision.crop,
                                       max_hops=decision.max_hops, limit=10)
            elif decision.route == "VECTOR":
                results = vector.search(question, k=10)
            elif decision.route == "HYBRID":
                results = hybrid_search(
                    question, graph_retriever=graph, vector_index=vector,
                    crop=decision.crop, max_hops=decision.max_hops,
                    graph_limit=10, vector_limit=10, limit=10,
                )
            else:
                results = []
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            hits = [_hit_row(rank, result) for rank, result in enumerate(results, 1)]
            docs[case["case_id"]] = {
                "case_id": case["case_id"], "route": decision.route,
                "k": 10, "result_count": len(hits), "hits": hits,
            }
            route_details.append({"case_id": case["case_id"], "actual_route": decision.route})
            rows.append({"case_id": case["case_id"], "expected_route": expected_route_of(case),
                         "actual_route": decision.route, "elapsed_ms": elapsed_ms,
                         "hits": hits})
            print(f"{case['case_id']} expected={expected_route_of(case)} "
                  f"actual={decision.route} elapsed_ms={elapsed_ms}", flush=True)
        details, summary = evaluate_multihop_paths(
            cases, docs, route_details, PROJECT_ROOT / "data/processed/relations.jsonl",
        )
        output = {
            "mode": "actual router plus actual route retrieval",
            "case_count": len(cases), "summary": summary,
            "case_details": details, "retrieval_rows": rows,
        }
        path = PROJECT_ROOT / "data/evaluation/reports/router_p0_multihop_actual.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Report: {path}")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
