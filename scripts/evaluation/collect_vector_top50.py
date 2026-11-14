"""Collect fresh VECTOR Top-50 candidates for the 19 READY golden cases.

Run from the project root with the local environment:
    .venv/Scripts/python.exe -m scripts.evaluation.collect_vector_top50

This read-only collector does not alter Neo4j nodes or indexes. Its output is
separate from the historical Top-10 evaluation files.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from scripts.evaluation.evaluate_multihop import (
    DEFAULT_GOLDEN_PATH,
    expected_route_of,
    load_golden_cases,
)
from src.retrievers.vector_retriever import (
    DEFAULT_DIMENSION,
    DEFAULT_INDEX_NAME,
    DEFAULT_MODEL_NAME,
    VectorIndex,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
DEFAULT_OUTPUT = REPORT_DIR / "vector_reranker_top50_candidates_v1.json"
TOP_K = 50
EXPECTED_VECTOR_CASE_COUNT = 19


def _vector_cases() -> list[dict[str, Any]]:
    cases = [
        case
        for case in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if expected_route_of(case) == "VECTOR"
    ]
    if len(cases) != EXPECTED_VECTOR_CASE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_CASE_COUNT} READY VECTOR cases, found {len(cases)}"
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case_id in VECTOR golden cases")
    return cases


def _fetch_candidate_metadata(
    driver: Any, chunk_ids: set[str], database: str
) -> dict[str, dict[str, Any]]:
    records, _, _ = driver.execute_query(
        "UNWIND $chunk_ids AS requested_id "
        "MATCH (c:Chunk {chunk_id: requested_id}) "
        "RETURN c.chunk_id AS chunk_id, c.evidence_id AS evidence_id, "
        "c.document_id AS document_id, c.title AS title, "
        "c.source_id AS source_id, c.source_url AS source_url, "
        "c.embedding_text AS embedding_text, c.raw_text AS raw_text, "
        "c.clean_text AS clean_text",
        chunk_ids=sorted(chunk_ids),
        database_=database,
    )
    return {str(record["chunk_id"]): dict(record) for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    env_path = PROJECT_ROOT / ".env"
    if not load_dotenv(env_path):
        raise FileNotFoundError(f"Project .env was not found: {env_path}")
    for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        if not os.getenv(name):
            raise RuntimeError(f"Required environment variable is missing: {name}")

    cases = _vector_cases()
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        embedder = SentenceTransformer(DEFAULT_MODEL_NAME, device=args.device)
        index = VectorIndex(
            driver,
            embedder,
            index_name=DEFAULT_INDEX_NAME,
            model_name=DEFAULT_MODEL_NAME,
            dimension=DEFAULT_DIMENSION,
            database=database,
        )
        index_info = index.verify_ready()
        print(
            f"Collecting {len(cases)} VECTOR cases from {index_info['name']} "
            f"with Top-{TOP_K}...",
            flush=True,
        )

        retrieved_cases: list[dict[str, Any]] = []
        all_chunk_ids: set[str] = set()
        for case in cases:
            hits = index._search_verified(case["question"], k=TOP_K)
            serialized_hits = []
            for rank, hit in enumerate(hits, 1):
                if not hit.chunk_id:
                    raise ValueError(f"Missing chunk_id in {case['case_id']} rank {rank}")
                all_chunk_ids.add(hit.chunk_id)
                serialized_hits.append({
                    "rank": rank,
                    "chunk_id": hit.chunk_id,
                    "evidence_ids": hit.evidence_ids,
                    "score": hit.score,
                })
            retrieved_cases.append({
                "case_id": case["case_id"],
                "question": case["question"],
                "expected_evidence_ids": case.get("expected_evidence_ids") or [],
                "expected_chunk_ids": case.get("expected_source_chunk_ids") or [],
                "expected_facts": case.get("expected_facts") or [],
                "vector_hits": serialized_hits,
            })
            print(f"{case['case_id']}: {len(serialized_hits)}/{TOP_K}", flush=True)

        metadata = _fetch_candidate_metadata(driver, all_chunk_ids, database)
        missing_ids = all_chunk_ids - metadata.keys()
        if missing_ids:
            raise ValueError(f"Missing candidate Chunk nodes: {sorted(missing_ids)[:10]}")

        for case in retrieved_cases:
            case["candidate_texts_by_chunk_id"] = {
                hit["chunk_id"]: metadata[hit["chunk_id"]]
                for hit in case["vector_hits"]
            }

        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "golden_set": str(DEFAULT_GOLDEN_PATH),
            "index": index_info,
            "model": DEFAULT_MODEL_NAME,
            "dimension": DEFAULT_DIMENSION,
            "filter": None,
            "k": TOP_K,
            "case_count": len(retrieved_cases),
            "cases": retrieved_cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Top-50 candidates written: {args.output.resolve()}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
