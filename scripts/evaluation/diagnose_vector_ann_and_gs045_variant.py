"""Read-only local diagnostics for GS065 ANN recall and the GS045 text variant.

Run from the project root using its .venv. The script embeds each question
once, repeats ANN retrieval with that exact query vector, and never writes to
Neo4j. GS045's alternate vector is evaluated only as a hypothetical candidate;
it is not inserted into the production index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from scripts.evaluation.check_vector import (
    _fetch_direct_candidate_scores,
    _fetch_target_vector,
    _search_with_query_vector,
)
from scripts.evaluation.evaluate_multihop import DEFAULT_GOLDEN_PATH, load_golden_cases
from src.retrievers.vector_retriever import (
    DEFAULT_DIMENSION,
    DEFAULT_INDEX_NAME,
    DEFAULT_MODEL_NAME,
    VectorIndex,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = DEFAULT_GOLDEN_PATH.parent / "reports"
DEFAULT_OUTPUT = REPORT_DIR / "vector_ann_metadata_experiment_v1.json"
VARIANT_PATH = REPORT_DIR / "gs045_metadata_embedding_variant_v1.json"
TARGETS = {"GS045", "GS065"}
REPEATS = 5
K = 100


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Vector shape mismatch: {a.shape} != {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denom or not math.isfinite(denom):
        raise ValueError("Invalid zero or non-finite vector")
    # Neo4j vector.similarity.cosine maps mathematical cosine [-1,1] to [0,1].
    return (float(np.dot(a, b) / denom) + 1.0) / 2.0


def _sha256_vector(vector: list[float]) -> str:
    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("Use at least two repeated searches to compare ANN consistency")

    env_path = ROOT / ".env"
    if not load_dotenv(env_path):
        raise FileNotFoundError(f"Project .env was not found: {env_path}")
    for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        if not os.getenv(name):
            raise RuntimeError(f"Required environment variable is missing: {name}")

    cases = {
        row["case_id"]: row
        for row in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if row.get("case_id") in TARGETS
    }
    if cases.keys() != TARGETS:
        raise ValueError(f"Expected cases {sorted(TARGETS)}, found {sorted(cases)}")
    variant = json.loads(VARIANT_PATH.read_text(encoding="utf-8"))
    gs045_alt_text_by_chunk = {
        row["chunk_id"]: row["candidate_embedding_text"]
        for row in variant["variants"]
    }

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        model = SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu", local_files_only=True)
        index = VectorIndex(
            driver,
            model,
            index_name=DEFAULT_INDEX_NAME,
            model_name=DEFAULT_MODEL_NAME,
            dimension=DEFAULT_DIMENSION,
            database=database,
        )
        index_info = index.verify_ready()
        reports: dict[str, Any] = {}

        for case_id in ("GS065", "GS045"):
            case = cases[case_id]
            chunk_id = case["expected_source_chunk_ids"][0]
            question = case["question"]
            query_vector = index.embed_question(question)
            ann_runs = []
            union_ids: set[str] = set()
            for run_no in range(1, args.repeats + 1):
                hits = _search_with_query_vector(index, question, query_vector, K)
                rows = [
                    {"rank": rank, "chunk_id": hit.chunk_id, "score": hit.score}
                    for rank, hit in enumerate(hits, 1)
                ]
                union_ids.update(row["chunk_id"] for row in rows if row.get("chunk_id"))
                target_hit = next((row for row in rows if row["chunk_id"] == chunk_id), None)
                ann_runs.append({
                    "run": run_no,
                    "target_rank": target_hit["rank"] if target_hit else None,
                    "target_score": target_hit["score"] if target_hit else None,
                    "top100_ids": [row["chunk_id"] for row in rows],
                    "top100_cutoff": rows[-1]["score"] if len(rows) == K else None,
                })

            target = _fetch_target_vector(driver, chunk_id, query_vector, database)
            if not target or target.get("embedding") is None:
                raise ValueError(f"Target vector missing for {case_id}: {chunk_id}")
            stored_vector = [float(value) for value in target.pop("embedding")]
            direct_scores = _fetch_direct_candidate_scores(driver, union_ids, query_vector, database)
            old_score = _cosine(query_vector, stored_vector)
            score_map = dict(direct_scores)
            score_map[chunk_id] = old_score
            old_rank_in_union = 1 + sum(score > old_score for score in score_map.values() if score is not None)

            row_report: dict[str, Any] = {
                "question": question,
                "chunk_id": chunk_id,
                "query_vector_sha256_float32": _sha256_vector(query_vector),
                "query_dimension": len(query_vector),
                "repeats": args.repeats,
                "target_in_ann_run_count": sum(run["target_rank"] is not None for run in ann_runs),
                "ann_runs": ann_runs,
                "current_vector": {
                    "neo4j_cosine_score": old_score,
                    "exact_rank_in_union_of_ann_candidates_and_target": old_rank_in_union,
                    "union_candidate_count": len(union_ids),
                    "target_present_in_production_vector_index": True,
                },
            }
            if case_id == "GS065":
                row_report["interpretation"] = (
                    "Same query vector and index were used for every ANN run. Stable absence with a better direct score "
                    "than returned candidates supports ANN recall/visibility inconsistency; unstable runs indicate ANN variability."
                )
            else:
                alt_text = gs045_alt_text_by_chunk[chunk_id]
                alt_vector_raw = np.asarray(
                    model.encode([alt_text], convert_to_numpy=True, normalize_embeddings=False)[0],
                    dtype=np.float32,
                )
                alt_norm = float(np.linalg.norm(alt_vector_raw.astype(np.float64)))
                if not alt_norm or alt_vector_raw.shape != (DEFAULT_DIMENSION,):
                    raise ValueError("Alternate GS045 vector has invalid dimension or norm")
                alt_vector = (alt_vector_raw / alt_norm).tolist()
                alt_score = _cosine(query_vector, alt_vector)
                hypothetical_scores = dict(direct_scores)
                hypothetical_scores[chunk_id] = alt_score
                alt_rank = 1 + sum(
                    score > alt_score for score in hypothetical_scores.values() if score is not None
                )
                row_report["metadata_variant"] = {
                    "embedding_text": alt_text,
                    "vector_dimension": len(alt_vector),
                    "vector_sha256_float32": _sha256_vector(alt_vector),
                    "old_direct_score": old_score,
                    "new_direct_score": alt_score,
                    "score_delta_new_minus_old": alt_score - old_score,
                    "old_rank_among_ann_union_plus_target": old_rank_in_union,
                    "new_hypothetical_rank_among_ann_union_plus_target": alt_rank,
                    "rank_scope": "proxy among the union of existing ANN candidates plus target; not a corpus-wide ANN rank",
                    "production_index_updated": False,
                }
                row_report["interpretation"] = (
                    "The alternate vector is not indexed. This compares direct similarity and a candidate-union proxy only; "
                    "the actual production ANN rank cannot change until a separate isolated index contains that vector."
                )
            reports[case_id] = row_report

        output = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "index": index_info,
            "model": DEFAULT_MODEL_NAME,
            "dimension": DEFAULT_DIMENSION,
            "k": K,
            "cases": reports,
            "read_only": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for case_id, row in reports.items():
            print(
                f"{case_id}: target in {row['target_in_ann_run_count']}/{args.repeats} ANN runs; "
                f"direct score={row['current_vector']['neo4j_cosine_score']:.6f}"
            )
            if "metadata_variant" in row:
                variant_row = row["metadata_variant"]
                print(
                    f"  GS045 metadata score {variant_row['old_direct_score']:.6f} -> "
                    f"{variant_row['new_direct_score']:.6f}; proxy rank "
                    f"{variant_row['old_rank_among_ann_union_plus_target']} -> "
                    f"{variant_row['new_hypothetical_rank_among_ann_union_plus_target']}"
                )
        print(f"Read-only diagnostic written: {args.output.resolve()}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
