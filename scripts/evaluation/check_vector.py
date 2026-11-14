"""Diagnose four VECTOR Hit@10 misses against the live Neo4j vector index.

Run from the project root with its virtual environment:
    .venv/Scripts/python.exe -m scripts.evaluation.check_vector

The script only reads Neo4j data and writes a diagnostic JSON report. It does
not modify the index, embeddings, retrieval code, or golden-set cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
DEFAULT_OUTPUT = REPORT_DIR / "vector_failure_diagnostic_v3.json"
DEFAULT_BASELINE_TOP10 = REPORT_DIR / "top10_results.jsonl"
TARGET_CASE_IDS = ("GS039", "GS045", "GS056", "GS065")
SEARCH_KS = (10, 20, 50, 100)

# These terms are taken from each question and its expected facts. Normalization
# ignores spacing and punctuation so dates such as 2023-10-27/20231027 compare.
KEY_TERMS: dict[str, tuple[str, ...]] = {
    "GS039": (
        "고추", "칼라탄", "노지", "터널", "건고추", "화건",
        "토마토반점위조바이러스", "TSWV", "칼라병",
    ),
    "GS045": (
        "양파", "잎마름병", "생육 후기", "4~5월", "강우일수", "다습",
        "건전한 종자", "소독", "병에 잘 걸리지 않는 품종", "균형시비",
    ),
    "GS056": ("마늘", "토양검정", "농업기술센터", "농협", "그 결과"),
    "GS065": (
        "경상남도", "밀양시", "2023년 10월 27일", "20231027", "상추", "안랩",
    ),
}

_NON_ALNUM = re.compile(r"[^0-9a-z가-힣]+", re.IGNORECASE)


def _normalized(text: str | None) -> str:
    return _NON_ALNUM.sub("", (text or "").lower())


def _key_term_audit(
    case_id: str, question: str, embedding_text: str | None, raw_text: str | None
) -> list[dict[str, Any]]:
    embedding_normalized = _normalized(embedding_text)
    raw_normalized = _normalized(raw_text)
    question_normalized = _normalized(question)
    return [
        {
            "term": term,
            "in_question": _normalized(term) in question_normalized,
            "in_expected_embedding_text": _normalized(term) in embedding_normalized,
            "in_expected_raw_text": _normalized(term) in raw_normalized,
        }
        for term in KEY_TERMS[case_id]
    ]


def _load_targets() -> dict[str, dict[str, Any]]:
    cases = {
        case["case_id"]: case
        for case in load_golden_cases(DEFAULT_GOLDEN_PATH)
        if case.get("case_id") in TARGET_CASE_IDS
    }
    missing = set(TARGET_CASE_IDS) - cases.keys()
    if missing:
        raise ValueError(f"Golden-set cases missing: {sorted(missing)}")
    for case_id, case in cases.items():
        if expected_route_of(case) != "VECTOR":
            raise ValueError(f"{case_id} is not a VECTOR evaluation case")
        if len(case.get("expected_source_chunk_ids") or []) != 1:
            raise ValueError(f"{case_id} must have exactly one expected source chunk")
    return cases


def _load_baseline_top10(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not path.is_file():
        return {}, {"path": str(path), "available": False}
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") not in TARGET_CASE_IDS:
                continue
            if row.get("route") != "VECTOR" or row.get("k") != 10:
                raise ValueError(f"Invalid baseline Top-10 row at line {line_number}")
            rows[row["case_id"]] = row.get("hits") or []
    stat = path.stat()
    return rows, {
        "path": str(path.resolve()),
        "available": True,
        "file_mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "persisted_question": False,
        "persisted_query_vector": False,
        "persisted_model_revision": False,
    }


def _compare_baseline_top10(
    old_hits: list[dict[str, Any]] | None,
    current_hits: list[dict[str, Any]],
    expected_chunk_id: str,
    expected_direct_score: float | None,
) -> dict[str, Any]:
    if old_hits is None:
        return {"available_for_case": False}
    old_ids = [
        chunk_id
        for hit in old_hits
        for chunk_id in hit.get("chunk_ids", [])
    ]
    new_ids = [hit.get("chunk_id") for hit in current_hits]
    old_scores = {
        chunk_id: float(hit["score"])
        for hit in old_hits
        for chunk_id in hit.get("chunk_ids", [])
        if hit.get("score") is not None
    }
    new_scores = {
        hit["chunk_id"]: float(hit["score"])
        for hit in current_hits
        if hit.get("chunk_id") is not None and hit.get("score") is not None
    }
    shared_ids = sorted(set(old_scores) & set(new_scores))
    shared_score_deltas = {
        chunk_id: new_scores[chunk_id] - old_scores[chunk_id]
        for chunk_id in shared_ids
    }
    old_expected = next(
        (hit for hit in old_hits if expected_chunk_id in hit.get("chunk_ids", [])),
        None,
    )
    cutoff = float(old_hits[-1]["score"]) if old_hits else None
    return {
        "available_for_case": True,
        "baseline_k": 10,
        "baseline_expected_rank": old_expected.get("rank") if old_expected else None,
        "baseline_expected_score": old_expected.get("score") if old_expected else None,
        "baseline_top10_cutoff": cutoff,
        "expected_direct_score_minus_baseline_cutoff": (
            expected_direct_score - cutoff
            if expected_direct_score is not None and cutoff is not None
            else None
        ),
        "current_k10_expected_rank": next(
            (hit["rank"] for hit in current_hits if hit.get("chunk_id") == expected_chunk_id),
            None,
        ),
        "top10_chunk_id_overlap_count": len(set(old_ids) & set(new_ids)),
        "same_top10_order": old_ids == new_ids,
        "shared_candidate_scores_equal_within_1e-6": all(
            abs(delta) <= 1e-6 for delta in shared_score_deltas.values()
        ),
        "shared_candidate_score_deltas_current_minus_baseline": shared_score_deltas,
        "baseline_chunk_ids": old_ids,
        "current_chunk_ids": new_ids,
    }


def _fetch_chunk_metadata(
    driver: Any, chunk_ids: set[str], database: str
) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    records, _, _ = driver.execute_query(
        "UNWIND $chunk_ids AS requested_id "
        "MATCH (c:Chunk {chunk_id: requested_id}) "
        "RETURN c.chunk_id AS chunk_id, c.evidence_id AS evidence_id, "
        "c.title AS title, c.document_id AS document_id, "
        "c.embedding_text AS embedding_text, c.raw_text AS raw_text, "
        "c.clean_text AS clean_text, c.embedding_model AS embedding_model, "
        "c.embedding_dimension AS embedding_dimension",
        chunk_ids=sorted(chunk_ids),
        database_=database,
    )
    return {str(row["chunk_id"]): dict(row) for row in records}


def _fetch_target_vector(
    driver: Any,
    chunk_id: str,
    query_vector: list[float],
    database: str,
) -> dict[str, Any] | None:
    records, _, _ = driver.execute_query(
        "MATCH (c:Chunk {chunk_id: $chunk_id}) "
        "RETURN c.chunk_id AS chunk_id, c.embedding AS embedding, "
        "c.evidence_id AS evidence_id, "
        "c.title AS title, c.document_id AS document_id, "
        "c.source_id AS source_id, c.source_url AS source_url, "
        "c.embedding_text AS embedding_text, c.raw_text AS raw_text, "
        "c.clean_text AS clean_text, c.embedding_model AS embedding_model, "
        "c.embedding_dimension AS embedding_dimension, "
        "vector.similarity.cosine(c.embedding, $query_embedding) AS neo4j_cosine_score",
        chunk_id=chunk_id,
        query_embedding=query_vector,
        database_=database,
    )
    return dict(records[0]) if records else None


def _fetch_direct_candidate_scores(
    driver: Any,
    chunk_ids: set[str],
    query_vector: list[float],
    database: str,
) -> dict[str, float]:
    if not chunk_ids:
        return {}
    records, _, _ = driver.execute_query(
        "UNWIND $chunk_ids AS requested_id "
        "OPTIONAL MATCH (c:Chunk {chunk_id: requested_id}) "
        "RETURN requested_id AS chunk_id, "
        "vector.similarity.cosine(c.embedding, $query_embedding) AS direct_score",
        chunk_ids=sorted(chunk_ids),
        query_embedding=query_vector,
        database_=database,
    )
    return {
        str(row["chunk_id"]): float(row["direct_score"])
        for row in records
        if row.get("chunk_id") is not None and row.get("direct_score") is not None
    }


def _raw_cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or not math.isfinite(denominator):
        raise ValueError("Cannot compute cosine similarity for a zero/non-finite vector")
    return float(np.dot(a, b) / denominator)


def _serialize_hit(hit: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "evidence_ids": hit.evidence_ids,
        "score": hit.score,
    }


def _search_with_query_vector(
    index: VectorIndex, question: str, query_vector: list[float], k: int
) -> list[Any]:
    """Run the production Cypher with a fixed query vector for comparable K values."""
    records, _, _ = index.driver.execute_query(
        index._search_query,
        k=k,
        embedding=query_vector,
        database_=index.database,
    )
    from src.retrievers.vector_retriever import vector_record_to_result

    return [vector_record_to_result(record) for record in records]


def _classify_direct_score(
    target_returned_at_100: int | None,
    direct_score: float | None,
    top100_cutoff: float | None,
) -> str:
    if target_returned_at_100 is not None:
        return "RETURNED_BY_VECTOR_INDEX"
    if direct_score is None or top100_cutoff is None:
        return "INSUFFICIENT_DATA"
    if direct_score > top100_cutoff + 1e-6:
        return "POSSIBLE_ANN_RECALL_MISS"
    return "TARGET_BELOW_TOP100_CUTOFF"


def diagnose_case(
    case_id: str,
    case: dict[str, Any],
    index: VectorIndex,
    baseline_hits: list[dict[str, Any]] | None = None,
    baseline_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    question = case["question"]
    expected_chunk_id = case["expected_source_chunk_ids"][0]
    query_vector = index.embed_question(question)
    searches: dict[str, list[dict[str, Any]]] = {}
    all_candidate_ids = {expected_chunk_id}

    # Every K uses the same freshly generated query vector and production Cypher.
    for k in SEARCH_KS:
        hits = _search_with_query_vector(index, question, query_vector, k)
        serialized = [_serialize_hit(hit, rank) for rank, hit in enumerate(hits, 1)]
        searches[str(k)] = serialized
        all_candidate_ids.update(
            hit["chunk_id"] for hit in serialized if hit.get("chunk_id")
        )

    target = _fetch_target_vector(
        index.driver, expected_chunk_id, query_vector, index.database
    )
    metadata = _fetch_chunk_metadata(index.driver, all_candidate_ids, index.database)
    direct_scores = _fetch_direct_candidate_scores(
        index.driver, all_candidate_ids, query_vector, index.database
    )

    if target is None:
        target_details: dict[str, Any] = {"chunk_id": expected_chunk_id, "exists": False}
        direct_score = None
        raw_cosine = None
    else:
        stored_vector = target.pop("embedding", None)
        target_details = {"chunk_id": expected_chunk_id, "exists": True, **target}
        direct_score = target.get("neo4j_cosine_score")
        raw_cosine = (
            _raw_cosine(query_vector, [float(value) for value in stored_vector])
            if stored_vector is not None
            else None
        )

    candidate_texts = {
        chunk_id: {
            **metadata.get(chunk_id, {"chunk_id": chunk_id, "exists": False}),
            "direct_neo4j_cosine_score": direct_scores.get(chunk_id),
        }
        for chunk_id in sorted(all_candidate_ids)
    }
    for k in SEARCH_KS:
        for hit in searches[str(k)]:
            hit["direct_neo4j_cosine_score"] = direct_scores.get(hit["chunk_id"])
    expected_embedding_text = (
        target_details.get("embedding_text") if target_details.get("exists") else None
    )
    expected_raw_text = (
        target_details.get("raw_text") if target_details.get("exists") else None
    )

    ranks: dict[str, dict[str, Any] | None] = {}
    for k in SEARCH_KS:
        match = next(
            (row for row in searches[str(k)] if row.get("chunk_id") == expected_chunk_id),
            None,
        )
        ranks[str(k)] = match

    top100 = searches[str(max(SEARCH_KS))]
    cutoff = float(top100[-1]["score"]) if len(top100) == max(SEARCH_KS) else None
    rank100 = ranks[str(max(SEARCH_KS))]
    exact_candidate_scores = [
        direct_scores[hit["chunk_id"]]
        for hit in top100
        if hit.get("chunk_id") in direct_scores
    ]
    exact_candidate_cutoff = min(exact_candidate_scores) if len(top100) == max(SEARCH_KS) and exact_candidate_scores else None
    exact_rank_in_top100_union = (
        1 + sum(score > float(direct_score) for score in exact_candidate_scores)
        if direct_score is not None and exact_candidate_scores
        else None
    )
    direct_minus_exact_cutoff = (
        float(direct_score) - exact_candidate_cutoff
        if direct_score is not None and exact_candidate_cutoff is not None
        else None
    )
    report = {
        "case_id": case_id,
        "question": question,
        "expected_facts": case.get("expected_facts") or [],
        "expected_evidence_ids": case.get("expected_evidence_ids") or [],
        "expected_chunk_id": expected_chunk_id,
        "expected_chunk_evidence_id_matches_golden": (
            target_details.get("evidence_id") in (case.get("expected_evidence_ids") or [])
        ),
        "key_term_audit_mode": "normalized literal substring; paraphrases are not inferred",
        "expected_chunk": target_details,
        "key_term_audit": _key_term_audit(
            case_id, question, expected_embedding_text, expected_raw_text
        ),
        "searches_by_k": searches,
        "expected_chunk_rank_and_score_by_k": ranks,
        "baseline_evaluation_comparison": {
            **(baseline_metadata or {}),
            **_compare_baseline_top10(
                baseline_hits, searches["10"], expected_chunk_id,
                float(direct_score) if direct_score is not None else None,
            ),
        },
        "top100_score_cutoff": cutoff,
        "query_vector": {
            "dimension": len(query_vector),
            "sha256_float32": hashlib.sha256(
                np.asarray(query_vector, dtype=np.float32).tobytes()
            ).hexdigest(),
        },
        "direct_similarity": {
            "neo4j_cosine_score": direct_score,
            "raw_mathematical_cosine": raw_cosine,
            "exact_rank_among_top100_candidates": exact_rank_in_top100_union,
            "exact_top100_candidate_score_cutoff": exact_candidate_cutoff,
            "direct_minus_exact_top100_cutoff": direct_minus_exact_cutoff,
            "score_comparison_note": (
                "neo4j_cosine_score uses Neo4j's vector similarity function and is "
                "comparable to the vector-index scores; raw_mathematical_cosine is "
                "the conventional cosine in [-1, 1]."
            ),
        },
        "diagnosis_signal": _classify_direct_score(
            rank100["rank"] if rank100 else None,
            float(direct_score) if direct_score is not None else None,
            exact_candidate_cutoff,
        ),
        "candidate_texts_by_chunk_id": candidate_texts,
    }
    return report, candidate_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-top10", type=Path, default=DEFAULT_BASELINE_TOP10)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    env_path = PROJECT_ROOT / ".env"
    if not load_dotenv(env_path):
        raise FileNotFoundError(f"Project .env was not found: {env_path}")
    for required in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        if not os.getenv(required):
            raise RuntimeError(f"Required environment variable is missing: {required}")

    cases = _load_targets()
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        print("Neo4j read-only diagnostic connection: OK", flush=True)
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
        print(f"Vector index ready: {index_info}", flush=True)

        baseline_by_case, baseline_metadata = _load_baseline_top10(args.baseline_top10)
        reports = []
        for case_id in TARGET_CASE_IDS:
            report, _ = diagnose_case(
                case_id,
                cases[case_id],
                index,
                baseline_hits=baseline_by_case.get(case_id),
                baseline_metadata=baseline_metadata,
            )
            reports.append(report)
            ranks = report["expected_chunk_rank_and_score_by_k"]
            rank_summary = ", ".join(
                f"Top{k}="
                + (
                    f"rank {ranks[str(k)]['rank']} score {ranks[str(k)]['score']:.6f}"
                    if ranks[str(k)]
                    else "not found"
                )
                for k in SEARCH_KS
            )
            print(
                f"{case_id} {rank_summary}; "
                f"direct Neo4j cosine={report['direct_similarity']['neo4j_cosine_score']}; "
                f"signal={report['diagnosis_signal']}",
                flush=True,
            )

        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "index": index_info,
            "model": DEFAULT_MODEL_NAME,
            "dimension": DEFAULT_DIMENSION,
            "search_k_values": list(SEARCH_KS),
            "cases": reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Diagnostic report written: {args.output.resolve()}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
