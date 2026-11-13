"""Neo4j vector index used by the production vector retriever."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .search_result import SearchResult

DEFAULT_INDEX_NAME = "chunk_embedding"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DIMENSION = 1024
_VALID_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
_VALID_STRATEGIES = frozenset({"all", "vector", "graph", "hybrid"})


class Neo4jDriver(Protocol):
    def execute_query(self, query: str, **kwargs: object) -> object: ...


class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


class VectorIndex:
    """Query a Neo4j Chunk cosine index and validate its configuration."""

    def __init__(
        self,
        driver: Neo4jDriver,
        embedder: Embedder,
        *,
        index_name: str = DEFAULT_INDEX_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_DIMENSION,
        database: str = "neo4j",
    ) -> None:
        if not isinstance(index_name, str) or not _VALID_IDENTIFIER.fullmatch(index_name):
            raise ValueError("index_name must be a valid Neo4j identifier")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be nonempty")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(database, str) or not database.strip():
            raise ValueError("database must be nonempty")

        self.driver = driver
        self.embedder = embedder
        self.index_name = index_name
        self.model_name = model_name
        self.dimension = dimension
        self.database = database
        # Cypher 25 SEARCH syntax requires a literal index name; the identifier is validated.
        self._search_query = (
            "CYPHER 25 MATCH (node:Chunk) "
            f"SEARCH node IN (VECTOR INDEX {index_name} "
            "FOR $embedding LIMIT $k) SCORE AS score "
            "RETURN node.chunk_id AS chunk_id, "
            "node.source_id AS source_id, "
            "node.evidence_id AS evidence_id, "
            "node.document_id AS document_id, "
            "node.title AS title, "
            "node.source_url AS url, "
            "node.raw_text AS text, "
            "node.embedding_model AS embedding_model, "
            "node.embedding_dimension AS embedding_dimension, "
            "node.source_official AS source_official, "
            "score ORDER BY score DESC"
        )

    def embed_question(self, question: str) -> list[float]:
        """Encode and L2-normalize one question; reject invalid embeddings."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        encoded = np.asarray(self.embedder.encode([question]), dtype=np.float32)
        if encoded.shape != (1, self.dimension):
            raise ValueError(
                f"Query embedding dimension must be {self.dimension}, got shape {encoded.shape}"
            )
        vector = encoded[0]
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding must be a finite, nonzero vector")
        # float64 accumulation avoids float32 overflow for unusually large embeddings.
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Query embedding must be a finite, nonzero vector")
        normalized = vector / norm
        return normalized.tolist()

    def verify_ready(self) -> dict[str, Any]:
        """Check that the expected Neo4j vector index is ONLINE and compatible."""
        records, _, _ = self.driver.execute_query(
            "SHOW VECTOR INDEXES YIELD name, state, labelsOrTypes, properties, options "
            "WHERE name = $name RETURN name, state, labelsOrTypes, properties, options",
            name=self.index_name,
            database_=self.database,
        )
        if not records:
            raise ValueError(f"Vector index not found: {self.index_name}")
        record = records[0]
        config = (record.get("options") or {}).get("indexConfig") or {}
        if record.get("state") != "ONLINE":
            raise ValueError(f"Vector index must be ONLINE, got {record.get('state')}")
        actual_dimension = config.get("vector.dimensions")
        if actual_dimension != self.dimension:
            raise ValueError(
                f"Vector index dimension must be {self.dimension}, got {actual_dimension}"
            )
        similarity = str(config.get("vector.similarity_function", "")).lower()
        if (record.get("labelsOrTypes") != ["Chunk"]
                or record.get("properties") != ["embedding"]
                or similarity != "cosine"):
            raise ValueError("Vector index configuration is incompatible")
        return {
            "name": record["name"],
            "state": record["state"],
            "model_name": self.model_name,
            "dimension": actual_dimension,
            "similarity": similarity,
        }

    def _search_verified(self, question: str, k: int) -> list[SearchResult]:
        """Search after caller has verified the index (internal evaluation fast path)."""
        query_vector = self.embed_question(question)
        records, _, _ = self.driver.execute_query(
            self._search_query,
            k=k,
            embedding=query_vector,
            database_=self.database,
        )
        results: list[SearchResult] = []
        for record in records:
            score = record.get("score")
            if score is None or not math.isfinite(float(score)):
                raise ValueError("Neo4j returned a non-finite vector score")
            chunk_id = record.get("chunk_id")
            source_id = record.get("source_id")
            evidence_id = record.get("evidence_id")
            raw_text = record.get("text")
            results.append(
                SearchResult(
                    source="vector",
                    text=raw_text if isinstance(raw_text, str) else "",
                    score=float(score),
                    evidence_ids=[str(evidence_id)] if evidence_id else [],
                    source_ids=[str(source_id)] if source_id else [],
                    chunk_ids=[str(chunk_id)] if chunk_id else [],
                    path=None,
                    relation_types=None,
                    metadata={
                        "embedding_model": record.get("embedding_model"),
                        "embedding_dimension": record.get("embedding_dimension"),
                        "source_official": record.get("source_official"),
                        'document_id': record.get('document_id'),
                        'title': record.get('title'),
                        'url': record.get('url'),
                    },
                )
            )
        return results

    def search(self, question: str, *, k: int = 5) -> list[SearchResult]:
        """Embed a question and retrieve its nearest Chunk nodes from Neo4j."""
        _validate_k(k)
        # Single production requests always check the current index state.
        self.verify_ready()
        return self._search_verified(question, k)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def evaluate_golden_set(
    index: VectorIndex,
    golden_path: str | Path,
    *,
    k: int = 5,
    strategy: str = "vector",
) -> dict[str, Any]:
    """Calculate Hit@K, Recall@K and MRR@K for READY golden cases."""
    _validate_k(k)
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    cases: list[dict[str, Any]] = []
    with Path(golden_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {golden_path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Golden case at line {line_number} must be an object")
            if row.get("status") == "READY" and (
                strategy == "all" or row.get("query_strategy") == strategy
            ):
                cases.append(row)
    if not cases:
        raise ValueError(f"No READY golden cases for strategy {strategy}")

    # One readiness check for this evaluation run instead of one per question.
    index.verify_ready()
    details: list[dict[str, Any]] = []
    totals = {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}
    for case in cases:
        relevant = set(case.get("expected_source_chunk_ids") or [])
        if not relevant:
            raise ValueError(f"Case {case.get('case_id')} has no expected source chunks")
        hits = index._search_verified(case.get("question"), k)
        retrieved = [chunk_id for hit in hits for chunk_id in hit.chunk_ids]
        first_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant),
            None,
        )
        metrics = {
            "hit_at_k": float(first_rank is not None),
            "recall_at_k": len(relevant.intersection(retrieved)) / len(relevant),
            "mrr_at_k": 1 / first_rank if first_rank is not None else 0.0,
        }
        for metric, value in metrics.items():
            totals[metric] += value
        details.append({
            "case_id": case.get("case_id"),
            "query_strategy": case.get("query_strategy"),
            "expected_source_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": retrieved,
            **metrics,
        })
    count = len(details)
    return {
        "model_name": index.model_name,
        "k": k,
        "strategy": strategy,
        "count": count,
        **{metric: value / count for metric, value in totals.items()},
        "cases": details,
    }
