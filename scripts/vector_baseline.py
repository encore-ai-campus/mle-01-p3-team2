"""Build and evaluate an exact cosine-search baseline over processed chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 4
DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "chunks.jsonl"
DEFAULT_SOURCE_REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "processed" / "source_registry.jsonl"


class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


class Neo4jDriver(Protocol):
    def execute_query(self, query: str, **kwargs: object) -> object: ...


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def load_chunks(path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict]:
    """Read processed chunks and validate the fields needed for embedding."""
    rows = list(_jsonl(Path(path)))
    if not rows:
        raise ValueError("No chunks found")
    seen_ids = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Chunk {number} must be a JSON object")
        for field in ("chunk_id", "embedding_text"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Chunk {number} needs a nonempty string {field}")
        if row["chunk_id"] in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {row['chunk_id']}")
        seen_ids.add(row["chunk_id"])
    return rows


def load_source_official(path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH) -> dict[str, bool]:
    """Read each source's official status for retrieval safety checks."""
    official = {}
    for number, row in enumerate(_jsonl(Path(path)), 1):
        if not isinstance(row, dict):
            raise ValueError(f"Source {number} must be a JSON object")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"Source {number} needs a nonempty source_id")
        if not isinstance(row.get("official"), bool):
            raise ValueError(f"Source {number} needs a boolean official")
        if source_id in official:
            raise ValueError(f"Duplicate source_id: {source_id}")
        official[source_id] = row["official"]
    if not official:
        raise ValueError("No sources found")
    return official


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Embeddings must be a nonempty 2D array")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Embeddings must contain finite, nonzero vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("Embeddings must contain finite, nonzero vectors")
    return vectors / norms


def iter_embedding_batches(
    rows: Sequence[dict], embedder: Embedder, batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[list[dict], np.ndarray]]:
    """Encode document text in bounded batches, preserving chunk order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dimension = None
    for start in range(0, len(rows), batch_size):
        batch_rows = list(rows[start:start + batch_size])
        texts = [row["embedding_text"] for row in batch_rows]
        vectors = np.asarray(embedder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(batch_rows):
            raise ValueError("Embedder returned an unexpected shape")
        vectors = _normalize(vectors)
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError("Embedding dimension changed between batches")
        dimension = vectors.shape[1]
        yield batch_rows, vectors


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_index(
    chunks_path: str | Path,
    index_dir: str | Path,
    embedder: Embedder,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Embed every chunk and persist a matrix plus aligned chunk metadata."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = load_chunks(chunks_path)

    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Write batches directly to a disk-backed array instead of retaining every
    # embedding batch in memory. Publish the completed array atomically.
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".vectors-", suffix=".npy", dir=target, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)

        matrix = None
        position = 0
        for _, vectors in iter_embedding_batches(rows, embedder, batch_size):
            if matrix is None:
                matrix = np.lib.format.open_memmap(
                    temp_path, mode="w+", dtype=np.float32,
                    shape=(len(rows), vectors.shape[1]),
                )
            stop = position + len(vectors)
            matrix[position:stop] = vectors
            position = stop
        assert matrix is not None
        dimension = int(matrix.shape[1])
        matrix.flush()
        del matrix
        os.replace(temp_path, target / "vectors.npy")
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    metadata_keys = (
        "chunk_id", "document_id", "source_id", "title", "source_url",
        "page", "chunk_type", "clean_text", "quality_flags",
    )
    with (target / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            metadata = {key: row[key] for key in metadata_keys if key in row}
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    manifest = {
        "model_name": model_name, "count": len(rows), "dimension": dimension,
        "metric": "cosine", "vectors_sha256": _sha256(target / "vectors.npy"),
        "metadata_sha256": _sha256(target / "metadata.jsonl"),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def store_chunk_embeddings(
    rows: Sequence[dict],
    driver: Neo4jDriver,
    embedder: Embedder,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    database: str = "neo4j",
) -> dict:
    """Upsert Chunk nodes and their document embeddings in Neo4j batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not model_name.strip():
        raise ValueError("model_name must be nonempty")
    if not rows:
        raise ValueError("No chunks found")

    driver.execute_query(
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        database_=database,
    )
    write_query = (
        "UNWIND $rows AS row "
        "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
        "SET c += row.properties "
        "SET c.embedding = row.embedding, c.embedding_model = $model_name, "
        "c.embedding_dimension = $dimension"
    )
    property_names = (
        "embedding_text", "clean_text", "document_id", "source_id", "evidence_id",
        "title", "source_url", "page", "chunk_type",
    )
    dimension = None
    for batch_rows, batch_vectors in iter_embedding_batches(rows, embedder, batch_size):
        dimension = int(batch_vectors.shape[1])
        batch = []
        for row, vector in zip(batch_rows, batch_vectors):
            properties = {key: row[key] for key in property_names
                          if isinstance(row.get(key), (str, int, float, bool))}
            batch.append({
                "chunk_id": row["chunk_id"], "properties": properties,
                "embedding": vector.tolist(),
            })
        driver.execute_query(
            write_query, rows=batch, model_name=model_name,
            dimension=dimension, database_=database,
        )

    return {"count": len(rows), "dimension": dimension}


def store_chunk_metadata(
    rows: Sequence[dict],
    driver: Neo4jDriver,
    batch_size: int = 500,
    database: str = "neo4j",
    source_official: Mapping[str, bool] | None = None,
) -> dict:
    """Upsert source IDs and original text without recomputing embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not rows:
        raise ValueError("No chunks found")
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Chunk {number} must be a JSON object")
        for field in ("chunk_id", "source_id", "evidence_id", "raw_text"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Chunk {number} needs a nonempty string {field}")
        if source_official is not None and not isinstance(source_official.get(row["source_id"]), bool):
            raise ValueError(f"Chunk {number} has no official status for source_id {row['source_id']}")

    driver.execute_query(
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        database_=database,
    )
    query = (
        "UNWIND $rows AS row "
        "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
        "SET c.source_id = row.source_id, c.evidence_id = row.evidence_id, "
        "c.raw_text = row.raw_text"
    )
    if source_official is not None:
        query += ", c.source_official = row.source_official"
    for start in range(0, len(rows), batch_size):
        batch = [
            {"chunk_id": row["chunk_id"], "source_id": row["source_id"],
             "evidence_id": row["evidence_id"],
             "raw_text": row["raw_text"],
             **({"source_official": source_official[row["source_id"]]}
                if source_official is not None else {})}
            for row in rows[start:start + batch_size]
        ]
        driver.execute_query(query, rows=batch, database_=database)
    return {"count": len(rows)}


def build_neo4j_index(
    chunks_path: str | Path,
    driver: Neo4jDriver,
    embedder: Embedder,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    database: str = "neo4j",
) -> dict:
    """Store chunks and create a cosine vector index in Neo4j."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = load_chunks(chunks_path)
    stored = store_chunk_embeddings(rows, driver, embedder, model_name, batch_size, database)
    dimension = stored["dimension"]
    driver.execute_query(
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {dimension}, "
        "`vector.similarity_function`: 'cosine' }}",
        database_=database,
    )
    return {"backend": "neo4j", "model_name": model_name, "count": len(rows),
            "dimension": dimension, "index_name": "chunk_embedding"}


class VectorIndex:
    def __init__(self, vectors: np.ndarray, metadata: list[dict], manifest: dict):
        if vectors.ndim != 2 or len(vectors) != len(metadata) or len(metadata) != manifest["count"]:
            raise ValueError("Index files have inconsistent row counts")
        if vectors.shape[1] != manifest["dimension"]:
            raise ValueError("Index dimension differs from manifest")
        self.vectors = vectors
        self.metadata = metadata
        self.manifest = manifest

    @classmethod
    def load(cls, index_dir: str | Path) -> VectorIndex:
        root = Path(index_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        metadata = list(_jsonl(root / "metadata.jsonl"))
        index = cls(vectors, metadata, manifest)
        for filename, key in (("vectors.npy", "vectors_sha256"), ("metadata.jsonl", "metadata_sha256")):
            if manifest.get(key) != _sha256(root / filename):
                raise ValueError(f"Index integrity check failed for {filename}")
        return index

    def search(self, question: str, embedder: Embedder, k: int = 5) -> list[dict]:
        if not question.strip() or k < 1:
            raise ValueError("question must be nonempty and k must be positive")
        query = _normalize(np.asarray(embedder.encode([question]), dtype=np.float32))
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension differs from index")
        scores = self.vectors @ query[0]
        count = min(k, len(scores))
        if count == len(scores):
            order = np.argsort(-scores, kind="stable")
        else:
            # Include ties at the cutoff to preserve stable (original-order)
            # behavior of the original full argsort implementation.
            cutoff = np.partition(scores, len(scores) - count)[len(scores) - count]
            candidates = np.flatnonzero(scores >= cutoff)
            order = candidates[np.argsort(-scores[candidates], kind="stable")][:count]
        return [{**self.metadata[int(i)], "score": float(scores[i])} for i in order]


def evaluate(
    index: VectorIndex,
    golden_path: str | Path,
    embedder: Embedder,
    k: int = 5,
    strategy: str = "vector",
) -> dict:
    """Score READY cases matching the selected strategy (vector by default)."""
    if k < 1:
        raise ValueError("k must be positive")
    if strategy not in {"all", "vector", "graph", "hybrid"}:
        raise ValueError("Unknown strategy")
    cases = [row for row in _jsonl(Path(golden_path)) if row.get("status") == "READY"
             and (strategy == "all" or row.get("query_strategy") == strategy)]
    if not cases:
        raise ValueError("No READY golden cases for this strategy")
    details = []
    for case in cases:
        relevant = set(case.get("expected_source_chunk_ids") or [])
        if not relevant:
            raise ValueError(f"Case {case.get('case_id')} has no expected source chunks")
        hits = index.search(case["question"], embedder, k)
        retrieved = [hit["chunk_id"] for hit in hits]
        first_rank = next((rank for rank, identifier in enumerate(retrieved, 1) if identifier in relevant), None)
        details.append({
            "case_id": case.get("case_id"),
            "query_strategy": case.get("query_strategy"),
            "expected_source_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": retrieved,
            "hit_at_k": float(first_rank is not None),
            "recall_at_k": len(relevant.intersection(retrieved)) / len(relevant),
            "mrr_at_k": 1 / first_rank if first_rank else 0.0,
        })
    return {
        "model_name": index.manifest["model_name"], "k": k, "strategy": strategy,
        "count": len(details),
        **{metric: sum(row[metric] for row in details) / len(details)
           for metric in ("hit_at_k", "recall_at_k", "mrr_at_k")},
        "cases": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Embed chunks and write an exact vector index")
    build.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    build.add_argument("--index-dir", type=Path, default=Path("data/vector_index"))
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    for name, help_text in (
        ("neo4j-store", "Upsert Chunk nodes and embeddings in Neo4j"),
        ("neo4j-build", "Upsert chunks and create a Neo4j vector index"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        command.add_argument("--database", help="Neo4j database name (default: neo4j)")
        if name == "neo4j-store":
            command.add_argument("--resume", action="store_true",
                                 help="Skip chunks already stored with the same embedding model")
    metadata = sub.add_parser("neo4j-metadata", help="Store Chunk IDs, source IDs, and original text")
    metadata.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    metadata.add_argument("--batch-size", type=int, default=500)
    metadata.add_argument("--database", help="Neo4j database name (default: neo4j)")
    metadata.add_argument("--source-registry", type=Path,
                          default=DEFAULT_SOURCE_REGISTRY_PATH)
    for name in ("search", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--index-dir", type=Path, default=Path("data/vector_index"))
        command.add_argument("--model", help="Must match the model used to build the index")
        command.add_argument("--k", type=int, default=5)
        if name == "search":
            command.add_argument("question")
        else:
            command.add_argument("--golden", type=Path, default=Path("data/processed/golden_set.jsonl"))
            command.add_argument("--strategy", choices=("all", "vector", "graph", "hybrid"), default="vector")
            command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in ("neo4j-store", "neo4j-build", "neo4j-metadata"):
        from dotenv import load_dotenv
        from neo4j import GraphDatabase

        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            parser.error("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            database = args.database or os.getenv("NEO4J_DATABASE", "neo4j")
            if args.command == "neo4j-metadata":
                result = store_chunk_metadata(
                    load_chunks(args.chunks), driver, args.batch_size, database,
                    load_source_official(args.source_registry),
                )
            elif args.command == "neo4j-store":
                from sentence_transformers import SentenceTransformer
                rows = load_chunks(args.chunks)
                if args.resume:
                    existing, _, _ = driver.execute_query(
                        "MATCH (c:Chunk) WHERE c.embedding_model = $model_name "
                        "AND c.embedding IS NOT NULL RETURN c.chunk_id AS chunk_id",
                        model_name=args.model, database_=database,
                    )
                    stored_ids = {record["chunk_id"] for record in existing}
                    rows = [row for row in rows if row["chunk_id"] not in stored_ids]
                if rows:
                    embedder = SentenceTransformer(args.model)
                    result = store_chunk_embeddings(
                        rows, driver, embedder, args.model, args.batch_size, database,
                    )
                else:
                    result = {"count": 0, "dimension": None}
            else:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(args.model)
                result = build_neo4j_index(
                    args.chunks, driver, embedder, args.model,
                    args.batch_size, database,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "build":
        model_name = args.model
        index = None
    else:
        index = VectorIndex.load(args.index_dir)
        model_name = args.model or index.manifest["model_name"]
        if model_name != index.manifest["model_name"]:
            parser.error("--model must match the model in manifest.json")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    if args.command == "build":
        result = build_index(args.chunks, args.index_dir, embedder, model_name, args.batch_size)
    elif args.command == "search":
        result = index.search(args.question, embedder, args.k)
    else:
        result = evaluate(index, args.golden, embedder, args.k, args.strategy)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
