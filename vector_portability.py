"""Share and restore the project's Neo4j Chunk vectors as one ZIP file."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import json
import os
import time
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ARCHIVE = Path(__file__).resolve().parent / "data" / "portable" / "chunk_vectors_bge_m3.zip"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIMENSION = 1024
INDEX_NAME = "chunk_embedding"


def _required_metadata(metadata: dict, model_name: str, dimension: int) -> None:
    for field in ("chunk_id", "source_id", "evidence_id", "raw_text"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty {field}")
    if not isinstance(metadata.get("source_official"), bool):
        raise ValueError("Missing or invalid source_official")
    if metadata.get("embedding_model") != model_name:
        raise ValueError("Embedding model differs from archive model")
    if metadata.get("embedding_dimension") != dimension:
        raise ValueError("Embedding dimension differs from archive dimension")


def write_portable(
    rows: Iterable[dict], archive_path: str | Path, model_name: str, dimension: int,
) -> dict:
    """Stream ordered Chunk records into a single portable ZIP archive."""
    if not model_name.strip() or dimension < 1:
        raise ValueError("Model name and dimension must be valid")
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Archive already exists: {target}")

    # ZIP writes one entry at a time. Spool metadata to disk rather than RAM.
    vector_hash = hashlib.sha256()
    metadata_hash = hashlib.sha256()
    count = 0
    previous_id = None
    manifest = {}
    try:
        with tempfile.TemporaryFile(mode="w+b") as metadata_spool:
            with zipfile.ZipFile(target, mode="x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                with archive.open("vectors.f32", mode="w", force_zip64=True) as vectors_file:
                    for row in rows:
                        if not isinstance(row, dict):
                            raise ValueError("Chunk must be an object")
                        metadata = {key: value for key, value in row.items() if key != "embedding"}
                        _required_metadata(metadata, model_name, dimension)
                        chunk_id = metadata["chunk_id"]
                        if previous_id is not None and chunk_id <= previous_id:
                            raise ValueError("Chunk IDs must be unique and sorted")
                        previous_id = chunk_id
                        vector = np.asarray(row.get("embedding"), dtype="<f4")
                        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
                            raise ValueError(f"Invalid embedding for {chunk_id}")
                        vector_bytes = vector.tobytes()
                        vectors_file.write(vector_bytes)
                        vector_hash.update(vector_bytes)
                        metadata_bytes = (json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
                                          + "\n").encode("utf-8")
                        metadata_spool.write(metadata_bytes)
                        metadata_hash.update(metadata_bytes)
                        count += 1
                if not count:
                    raise ValueError("No Chunk vectors found")
                metadata_spool.seek(0)
                with archive.open("metadata.jsonl", mode="w", force_zip64=True) as stream:
                    shutil.copyfileobj(metadata_spool, stream, length=1024 * 1024)
                manifest = {
                    "format_version": 1,
                    "model_name": model_name,
                    "dimension": dimension,
                    "count": count,
                    "metric": "cosine",
                    "vector_dtype": "float32_le",
                    "vector_file": "vectors.f32",
                    "metadata_file": "metadata.jsonl",
                    "vectors_sha256": vector_hash.hexdigest(),
                    "metadata_sha256": metadata_hash.hexdigest(),
                    "index_name": INDEX_NAME,
                }
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    except BaseException:
        # Never leave a partial ZIP that could be mistaken for a valid export.
        target.unlink(missing_ok=True)
        raise
    return manifest


def verify_portable(archive_path: str | Path) -> dict:
    """Check file integrity, record count, and vector/metadata alignment."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != 3 or set(names) != {"manifest.json", "metadata.jsonl", "vectors.f32"}:
                raise ValueError("Archive integrity: unexpected files")
            manifest = json.loads(archive.read("manifest.json"))
            if (not isinstance(manifest, dict)
                    or manifest.get("format_version") != 1
                    or type(manifest.get("count")) is not int or manifest["count"] < 1
                    or type(manifest.get("dimension")) is not int or manifest["dimension"] < 1
                    or not isinstance(manifest.get("model_name"), str)
                    or not manifest["model_name"].strip()
                    or manifest.get("vector_dtype") != "float32_le"
                    or manifest.get("metric") != "cosine"
                    or manifest.get("vector_file") != "vectors.f32"
                    or manifest.get("metadata_file") != "metadata.jsonl"
                    or manifest.get("index_name") != INDEX_NAME):
                raise ValueError("Archive integrity: invalid manifest")
            hashes = (("vectors.f32", "vectors_sha256"),
                      ("metadata.jsonl", "metadata_sha256"))
            for filename, key in hashes:
                digest = hashlib.sha256()
                with archive.open(filename) as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != manifest.get(key):
                    raise ValueError(f"Archive integrity: {filename} hash mismatch")
            if archive.getinfo("vectors.f32").file_size != manifest["count"] * manifest["dimension"] * 4:
                raise ValueError("Archive integrity: vector byte count mismatch")
            count = 0
            previous_id = None
            # Check vectors and metadata together, without loading them into memory.
            vector_size = manifest["dimension"] * 4
            with archive.open("metadata.jsonl") as raw_stream, archive.open("vectors.f32") as vectors:
                for line in raw_stream:
                    metadata = json.loads(line)
                    if not isinstance(metadata, dict):
                        raise ValueError("Archive integrity: metadata must be an object")
                    _required_metadata(metadata, manifest["model_name"], manifest["dimension"])
                    chunk_id = metadata["chunk_id"]
                    if previous_id is not None and chunk_id <= previous_id:
                        raise ValueError("Archive integrity: Chunk IDs are not unique and sorted")
                    previous_id = chunk_id
                    vector_bytes = vectors.read(vector_size)
                    if len(vector_bytes) != vector_size:
                        raise ValueError("Archive integrity: truncated vector")
                    if not np.all(np.isfinite(np.frombuffer(vector_bytes, dtype="<f4"))):
                        raise ValueError(f"Archive integrity: non-finite vector in {chunk_id}")
                    count += 1
                if vectors.read(1):
                    raise ValueError("Archive integrity: extra vector data")
            if count != manifest["count"]:
                raise ValueError("Archive integrity: metadata row count mismatch")
            return manifest
    except (zipfile.BadZipFile, KeyError, OSError, UnicodeError, json.JSONDecodeError,
            TypeError, OverflowError, EOFError) as exc:
        raise ValueError(f"Archive integrity: {exc}") from exc


def _iter_rows(archive_path: str | Path, manifest: dict) -> Iterator[dict]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("metadata.jsonl") as metadata_file, archive.open("vectors.f32") as vector_file:
            for line in metadata_file:
                metadata = json.loads(line)
                vector_size = manifest["dimension"] * 4
                vector_bytes = vector_file.read(vector_size)
                if len(vector_bytes) != vector_size:
                    raise ValueError("Archive integrity: truncated vector")
                metadata["embedding"] = np.frombuffer(vector_bytes, dtype="<f4").tolist()
                yield metadata


def iter_portable_rows(archive_path: str | Path) -> Iterator[dict]:
    manifest = verify_portable(archive_path)
    yield from _iter_rows(archive_path, manifest)


def export_neo4j(
    driver: Any, archive_path: str | Path = DEFAULT_ARCHIVE,
    model_name: str = DEFAULT_MODEL, dimension: int = DEFAULT_DIMENSION,
    database: str = "neo4j",
) -> dict:
    """Export only this model's Chunk nodes; unrelated graph data stays out."""
    with driver.session(database=database, fetch_size=256) as session:
        summary = session.run(
            "MATCH (c:Chunk) WHERE c.embedding_model = $model "
            "RETURN count(c) AS total, min(size(c.embedding)) AS min_dim, "
            "max(size(c.embedding)) AS max_dim, count(c.source_id) AS sources, "
            "count(c.evidence_id) AS evidences, count(c.raw_text) AS originals, "
            "count(c.source_official) AS official_statuses, "
            "count(c.embedding_dimension) AS dimensions",
            model=model_name,
        ).single()
        if (not summary or summary["total"] < 1
                or summary["min_dim"] != dimension or summary["max_dim"] != dimension
                or any(summary[field] != summary["total"]
                       for field in ("sources", "evidences", "originals", "official_statuses", "dimensions"))):
            raise ValueError("Neo4j Chunk vectors or metadata are incomplete")
        records = session.run(
            "MATCH (c:Chunk) WHERE c.embedding_model = $model "
            "RETURN properties(c) AS properties ORDER BY c.chunk_id",
            model=model_name,
        )
        manifest = write_portable(
            (dict(record["properties"]) for record in records),
            archive_path, model_name, dimension,
        )
        if manifest["count"] != summary["total"]:
            raise ValueError("Neo4j Chunk count changed during export")
        return manifest


def _show_vector_index(driver: Any, database: str) -> dict | None:
    rows, _, _ = driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, state, labelsOrTypes, properties, options "
        "WHERE name = $name RETURN name, state, labelsOrTypes, properties, options",
        name=INDEX_NAME, database_=database,
    )
    return dict(rows[0]) if rows else None


def _check_index(index: dict, dimension: int) -> None:
    config = index["options"].get("indexConfig", {})
    if (index["labelsOrTypes"] != ["Chunk"] or index["properties"] != ["embedding"]
            or config.get("vector.dimensions") != dimension
            or str(config.get("vector.similarity_function", "")).lower() != "cosine"):
        raise ValueError("Existing vector index has incompatible configuration")


def restore_portable(
    archive_path: str | Path, driver: Any, database: str = "neo4j",
    batch_size: int = 250, wait_seconds: int = 600,
) -> dict:
    """Restore Chunk nodes and the cosine vector index into Neo4j."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if wait_seconds < 0:
        raise ValueError("wait_seconds must be nonnegative")
    manifest = verify_portable(archive_path)
    existing_index = _show_vector_index(driver, database)
    if existing_index:
        _check_index(existing_index, manifest["dimension"])
    driver.execute_query(
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        database_=database,
    )
    write_query = (
        "UNWIND $rows AS row "
        "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
        "SET c += row.properties "
        "SET c.embedding = row.embedding"
    )
    batch = []
    for row in _iter_rows(archive_path, manifest):
        batch.append({
            "chunk_id": row["chunk_id"],
            "properties": {key: value for key, value in row.items()
                           if key not in ("chunk_id", "embedding")},
            "embedding": row["embedding"],
        })
        if len(batch) == batch_size:
            driver.execute_query(write_query, rows=batch, database_=database)
            batch = []
    if batch:
        driver.execute_query(write_query, rows=batch, database_=database)

    if not existing_index:
        driver.execute_query(
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {manifest['dimension']}, "
            "`vector.similarity_function`: 'cosine'}}",
            database_=database,
        )
    deadline = time.monotonic() + wait_seconds
    while True:
        index = _show_vector_index(driver, database)
        if index:
            _check_index(index, manifest["dimension"])
            if index["state"] == "ONLINE":
                return {"count": manifest["count"], "model_name": manifest["model_name"],
                        "dimension": manifest["dimension"], "index_state": "ONLINE"}
        if time.monotonic() >= deadline:
            raise TimeoutError("Vector index did not become ONLINE")
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
        if name != "verify":
            command.add_argument("--database", help="Neo4j database name (default: neo4j)")
        if name == "restore":
            command.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_portable(args.archive)
    else:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase

        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            parser.error("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
        database = args.database or os.getenv("NEO4J_DATABASE", "neo4j")
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            if args.command == "export":
                result = export_neo4j(driver, args.archive, database=database)
            else:
                result = restore_portable(args.archive, driver, database, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
