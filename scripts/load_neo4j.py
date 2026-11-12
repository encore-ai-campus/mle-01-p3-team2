"""전처리된 엔터티·관계 그래프를 Neo4j에 적재합니다.

프로젝트에서 제공한 entity_id와 relation_id를 고유 키로 사용하므로,
스크립트를 다시 실행해도 그래프 데이터가 중복 생성되지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


ROOT = Path(__file__).parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_records(
    entities: Iterable[dict[str, Any]], relations: Iterable[dict[str, Any]]
) -> list[str]:
    entities = list(entities)
    relations = list(relations)
    errors: list[str] = []
    entity_ids = [row.get("entity_id") for row in entities]
    duplicate_ids = {key for key in entity_ids if entity_ids.count(key) > 1}
    if duplicate_ids:
        errors.append(f"duplicate entity_id values: {sorted(duplicate_ids)[:5]}")

    known_ids = set(entity_ids)
    missing = [
        (row.get("relation_id"), endpoint)
        for row in relations
        for endpoint in (row.get("from_id"), row.get("to_id"))
        if endpoint not in known_ids
    ]
    if missing:
        errors.append(f"unresolved relation endpoints: {missing[:5]}")

    relation_ids = [row.get("relation_id") for row in relations]
    if len(relation_ids) != len(set(relation_ids)):
        errors.append("duplicate relation_id values")
    return errors


def prepare_entities(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["attributes"] = json.dumps(row.get("attributes") or {}, ensure_ascii=False)
        prepared.append(item)
    return prepared


def prepare_relations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["properties"] = json.dumps(row.get("properties") or {}, ensure_ascii=False)
        prepared.append(item)
    return prepared


def create_constraints(
    tx: Any, entity_types: list[str], relation_types: list[str]
) -> None:
    for entity_type in entity_types:
        tx.run(
            f"CREATE CONSTRAINT `{entity_type.lower()}_id_unique` IF NOT EXISTS "
            f"FOR (n:`{entity_type}`) REQUIRE n.entity_id IS UNIQUE"
        )
    for relation_type in relation_types:
        tx.run(
            f"CREATE CONSTRAINT `{relation_type.lower()}_id_unique` IF NOT EXISTS "
            f"FOR ()-[r:`{relation_type}`]-() REQUIRE r.relation_id IS UNIQUE"
        )


def write_entities(tx: Any, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity_type = row["entity_type"]
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", entity_type):
            raise ValueError(f"Invalid Neo4j label: {entity_type!r}")
        grouped.setdefault(entity_type, []).append(row)

    for entity_type, entity_rows in grouped.items():
        tx.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:`{entity_type}` {{entity_id: row.entity_id}})
            SET n.name = row.name,
                n.entity_type = row.entity_type,
                n.attributes = row.attributes,
                n.quality_flags = row.quality_flags,
                n.source_ids = row.source_ids,
                n.source_chunk_ids = row.source_chunk_ids,
                n.evidence_ids = row.evidence_ids
            """,
            rows=entity_rows,
        )


def write_relations(tx: Any, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        relation_type = row["relation_type"]
        from_type = row["from_type"]
        to_type = row["to_type"]
        for value, kind in ((relation_type, "relationship type"), (from_type, "source label"), (to_type, "target label")):
            if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", value):
                raise ValueError(f"Invalid Neo4j {kind}: {value!r}")
        grouped.setdefault((relation_type, from_type, to_type), []).append(row)

    for (relation_type, from_type, to_type), relation_rows in grouped.items():
        tx.run(
            f"""
            UNWIND $rows AS row
            MATCH (source:`{from_type}` {{entity_id: row.from_id}})
            MATCH (target:`{to_type}` {{entity_id: row.to_id}})
            MERGE (source)-[r:`{relation_type}` {{relation_id: row.relation_id}}]->(target)
            SET r.confidence = row.confidence,
                r.properties = row.properties,
                r.retrieval_eligible = row.retrieval_eligible,
                r.source_ids = row.source_ids,
                r.source_chunk_ids = row.source_chunk_ids,
                r.evidence_ids = row.evidence_ids,
                r.exclusion_reasons = row.exclusion_reasons
            """,
            rows=relation_rows,
        )


def batched(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def load_graph(driver: Driver, entities: list[dict[str, Any]], relations: list[dict[str, Any]], batch_size: int) -> None:
    entity_types_by_id = {row["entity_id"]: row["entity_type"] for row in entities}
    relations = [
        {
            **row,
            "from_type": entity_types_by_id[row["from_id"]],
            "to_type": entity_types_by_id[row["to_id"]],
        }
        for row in relations
    ]
    with driver.session() as session:
        entity_types = sorted({row["entity_type"] for row in entities})
        relation_types = sorted({row["relation_type"] for row in relations})
        session.execute_write(create_constraints, entity_types, relation_types)
        for batch in batched(entities, batch_size):
            session.execute_write(write_entities, batch)
        for batch in batched(relations, batch_size):
            session.execute_write(write_relations, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    entities = load_records(args.data_dir / "entities.jsonl")
    relations = load_records(args.data_dir / "relations.jsonl")
    errors = validate_records(entities, relations)
    if errors:
        raise SystemExit("Input validation failed:\n- " + "\n- ".join(errors))

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise SystemExit("NEO4J_PASSWORD is required")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        load_graph(
            driver,
            prepare_entities(entities),
            prepare_relations(relations),
            args.batch_size,
        )
    finally:
        driver.close()
    print(f"Loaded {len(entities):,} entities and {len(relations):,} relations.")


if __name__ == "__main__":
    main()
