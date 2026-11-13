"""evidence_id로 원문·제목·URL을 조회하는 lookup 구현입니다."""

from __future__ import annotations

from typing import Any


LOOKUP_QUERY = (
    "MATCH (c:Chunk) WHERE c.evidence_id IN $ids "
    "RETURN c.evidence_id AS evidence_id, c.document_id AS document_id, "
    "c.title AS title, c.source_url AS url, c.raw_text AS text, "
    "c.chunk_id AS chunk_id"
)

def neo4j_lookup(driver: Any, ids: list[str], database: str = "neo4j") -> dict[str, dict]:
    """evidence_id → {document_id, title, url, text, chunk_id}"""
    records, _, _ = driver.execute_query(LOOKUP_QUERY, ids=list(ids), database_=database)
    return {r["evidence_id"]: dict(r) for r in records}
