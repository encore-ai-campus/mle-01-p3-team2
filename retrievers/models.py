from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SearchResult:
    source: Literal["graph", "vector"]
    text: str
    score: float | None
    evidence_ids: list[str]
    source_ids: list[str]
    chunk_ids: list[str]
    path: list[str] | None
    relation_types: list[str] | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResponse:
    route: Literal["GRAPH", "VECTOR", "HYBRID", "WITHHOLD"]
    results: list[SearchResult]
    reason: str
    warnings: list[str] = field(default_factory=list)
