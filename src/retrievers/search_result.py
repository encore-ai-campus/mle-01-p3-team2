
"""Shared result contract for graph and vector retrievers."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


SourceType: TypeAlias = Literal["graph", "vector"]
RouteType: TypeAlias = Literal["GRAPH", "VECTOR", "HYBRID", "WITHHOLD"]


@dataclass(slots=True)
class SearchResult:
    """Normalized search result from graph or vector retrieval."""

    source: SourceType
    text: str

    score: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    path: list[str] | None = None
    relation_types: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResponse:
    """Unified response returned by the retrieval router."""

    route: RouteType
    results: list[SearchResult] = field(default_factory=list)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
