
"""Shared result contract for graph and vector retrievers."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypedDict


SourceType: TypeAlias = Literal["graph", "vector"]
RouteType: TypeAlias = Literal["GRAPH", "VECTOR", "HYBRID", "WITHHOLD"]


class GraphPathNode(TypedDict):
    entity_id: str
    name: str
    entity_type: str


class OriginalChunk(TypedDict):
    chunk_id: str
    evidence_id: str
    source_id: str
    text: str


class GraphPathEdge(TypedDict):
    hop: int
    relation_id: str | None
    relation_type: str
    from_entity_id: str
    to_entity_id: str
    evidence_ids: list[str]
    chunk_ids: list[str]
    source_ids: list[str]
    original_chunks: list[OriginalChunk]


@dataclass(slots=True)
class SearchResult:
    """One retrieval hit. Singular IDs identify a single evidence/chunk when possible.

    Graph paths may cite several IDs; their full sets remain in the plural fields.
    A singular ID is None when its plural field contains multiple values.
    """

    source: SourceType
    text: str

    score: float | None = None
    evidence_id: str | None = None
    chunk_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    path: list[str] | None = None
    path_nodes: list[GraphPathNode] | None = None
    path_edges: list[GraphPathEdge] | None = None
    relation_types: list[str] | None = None
    hop_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.evidence_id, self.evidence_ids = self._sync_id(
            self.evidence_id, self.evidence_ids, "evidence_id"
        )
        self.chunk_id, self.chunk_ids = self._sync_id(
            self.chunk_id, self.chunk_ids, "chunk_id"
        )

    @staticmethod
    def _sync_id(
        singular: str | None, plural: list[str], field_name: str
    ) -> tuple[str | None, list[str]]:
        if singular is not None:
            if plural and (len(plural) != 1 or plural[0] != singular):
                raise ValueError(f"{field_name} conflicts with its plural ID list")
            return singular, [singular]
        return (plural[0] if len(plural) == 1 else None), plural


@dataclass(slots=True)
class RetrievalResponse:
    """Unified response returned by the retrieval router."""

    route: RouteType
    results: list[SearchResult] = field(default_factory=list)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
