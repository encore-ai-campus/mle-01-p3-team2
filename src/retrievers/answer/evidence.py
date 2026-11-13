"""검색 결과를 인용 가능한 공통 근거(Evidence) 형식으로 변환합니다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.retrievers.search_result import SearchResult


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    document_id: str
    title: str
    url: str
    quote: str
    relation_ids: list[str] = field(default_factory=list)
    graph_path: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    # 아래는 진단용. LLM 프롬프트에는 넣지 않음
    source: str = "vector"
    chunk_ids: list[str] = field(default_factory=list)


def build_evidence(
    results: list[SearchResult],
    lookup: Callable[[list[str]], dict[str, dict]],
) -> tuple[list[Evidence], list[dict]]:
    """검색 결과를 인용 가능한 Evidence 목록으로 펼칩니다.

    반환: (evidences, dropped) — dropped는 원문을 못 찾아 인용 불가가 된 ID와 사유
    """
    ids = list(dict.fromkeys(eid for r in results for eid in r.evidence_ids))
    meta = lookup(ids) if ids else {}

    evidences: list[Evidence] = []
    dropped: list[dict] = []
    seen: dict[str, Evidence] = {}
    for result in results:              # hybrid_score 정렬 순서를 그대로 유지
        for eid in result.evidence_ids:
            is_graph = result.source == "graph"
            relation_ids = list(result.metadata.get("relation_ids") or []) if is_graph else []
            if is_graph and result.path_edges:
                relation_ids = list(dict.fromkeys(
                    str(edge["relation_id"])
                    for edge in result.path_edges
                    if edge.get("relation_id") and eid in (edge.get("evidence_ids") or [])
                ))
            if eid in seen:
                existing = seen[eid]
                if is_graph and existing.source == "graph":
                    for relation_id in relation_ids:
                        if relation_id not in existing.relation_ids:
                            existing.relation_ids.append(relation_id)
                    if result.path and result.path not in existing.graph_paths:
                        existing.graph_paths.append(list(result.path))
                continue
            row = meta.get(eid)
            if row is None or not (row.get("text") or "").strip():
                dropped.append({"evidence_id": eid,
                                "reason": "NO_SOURCE_TEXT" if row else "NOT_IN_STORE"})
                continue
            evidence = Evidence(
                evidence_id=eid,
                document_id=row.get("document_id") or "",
                title=row.get("title") or "",
                url=row.get("url") or "",
                quote=row["text"],
                relation_ids=relation_ids,
                graph_path=list(result.path or []) if is_graph else [],
                graph_paths=[list(result.path)] if is_graph and result.path else [],
                source=result.source,
                chunk_ids=list(result.chunk_ids),
            )
            evidences.append(evidence)
            seen[eid] = evidence
    return evidences, dropped

def render_evidence_block(evidences: list[Evidence]) -> str:
    """LLM 프롬프트에 넣을 근거 블록을 만듭니다.

    출처 번호는 1부터이고, merge_results가 정렬해 둔 순서를 그대로 씁니다.
    """
    parts: list[str] = []
    for index, evidence in enumerate(evidences, 1):
        paths = evidence.graph_paths or ([evidence.graph_path] if evidence.graph_path else [])
        path = "; ".join(" → ".join(nodes) for nodes in paths) if paths else "없음"
        parts.append(
            f"[{index}] evidence_id: {evidence.evidence_id}\n"
            f"제목: {evidence.title}\n"
            f"URL: {evidence.url}\n"
            f"그래프 경로: {path}\n"
            f"관계 ID: {', '.join(evidence.relation_ids) or '없음'}\n"
            f"원문: {evidence.quote}"
        )
    return "\n\n".join(parts)