"""검증된 구조화 답변을 사용자 표시용 문자열로 변환합니다."""

from __future__ import annotations

from .cited_answer import CitedAnswer
from .evidence import Evidence

def collect_used_evidence_ids(answer: CitedAnswer) -> list[str]:
    """답변에서 인용한 evidence_id를 최초 등장 순서로 수집합니다."""

    used_ids: list[str] = []
    seen: set[str] = set()

    for claim in answer.claims:
        for citation in claim.citations:
            if citation.evidence_id in seen:
                continue

            seen.add(citation.evidence_id)
            used_ids.append(citation.evidence_id)

    return used_ids

def render_cited_answer(answer: CitedAnswer, evidences: list[Evidence]) -> str:
    """답변 본문 뒤에 실제 근거의 제목/URL/원문을 표시"""

    evidence_map = {evidence.evidence_id: evidence for evidence in evidences}

    used_ids = collect_used_evidence_ids(answer)

    # 인용 ID별 사용자 출력 번호
    source_numbers = {
        evidence_id: index for index, evidence_id in enumerate(used_ids, 1)
        if evidence_id in evidence_map
    }

    lines: list[str] = [answer.answer.strip()]

    if answer.claims:
        lines.extend(["", "주장별 근거"])

        for claim in answer.claims:
            numbers = [
                source_numbers[citation.evidence_id] for citation in claim.citations
                if citation.evidence_id in source_numbers
            ]
            markers = "".join(f"[{number}]" for number in numbers)
            lines.append(f"- {claim.text} {markers}".rstrip())

    if source_numbers:
        lines.extend(["", "출처"])

        for evidence_id, number in source_numbers.items():
            evidence = evidence_map[evidence_id]

            lines.extend([
                f"[{number}] {evidence.title or '제목 없음'}",
                f"URL: {evidence.url or 'URL 없음'}",
                f"원문: '{evidence.quote}'"
            ])

            if evidence.graph_path:
                path = " → ".join(evidence.graph_path)
                lines.append(f"그래프 경로: {path}")

            lines.append("")

    return "\n".join(lines).strip()