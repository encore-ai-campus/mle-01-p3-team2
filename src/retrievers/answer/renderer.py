"""검증된 구조화 답변을 사용자 표시용 문자열로 변환합니다."""

from __future__ import annotations

import re

from .cited_answer import CitedAnswer
from .evidence import Evidence

# UI가 본문과 출처를 나눌 때 쓰는 고정 구분자
SOURCES_HEADING = "### 📚 출처"

# 보류 사유의 내부 코드를 본문에서 떼어내 토글로 보여줄 때 쓰는 구분자
WITHHOLD_DETAIL_HEADING = "<!--WITHHOLD-DETAIL-->"

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

_TOKEN_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

def _apply_inline_markers(body: str, source_numbers: dict[str, int]) -> str:
    """본문의 [[evidence_id]] 토큰을 출력 변호 [n]으로 치환한다.
    
    source_numbers에 없는 ID(미사용·누락 근거) 토큰은 제거한다.
    """

    def _replace(match: "re.Match[str]") -> str:
        evidence_id = match.group(1).strip()
        number = source_numbers.get(evidence_id)
        return f"[{number}]" if number is not None else ""

    replaced = _TOKEN_RE.sub(_replace, body)
    # 토큰 제거로 생긴 구두점 앞 공백 정리
    return re.sub(r"[ \t]+([.?!,])", r"\1", replaced)

def render_cited_answer(answer: CitedAnswer, evidences: list[Evidence]) -> str:
    """답변 본문 뒤에 실제 근거의 제목/URL/원문을 표시"""

    evidence_map = {evidence.evidence_id: evidence for evidence in evidences}

    used_ids = collect_used_evidence_ids(answer)

    # 인용 ID별 사용자 출력 번호
    # evidence_map에 없는 ID를 먼저 걸러낸 뒤 번호를 매겨, 출력 번호가
    # 항상 1부터 빠짐없이 이어지도록 한다.
    source_numbers = {
        evidence_id: index
        for index, evidence_id in enumerate(
            (eid for eid in used_ids if eid in evidence_map), 1
        )
    }

    lines: list[str] = [_apply_inline_markers(answer.answer.strip(), source_numbers)]

    if answer.claims:
        lines.extend(["", "주장별 근거"])

        for claim in answer.claims:
            numbers = [
                source_numbers[citation.evidence_id] for citation in claim.citations
                if citation.evidence_id in source_numbers
            ]
            markers = "".join(f"**[{number}]**" for number in numbers)
            lines.append(f"- {claim.text} {markers}".rstrip())

    if source_numbers:
        lines.extend(["", SOURCES_HEADING])

        for evidence_id, number in source_numbers.items():
            evidence = evidence_map[evidence_id]

            title = evidence.title or "제목 없음"
            if evidence.url:
                title_line = f"**[{number}]** [{title}]({evidence.url})"
            else:
                title_line = f"**[{number}]** {title} (URL 없음)"

            lines.extend([
                title_line, f"> {evidence.quote}"
            ])

            if evidence.graph_path:
                path = " → ".join(evidence.graph_path)
                lines.append(f"**그래프 경로**: {path}")

            lines.append("")

    return "\n".join(lines).strip()