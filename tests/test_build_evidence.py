from src.retrievers.answer.evidence import (
    build_evidence,
    render_evidence_block,
)
from src.retrievers.search_result import SearchResult


STORE = {
    "EVD_1": {"document_id": "DOC_1", "title": "농약 등록 정보", "url": "https://a",
              "text": "플루아지남 분제는 배추 반쪽시들음병에 등록되어 있다.", "chunk_id": "CHK_1"},
    "EVD_2": {"document_id": "DOC_2", "title": "병해 방제 지침", "url": "https://b",
              "text": "반쪽시들음병은 토양 전염성 병해다.", "chunk_id": "CHK_2"},
    "EVD_3": {"document_id": "DOC_3", "title": "원문 없는 근거", "url": "https://c",
              "text": "   ", "chunk_id": "CHK_3"},
}


def _lookup(ids):
    return {eid: STORE[eid] for eid in ids if eid in STORE}


def _vector_result(evidence_id="EVD_1", chunk_id="CHK_1"):
    return SearchResult(
        source="vector",
        text="공식 재배 문서 내용",
        score=0.91,
        evidence_ids=[evidence_id],
        source_ids=["SRC_1"],
        chunk_ids=[chunk_id],
    )


def _graph_result(evidence_ids=("EVD_1", "EVD_2")):
    return SearchResult(
        source="graph",
        text="배추 → 플루아지남 → 반쪽시들음병",
        evidence_ids=list(evidence_ids),
        path=["배추", "플루아지남", "반쪽시들음병"],
        relation_types=["REGISTERED_FOR", "CONTROLS"],
        metadata={"relation_ids": ["REL_1", "REL_2"], "hop_count": 2},
    )


def test_vector_result_becomes_one_evidence_without_graph_fields():
    evidences, dropped = build_evidence([_vector_result()], _lookup)

    assert dropped == []
    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.evidence_id == "EVD_1"
    assert evidence.quote == STORE["EVD_1"]["text"]
    assert evidence.title == "농약 등록 정보"
    assert evidence.url == "https://a"
    # 벡터 근거에는 관계·경로가 없어야 합니다.
    assert evidence.graph_path == []
    assert evidence.relation_ids == []


def test_graph_result_expands_into_one_evidence_per_id():
    evidences, dropped = build_evidence([_graph_result()], _lookup)

    assert dropped == []
    # 경로 하나에 근거 ID가 둘이면 인용 단위도 둘입니다.
    assert [e.evidence_id for e in evidences] == ["EVD_1", "EVD_2"]
    for evidence in evidences:
        assert evidence.graph_path == ["배추", "플루아지남", "반쪽시들음병"]
        assert evidence.relation_ids == ["REL_1", "REL_2"]


def test_duplicate_evidence_id_keeps_the_higher_ranked_result():
    # merge_results가 정렬한 순서대로 들어오므로 먼저 온 쪽이 남아야 합니다.
    evidences, _ = build_evidence(
        [_vector_result("EVD_1"), _graph_result(("EVD_1", "EVD_2"))], _lookup
    )

    assert [e.evidence_id for e in evidences] == ["EVD_1", "EVD_2"]
    assert evidences[0].source == "vector"
    assert evidences[0].graph_path == []


def test_unknown_evidence_id_is_dropped_with_reason():
    evidences, dropped = build_evidence([_vector_result("EVD_MISSING")], _lookup)

    assert evidences == []
    assert dropped == [{"evidence_id": "EVD_MISSING", "reason": "NOT_IN_STORE"}]


def test_blank_source_text_is_dropped_with_reason():
    evidences, dropped = build_evidence([_vector_result("EVD_3")], _lookup)

    assert evidences == []
    assert dropped == [{"evidence_id": "EVD_3", "reason": "NO_SOURCE_TEXT"}]


def test_lookup_is_called_once_with_deduplicated_ids():
    calls = []

    def recording_lookup(ids):
        calls.append(list(ids))
        return _lookup(ids)

    build_evidence([_vector_result("EVD_1"), _graph_result(("EVD_1", "EVD_2"))],
                   recording_lookup)

    # 결과마다 조회하지 않고 한 번에 모아 조회해야 합니다.
    assert calls == [["EVD_1", "EVD_2"]]


def test_render_block_exposes_only_the_evidence_id_handle():
    evidences, _ = build_evidence([_graph_result()], _lookup)
    block = render_evidence_block(evidences)

    assert "[1] evidence_id: EVD_1" in block
    assert "[2] evidence_id: EVD_2" in block
    assert "배추 → 플루아지남 → 반쪽시들음병" in block
    # chunk_id가 프롬프트에 새면 LLM이 잘못된 핸들을 인용합니다.
    assert "CHK_" not in block