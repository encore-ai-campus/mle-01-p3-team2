import pytest

from src.retrievers.search_result import SearchResult


def test_single_ids_and_warnings_are_available_with_plural_ids():
    result = SearchResult(
        source="vector",
        text="Original passage",
        score=0.91,
        evidence_id="EVD_1",
        chunk_id="CHK_1",
        warnings=["Check source date"],
    )

    assert result.evidence_ids == ["EVD_1"]
    assert result.chunk_ids == ["CHK_1"]
    assert result.warnings == ["Check source date"]


def test_graph_path_keeps_multiple_ids_without_arbitrary_single_id():
    result = SearchResult(
        source="graph",
        text="Crop to region",
        evidence_ids=["EVD_1", "EVD_2"],
        chunk_ids=["CHK_1", "CHK_2"],
    )

    assert result.evidence_id is None
    assert result.chunk_id is None
    assert result.evidence_ids == ["EVD_1", "EVD_2"]
    assert result.chunk_ids == ["CHK_1", "CHK_2"]
    assert result.warnings == []


def test_existing_singleton_lists_expose_singular_ids():
    result = SearchResult(
        source="vector",
        text="Original passage",
        evidence_ids=["EVD_1"],
        chunk_ids=["CHK_1"],
    )

    assert result.evidence_id == "EVD_1"
    assert result.chunk_id == "CHK_1"


def test_conflicting_single_and_plural_ids_are_rejected():
    with pytest.raises(ValueError, match="evidence_id conflicts"):
        SearchResult(
            source="vector",
            text="Original passage",
            evidence_id="EVD_1",
            evidence_ids=["EVD_2"],
        )
