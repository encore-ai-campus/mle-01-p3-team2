import json
from pathlib import Path

from scripts.load_neo4j import load_records, validate_records


def test_processed_graph_has_expected_counts_and_resolved_references():
    data_dir = Path(__file__).parents[1] / "data" / "processed"
    entities = load_records(data_dir / "entities.jsonl")
    relations = load_records(data_dir / "relations.jsonl")

    assert len(entities) == 30_933
    assert len(relations) == 46_034
    assert validate_records(entities, relations) == []

