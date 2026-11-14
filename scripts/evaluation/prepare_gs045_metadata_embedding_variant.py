"""Prepare a metadata-enriched GS045 embedding-text variant for offline testing.

This only writes a separate experiment input. It does not update chunks, vectors,
the Neo4j index, golden data, or the service retriever.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluation.evaluate_multihop import DEFAULT_GOLDEN_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"
DEFAULT_OUTPUT = REPORT_DIR / "gs045_metadata_embedding_variant_v1.json"
TARGET_CASE_ID = "GS045"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    golden = {row["case_id"]: row for row in _read_jsonl(DEFAULT_GOLDEN_PATH)}
    case = golden[TARGET_CASE_ID]
    target_ids = set(case.get("expected_source_chunk_ids") or [])
    chunks = {
        row["chunk_id"]: row
        for row in _read_jsonl(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
        if row.get("chunk_id") in target_ids
    }
    if target_ids - chunks.keys():
        raise ValueError(f"Missing processed chunks: {sorted(target_ids - chunks.keys())}")

    entities = _read_jsonl(PROJECT_ROOT / "data" / "processed" / "entities.jsonl")
    matches = [
        entity
        for entity in entities
        if entity.get("entity_type") == "Disease"
        and target_ids.intersection(entity.get("source_chunk_ids") or [])
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one disease entity for GS045 chunks; found {len(matches)}")
    entity = matches[0]
    attrs = entity.get("attributes") or {}
    crop = attrs.get("crop") or ""
    disease = entity.get("name") or ""
    metadata_prefix = " ".join(part for part in (f"작물명: {crop}" if crop else "", f"병해명: {disease}" if disease else "") if part)

    variants = []
    for chunk_id in sorted(target_ids):
        chunk = chunks[chunk_id]
        original = chunk.get("embedding_text") or ""
        variants.append({
            "chunk_id": chunk_id,
            "document_id": chunk.get("document_id"),
            "evidence_id": chunk.get("evidence_id"),
            "original_embedding_text": original,
            "metadata_prefix": metadata_prefix,
            "candidate_embedding_text": f"{metadata_prefix}\n{original}" if metadata_prefix else original,
            "source_metadata": {
                "entity_id": entity.get("entity_id"),
                "entity_type": entity.get("entity_type"),
                "entity_name": disease,
                "crop": crop,
                "sick_key": attrs.get("sick_key"),
                "evidence_ids": entity.get("evidence_ids") or [],
            },
        })

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": TARGET_CASE_ID,
        "question": case["question"],
        "expected_chunk_ids": sorted(target_ids),
        "experiment": "metadata-prefix-only-alternate-embedding-text",
        "variants": variants,
        "constraints": {
            "golden_data_modified": False,
            "processed_data_modified": False,
            "stored_vectors_modified": False,
            "neo4j_index_modified": False,
            "service_retriever_modified": False,
            "next_step": "Embed candidate_embedding_text offline and compare direct cosine / retrieval on an isolated evaluation path.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"GS045 variant prepared: {args.output.resolve()}")
    print(f"Metadata prefix: {metadata_prefix}")
    print(f"Target chunk count: {len(variants)}")


if __name__ == "__main__":
    main()
