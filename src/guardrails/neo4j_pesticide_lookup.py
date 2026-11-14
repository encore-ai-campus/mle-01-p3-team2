
"""Neo4j PesticideUse 조회 어댑터.

DB에는 attributes가 JSON 문자열로 저장되어 있다.
제품명(품목명)과 상표명 모두 검색할 수 있다.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from src.retrievers.graph_retriever import GraphRetriever


def _norm(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).casefold()


def _parse_record(
    raw_attributes: Any,
    entity_id: str | None,
    source_ids: list[str] | None,
    quality_flags: Any,
) -> dict[str, Any] | None:
    if not isinstance(raw_attributes, str):
        return None

    try:
        attrs = json.loads(raw_attributes)
    except (ValueError, TypeError):
        return None

    if not isinstance(attrs, dict):
        return None

    # 실제 DB 필드를 Execution Rail 표준 필드로 변환
    record = {
        "entity_id": entity_id,
        "product_name": attrs.get("product", ""),
        "brand": attrs.get("brand", ""),
        "crop": attrs.get("canonical_crop") or attrs.get("crop", ""),
        "target_pest": attrs.get("target_name", ""),
        "dilution": attrs.get("dilution", ""),
        "application_amount": attrs.get("use_quantity", ""),
        "application_timing": attrs.get("application_method", ""),
        "max_applications": attrs.get("use_count", ""),
        "preharvest_interval": attrs.get("use_timing", ""),
        "registration_no": attrs.get("registration_no", ""),
        "registration_status": attrs.get("registration_status", ""),
        "registration_valid_until": attrs.get(
            "registration_valid_until"
        ),
        "verification_status": attrs.get("validation_status"),
        "recommendation_eligible": attrs.get(
            "recommendation_eligible"
        ),
        "checked_at": attrs.get("checked_at"),
        "active_ingredient_english": attrs.get(
            "active_ingredient_english"
        ),
        "source_ids": source_ids or [],
        "quality_flags": quality_flags,
    }

    return record


def make_neo4j_pesticide_lookup():
    """Execution Rail이 호출할 읽기 전용 조회 함수를 만든다."""

    def lookup(
        product_name: str,
        crop: str,
        target_pest: str,
    ) -> list[dict[str, Any]]:

        if not all(
            isinstance(v, str) and v.strip()
            for v in (product_name, crop, target_pest)
        ):
            return []

        db = GraphRetriever.from_env()

        try:
            # JSON 문자열을 저장한 기존 구조에 맞춰
            # 작물/병해충 후보를 좁힌 뒤 Python에서 정확히 검사.
            query = """
            MATCH (p:PesticideUse)
            WHERE p.attributes CONTAINS $crop
              AND p.attributes CONTAINS $target
            RETURN
                p.entity_id AS entity_id,
                p.attributes AS attributes,
                p.source_ids AS source_ids,
                p.quality_flags AS quality_flags
            """

            with db.driver.session() as session:
                rows = session.run(
                    query,
                    crop=crop.strip(),
                    target=target_pest.strip(),
                )

                matched = []

                for row in rows:
                    record = _parse_record(
                        row["attributes"],
                        row["entity_id"],
                        row["source_ids"],
                        row["quality_flags"],
                    )

                    if record is None:
                        continue

                    # 문자열 포함 여부가 아닌 정확한 필드 비교
                    if _norm(record["crop"]) != _norm(crop):
                        continue

                    if _norm(record["target_pest"]) != _norm(
                        target_pest
                    ):
                        continue

                    # 품목명 또는 상표명으로 조회 가능
                    product_matches = (
                        _norm(record["product_name"])
                        == _norm(product_name)
                    )
                    brand_matches = (
                        _norm(record["brand"])
                        == _norm(product_name)
                    )

                    if not (product_matches or brand_matches):
                        continue

                    # Execution Rail이 요청 이름과
                    # 동일한 필드를 비교할 수 있도록 보정
                    matched.append(record)

                return matched

        finally:
            db.close()

    return lookup
