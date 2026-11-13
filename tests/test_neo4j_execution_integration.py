
"""실제 Neo4j 농약 데이터와 Execution Rail 통합 테스트.

실행 전 조건:
- .env에 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD 설정
- 팀 Neo4j DB 접근 가능
- PesticideUse 노드에 등록정보 존재

이 테스트는 DB를 읽기만 하며 수정하지 않는다.
"""

import json

from dotenv import load_dotenv

from src.retrievers.graph_retriever import GraphRetriever
from src.guardrails.neo4j_pesticide_lookup import (
    make_neo4j_pesticide_lookup,
)
from src.guardrails.execution_rail import get_registered_usage
from src.guardrails.execution_adapter import execution_node


load_dotenv()


def get_sample_record():
    """실제 DB에서 테스트 가능한 농약 레코드 하나를 조회."""

    db = GraphRetriever.from_env()

    try:
        with db.driver.session() as session:
            rows = session.run(
                """
                MATCH (p:PesticideUse)
                RETURN p.attributes AS attributes
                LIMIT 100
                """
            )

            for row in rows:
                raw = row["attributes"]

                if not isinstance(raw, str):
                    continue

                try:
                    attrs = json.loads(raw)
                except (ValueError, TypeError):
                    continue

                if not isinstance(attrs, dict):
                    continue

                if all(
                    isinstance(attrs.get(key), str)
                    and attrs.get(key).strip()
                    for key in ("product", "crop", "target_name")
                ):
                    return attrs

    finally:
        db.close()

    raise AssertionError(
        "DB에서 테스트 가능한 PesticideUse 레코드를 찾지 못했습니다."
    )


def test_real_neo4j_lookup():
    """DB에서 품목명·작물·병해충로 실제 조회."""

    attrs = get_sample_record()
    lookup = make_neo4j_pesticide_lookup()

    records = lookup(
        attrs["product"],
        attrs["crop"],
        attrs["target_name"],
    )

    assert records, "Neo4j 조회 결과가 비어 있습니다."

    assert any(
        record["product_name"] == attrs["product"]
        and record["crop"] == attrs["crop"]
        and record["target_pest"] == attrs["target_name"]
        for record in records
    )


def test_real_neo4j_lookup_through_execution_rail():
    """실제 DB 조회 결과가 Execution Rail까지 전달되는지 확인."""

    attrs = get_sample_record()
    lookup = make_neo4j_pesticide_lookup()

    result = get_registered_usage(
        product_name=attrs["product"],
        crop=attrs["crop"],
        target_pest=attrs["target_name"],
        lookup_fn=lookup,
        recommend=False,
    )

    assert result.decision in ("WARN", "FALLBACK")
    assert result.decision != "ALLOW"


def test_real_neo4j_unverified_use_is_not_allowed():
    """최신 공식 검증이 없으면 농약 사용 추천을 허용하지 않는다."""

    attrs = get_sample_record()
    lookup = make_neo4j_pesticide_lookup()

    result = get_registered_usage(
        product_name=attrs["product"],
        crop=attrs["crop"],
        target_pest=attrs["target_name"],
        lookup_fn=lookup,
        recommend=True,
        current_verifier=None,
    )

    assert result.allowed is False
    assert result.decision in ("BLOCK", "FALLBACK")


def test_execution_adapter_uses_real_database():
    """Execution Adapter가 실제 DB 조회 함수를 호출하는지 확인."""

    attrs = get_sample_record()
    lookup = make_neo4j_pesticide_lookup()

    state = {
        "question": "수집된 농약 등록정보를 알려줘",
        "input_intent": "PESTICIDE_LOOKUP",
        "evidence_status": "PASS",
        "product_name": attrs["product"],
        "crop": attrs["crop"],
        "target_pest": attrs["target_name"],
        "warnings": [],
    }

    result = execution_node(
        state,
        lookup_fn=lookup,
    )

    assert result["execution_decision"] in (
        "WARN",
        "FALLBACK",
    )

    if result["execution_decision"] == "WARN":
        assert result["execution_record"] is not None
    else:
        assert result["final_status"] == "WITHHELD"
