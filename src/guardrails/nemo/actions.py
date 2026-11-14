
"""NeMo Guardrails 0.23.0 사용자 정의 액션."""

from collections.abc import Mapping
from typing import Any

from nemoguardrails.actions import action

from src.guardrails.pipeline import (
    GuardrailRequest,
    validate_request,
)


@action(name="validate_graph_rag_response", execute_async=True)
async def validate_graph_rag_response(
    context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """기존 Guardrails 검증 파이프라인을 실행한다."""

    # NeMo에서 전달받은 실행 상태를 딕셔너리로 변환
    payload = dict(context) if isinstance(context, Mapping) else {}

    # 추가로 전달받은 인자를 실행 상태에 반영
    payload.update(kwargs)

    # 요청 데이터를 검증 파이프라인에 전달하고 결과 반환
    return validate_request(
        GuardrailRequest.from_mapping(payload)
    ).to_dict()


# NeMo에서 사용할 수 있는 액션을 명시
__all__ = ["validate_graph_rag_response"]
