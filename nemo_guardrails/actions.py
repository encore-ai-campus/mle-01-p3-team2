"""NeMo Guardrails 0.23.0 custom action entry points."""

from collections.abc import Mapping
from typing import Any

from nemoguardrails.actions import action

from guardrails_core import GuardrailRequest, validate_request


@action(name="validate_graph_rag_response", execute_async=True)
async def validate_graph_rag_response(
    context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the shared validator pipeline from a Colang action."""

    payload = dict(context) if isinstance(context, Mapping) else {}
    payload.update(kwargs)
    return validate_request(GuardrailRequest.from_mapping(payload)).to_dict()


__all__ = ["validate_graph_rag_response"]
