"""Shared contracts for deterministic GraphRAG guardrail validators."""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


Decision = Literal["ALLOW", "WARN", "FALLBACK", "BLOCK"]


@dataclass(frozen=True, slots=True)
class GuardrailRequest:
    """Input shared by evidence, pesticide, and output validators."""

    question: str
    answer: str
    evidence: Sequence[Any] = field(default_factory=tuple)
    expected_facts: Sequence[str] = field(default_factory=tuple)
    pesticide_use_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GuardrailRequest":
        """Build a request from a NeMo action or LangGraph mapping."""

        answer_value = payload.get("bot_message", payload.get("answer", ""))
        if not isinstance(answer_value, str):
            answer_value = ""
        evidence = payload.get("evidence", ())
        if evidence is None:
            evidence = ()
        expected_facts = payload.get("expected_facts", ())
        if expected_facts is None:
            expected_facts = ()

        return cls(
            question=payload.get("question", "") if isinstance(payload.get("question", ""), str) else "",
            answer=answer_value,
            evidence=evidence,
            expected_facts=expected_facts,
            pesticide_use_requested=bool(payload.get("pesticide_use_requested", False)),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Stable result returned by every guardrail layer."""

    allowed: bool
    decision: Decision
    reason: str
    warnings: tuple[str, ...] = ()
    fallback_text: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "fallback_text": self.fallback_text,
            "details": dict(self.details),
        }
