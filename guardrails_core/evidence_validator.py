"""Validation of normalized graph/vector retrieval evidence."""

from collections.abc import Mapping
import re
from typing import Any

from .contracts import GuardrailRequest, ValidationResult


_VALID_SOURCES = frozenset({"graph", "vector"})
_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9-]*_[A-Za-z0-9-]+$")
_ID_FIELDS = ("evidence_ids", "source_ids", "chunk_ids")


def _as_mapping(item: Any) -> dict[str, Any] | None:
    if isinstance(item, Mapping):
        return dict(item)

    field_names = ("source", "text", "evidence_ids", "source_ids", "chunk_ids", "metadata")
    if all(hasattr(item, name) for name in field_names):
        return {name: getattr(item, name) for name in field_names}
    return None


def _validate_item(item: Any) -> list[str]:
    normalized = _as_mapping(item)
    if normalized is None:
        return ["evidence item must be a mapping or SearchResult"]

    errors: list[str] = []
    source = normalized.get("source")
    if not isinstance(source, str) or source not in _VALID_SOURCES:
        errors.append("source must be graph or vector")

    text = normalized.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append("text must be a non-empty string")

    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping")

    all_ids: list[str] = []
    for field_name in _ID_FIELDS:
        values = normalized.get(field_name, [])
        if values is None:
            values = []
        if not isinstance(values, (list, tuple)):
            errors.append(f"{field_name} must be a list")
            continue
        for value in values:
            if not isinstance(value, str):
                errors.append(f"{field_name} contains non-string id")
                continue
            if not _ID_PATTERN.fullmatch(value):
                errors.append(f"{field_name} contains malformed id")
            all_ids.append(value)

    if not all_ids:
        errors.append("at least one evidence, source, or chunk id is required")
    if len(all_ids) != len(set(all_ids)):
        errors.append("evidence ids must be unique")
    return errors


def validate_evidence(request: GuardrailRequest) -> ValidationResult:
    """Block empty or structurally unsafe retrieval evidence."""

    evidence = request.evidence
    if not evidence:
        return ValidationResult(
            allowed=False,
            decision="BLOCK",
            reason="검색 근거 없음",
            fallback_text="검색 근거를 확인하지 못해 답변을 제공할 수 없습니다.",
            details={"checked_items": 0, "invalid_items": ["evidence is empty"]},
        )
    if not isinstance(evidence, (list, tuple)):
        return ValidationResult(
            allowed=False,
            decision="BLOCK",
            reason="검색 근거 형식 검증 실패",
            fallback_text="검색 근거 목록 형식을 확인한 뒤 다시 시도해 주세요.",
            details={"checked_items": 0, "invalid_items": ["evidence must be a list"]},
        )

    invalid_items: dict[int, list[str]] = {}
    for index, item in enumerate(evidence):
        errors = _validate_item(item)
        if errors:
            invalid_items[index] = errors

    if invalid_items:
        return ValidationResult(
            allowed=False,
            decision="BLOCK",
            reason="검색 근거 필드 또는 ID 검증 실패",
            fallback_text="검색 근거의 식별자나 필수 필드를 확인한 뒤 다시 시도해 주세요.",
            details={"checked_items": len(evidence), "invalid_items": invalid_items},
        )

    return ValidationResult(
        allowed=True,
        decision="ALLOW",
        reason="검색 근거 구조 확인",
        details={"checked_items": len(evidence), "invalid_items": {}},
    )
