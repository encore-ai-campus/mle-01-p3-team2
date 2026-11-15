
"""검색 결과에서 농약 식별정보를 구조적으로 추출한다.

Execution Rail 진입 전에 작물·병해충·제품명을 상태에 채워
"제품명·작물·병해충 정보가 부족합니다" 오탐을 막는다.

질문에 제품명이 없으면 임의로 하나를 고르지 않고
후보 목록만 전달한다.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


CROP_TYPES = ("Crop",)
PEST_TYPES = ("Insect", "Disease", "Pest", "Weed")
PRODUCT_TYPES = ("PesticideUse", "Pesticide")


@dataclass
class PesticideIdentity:
    """검색 결과에서 추출한 농약 식별정보."""

    crop: str = ""
    target_pest: str = ""
    product_name: str = ""
    brand: str = ""
    product_candidates: list[str] = field(default_factory=list)

    @property
    def has_scope(self) -> bool:
        """작물 + 병해충이 모두 확인됐는지."""
        return bool(self.crop and self.target_pest)

    @property
    def is_single_product(self) -> bool:
        """질문이 특정 제품을 지목했는지."""
        return bool(self.product_name)


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).casefold()


def _path_nodes(result: Any) -> list[Mapping[str, Any]]:
    """SearchResult 또는 dict에서 경로 노드를 꺼낸다."""

    nodes = (
        result.get("path_nodes")
        if isinstance(result, Mapping)
        else getattr(result, "path_nodes", None)
    )

    if not isinstance(nodes, Sequence):
        return []

    return [node for node in nodes if isinstance(node, Mapping)]


def _collect(
    results: Iterable[Any],
    entity_types: tuple[str, ...],
) -> list[str]:
    """지정한 엔티티 타입의 노드 이름을 등장 순서대로 모은다."""

    names: list[str] = []
    seen: set[str] = set()

    for result in results:
        for node in _path_nodes(result):
            entity_type = str(node.get("entity_type") or "")

            if entity_type not in entity_types:
                continue

            name = str(node.get("name") or "").strip()
            key = _normalize(name)

            if not key or key in seen:
                continue

            seen.add(key)
            names.append(name)

    return names


def _mentioned(
    names: list[str],
    question: str,
) -> list[str]:
    """질문 본문에 직접 등장한 이름만 남긴다."""

    text = _normalize(question)

    if not text:
        return []

    return [
        name
        for name in names
        if _normalize(name) and _normalize(name) in text
    ]


def _pick(
    names: list[str],
    question: str,
) -> str:
    """질문에 등장한 이름을 우선하고, 없으면 유일할 때만 선택한다."""

    mentioned = _mentioned(names, question)

    if len(mentioned) == 1:
        return mentioned[0]

    if mentioned:
        # 질문에 여러 값이 등장하면 가장 긴(구체적인) 이름을 쓴다.
        return max(mentioned, key=lambda name: len(_normalize(name)))

    if len(names) == 1:
        return names[0]

    return ""


def extract_pesticide_identity(
    question: str,
    results: Iterable[Any],
    *,
    crop_hint: str | None = None,
) -> PesticideIdentity:
    """검색된 경로에서 작물·병해충·제품 후보를 추출한다."""

    results = list(results or [])
    question = question if isinstance(question, str) else ""

    crops = _collect(results, CROP_TYPES)
    pests = _collect(results, PEST_TYPES)
    products = _collect(results, PRODUCT_TYPES)

    crop = _pick(crops, question)

    if not crop and isinstance(crop_hint, str):
        crop = crop_hint.strip()

    target_pest = _pick(pests, question)

    # 제품명은 질문이 직접 지목했을 때만 확정한다.
    mentioned_products = _mentioned(products, question)
    product_name = (
        max(
            mentioned_products,
            key=lambda name: len(_normalize(name)),
        )
        if mentioned_products
        else ""
    )

    return PesticideIdentity(
        crop=crop,
        target_pest=target_pest,
        product_name=product_name,
        # 질문에 등장한 이름이 품목명인지 상표명인지는
        # DB 조회에서 두 필드를 모두 비교해 판단한다.
        brand="",
        product_candidates=products,
    )


def identity_state_update(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Supervisor 상태에 농약 식별정보를 채워 넣는다."""

    identity = extract_pesticide_identity(
        state.get("question", ""),
        state.get("results", []),
        crop_hint=state.get("crop"),
    )

    update: dict[str, Any] = {
        "product_candidates": identity.product_candidates,
    }

    # 이미 확정된 값은 덮어쓰지 않는다.
    for key, value in (
        ("crop", identity.crop),
        ("target_pest", identity.target_pest),
        ("product_name", identity.product_name),
    ):
        existing = state.get(key)

        if isinstance(existing, str) and existing.strip():
            continue

        if value:
            update[key] = value

    return update
