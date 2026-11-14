"""병해충 도감 화면이 사용하는 병해·해충 데이터 집계 모듈입니다.

Disease / Insect 노드는 작물별로 따로 들어 있어서 같은 이름이 여러 번
나옵니다(예: 노균병). 그래서 항목마다 어느 작물 것인지 함께 보여 줍니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.streamlit.crop_data import Entity, Graph


ANY = "전체"

DISEASE = "병해"
INSECT = "해충"
KINDS = (ANY, DISEASE, INSECT)

KIND_EMOJI = {DISEASE: "🦠", INSECT: "🐛"}

# (속성 이름, 화면 제목) — 병해와 해충이 쓰는 필드가 다릅니다.
DISEASE_SECTIONS = (
    ("symptoms", "증상"),
    ("development_condition", "발생 조건"),
    ("prevention_method", "예방 방법"),
)
INSECT_SECTIONS = (
    ("damage", "피해 양상"),
    ("ecology", "생태"),
    ("prevention", "방제"),
)


@dataclass(frozen=True)
class PestEntry:
    """병해 또는 해충 한 건입니다."""

    name: str
    kind: str
    crops: tuple[str, ...]
    aliases: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    pesticide_count: int

    @property
    def emoji(self) -> str:
        return KIND_EMOJI.get(self.kind, "🔍")

    @property
    def crop_label(self) -> str:
        return " · ".join(self.crops)

    @property
    def search_blob(self) -> str:
        """이름과 별칭을 합친 검색용 문자열입니다."""

        return " ".join((self.name, *self.aliases)).lower()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _aliases(attributes: dict[str, Any]) -> tuple[str, ...]:
    raw = attributes.get("aliases")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(_text(item) for item in raw if _text(item))


def build_entry(graph: Graph, entity: Entity) -> PestEntry:
    """Disease 또는 Insect 노드 하나를 도감 항목으로 바꿉니다."""

    kind = DISEASE if entity.entity_type == "Disease" else INSECT
    fields = DISEASE_SECTIONS if kind == DISEASE else INSECT_SECTIONS

    sections = tuple(
        (label, _text(entity.attributes.get(key)))
        for key, label in fields
        if _text(entity.attributes.get(key))
    )

    # 작물은 Crop -> 병해충 관계에서 가져오고, 없으면 속성을 씁니다.
    crops = tuple(
        sorted({crop.name for crop in graph.sources_of_type(entity.entity_id, "Crop")})
    )
    if not crops and _text(entity.attributes.get("crop")):
        crops = (_text(entity.attributes["crop"]),)

    # 이 병해충을 대상으로 등록된 약제 수입니다(PesticideUse -> 병해충).
    pesticide_count = len(
        graph.sources_of_type(entity.entity_id, "PesticideUse")
    )

    return PestEntry(
        name=entity.name,
        kind=kind,
        crops=crops,
        aliases=_aliases(entity.attributes),
        sections=sections,
        pesticide_count=pesticide_count,
    )


def build_entries(graph: Graph) -> list[PestEntry]:
    """그래프의 모든 병해·해충을 도감 항목으로 만듭니다."""

    entries = [
        build_entry(graph, entity)
        for entity in graph.entities.values()
        if entity.entity_type in ("Disease", "Insect")
    ]
    entries.sort(key=lambda e: (e.kind, e.name, e.crop_label))
    return entries


def crop_values(entries: Iterable[PestEntry]) -> list[str]:
    """작물 필터 선택지를 돌려줍니다."""

    values: set[str] = set()
    for entry in entries:
        values.update(entry.crops)
    values.discard("")
    return [ANY, *sorted(values)]


def filter_entries(
    entries: Iterable[PestEntry],
    *,
    crop: str = ANY,
    kind: str = ANY,
    keyword: str = "",
    only_with_pesticide: bool = False,
) -> list[PestEntry]:
    """선택한 조건에 맞는 항목만 골라 돌려줍니다."""

    needle = keyword.strip().lower()
    result = []
    for entry in entries:
        if crop != ANY and crop not in entry.crops:
            continue
        if kind != ANY and entry.kind != kind:
            continue
        if only_with_pesticide and entry.pesticide_count == 0:
            continue
        if needle and needle not in entry.search_blob:
            continue
        result.append(entry)
    return result


@dataclass
class PestCounts:
    """요약 숫자 칩에 쓰는 집계입니다."""

    total: int = 0
    diseases: int = 0
    insects: int = 0
    with_pesticide: int = 0
    extra: dict[str, int] = field(default_factory=dict)


def count_entries(entries: Iterable[PestEntry]) -> PestCounts:
    """항목 묶음의 종류별 건수를 셉니다."""

    entries = list(entries)
    return PestCounts(
        total=len(entries),
        diseases=sum(1 for e in entries if e.kind == DISEASE),
        insects=sum(1 for e in entries if e.kind == INSECT),
        with_pesticide=sum(1 for e in entries if e.pesticide_count > 0),
    )
