"""농약 검색 화면이 사용하는 등록 농약 데이터 집계 모듈입니다.

이 데이터는 모두 `recommendation_eligible=False`이고 검증 상태가
PRELIMINARY_PASS / REVIEW_REQUIRED 입니다. 즉 프로젝트 가드레일 기준으로
'확인되지 않은 등록 정보'이므로 화면에서는 추천이 아니라 조회 결과로만
다루고, 실제 사용 전에는 농약안전정보시스템 확인을 안내해야 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.streamlit.crop_data import Graph


PSIS_SEARCH_URL = "https://psis.rda.go.kr/psis/agc/agchmSearch.ps"

# 가드레일이 '확인됨'으로 보지 않는 검증 상태입니다.
UNCONFIRMED_STATUSES = frozenset({"PRELIMINARY_PASS", "REVIEW_REQUIRED"})

TOXICITY_ORDER = ("저독성(Ⅳ)", "보통독성(Ⅲ)", "고독성(Ⅱ)", "맹독성(Ⅰ)")

ANY = "전체"


@dataclass(frozen=True)
class PesticideRecord:
    """등록 농약 사용 정보 한 건입니다."""

    crop: str
    target: str
    brand: str
    product: str
    company: str
    purpose: str
    formulation: str
    dilution: str
    quantity: str
    application_timing: str
    safety_timing: str
    safety_count: str
    human_toxicity: str
    eco_toxicity: str
    registration_no: str
    valid_until: str
    validation_status: str
    cancellation_date: str
    official_url: str
    needs_review: bool

    @property
    def search_blob(self) -> str:
        """자유 검색어와 맞춰 볼 문자열입니다."""

        return " ".join(
            (self.brand, self.product, self.company, self.target)
        ).lower()


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("-", "None") else text


def _format_valid_until(value: str) -> str:
    """'20311206' 형태를 '2031-12-06'으로 바꿉니다."""

    digits = _text(value)
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return digits


def build_record(crop: str, attributes: dict[str, Any]) -> PesticideRecord:
    """PesticideUse 속성에서 화면용 레코드를 만듭니다."""

    cancellation = _text(attributes.get("cancellation_dates"))
    audit = _text(attributes.get("audit_disposition"))

    return PesticideRecord(
        crop=crop or _text(attributes.get("canonical_crop")),
        target=_text(attributes.get("target_name")),
        brand=_text(attributes.get("brand")),
        product=_text(attributes.get("product")),
        company=_text(attributes.get("company")),
        purpose=_text(attributes.get("purpose")),
        formulation=_text(attributes.get("formulation")),
        dilution=_text(attributes.get("dilution")),
        quantity=_text(attributes.get("use_quantity")),
        application_timing=_text(attributes.get("application_method")),
        safety_timing=_text(attributes.get("use_timing")),
        safety_count=_text(attributes.get("use_count")),
        human_toxicity=_text(attributes.get("human_livestock_toxicity")),
        eco_toxicity=_text(attributes.get("ecological_toxicity")),
        registration_no=_text(attributes.get("registration_no")),
        valid_until=_format_valid_until(attributes.get("registration_valid_until")),
        validation_status=_text(attributes.get("validation_status")),
        cancellation_date=cancellation,
        official_url=_text(attributes.get("official_query_url")),
        # 취소 이력이 있거나 추천 보류로 표시된 건은 따로 경고합니다.
        needs_review=bool(cancellation) or audit == "HOLD_RECOMMENDATION",
    )


def build_records(graph: Graph) -> list[PesticideRecord]:
    """그래프의 Crop -> PesticideUse 관계를 따라 전체 레코드를 만듭니다."""

    records: list[PesticideRecord] = []
    for entity in graph.entities.values():
        if entity.entity_type != "Crop":
            continue
        for use in graph.neighbours_of_type(entity.entity_id, "PesticideUse"):
            records.append(build_record(entity.name, use.attributes))

    records.sort(key=lambda r: (r.crop, r.target, r.brand))
    return records


def facet_values(records: Iterable[PesticideRecord], attribute: str) -> list[str]:
    """필터 선택지로 쓸 값 목록을 중복 없이 돌려줍니다."""

    values = {getattr(record, attribute) for record in records}
    values.discard("")
    return [ANY, *sorted(values)]


def toxicity_values(records: Iterable[PesticideRecord]) -> list[str]:
    """독성 등급은 사전순이 아니라 위험도 순으로 정렬합니다."""

    present = {record.human_toxicity for record in records}
    present.discard("")
    ordered = [value for value in TOXICITY_ORDER if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return [ANY, *ordered]


def filter_records(
    records: Iterable[PesticideRecord],
    *,
    crop: str = ANY,
    target: str = ANY,
    purpose: str = ANY,
    formulation: str = ANY,
    toxicity: str = ANY,
    keyword: str = "",
    hide_needs_review: bool = False,
) -> list[PesticideRecord]:
    """선택한 조건에 맞는 레코드만 골라 돌려줍니다."""

    needle = keyword.strip().lower()
    result = []
    for record in records:
        if crop != ANY and record.crop != crop:
            continue
        if target != ANY and record.target != target:
            continue
        if purpose != ANY and record.purpose != purpose:
            continue
        if formulation != ANY and record.formulation != formulation:
            continue
        if toxicity != ANY and record.human_toxicity != toxicity:
            continue
        if hide_needs_review and record.needs_review:
            continue
        if needle and needle not in record.search_blob:
            continue
        result.append(record)
    return result


def targets_for_crop(records: Iterable[PesticideRecord], crop: str) -> list[str]:
    """선택한 작물에 등록된 병해충 이름만 돌려줍니다."""

    values = {
        record.target
        for record in records
        if record.target and (crop == ANY or record.crop == crop)
    }
    return [ANY, *sorted(values)]
