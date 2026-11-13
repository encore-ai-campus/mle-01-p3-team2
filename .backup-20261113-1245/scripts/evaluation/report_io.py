"""평가 리포트를 저장하는 공통 입출력 함수입니다.

라우팅 평가와 multi-hop 평가가 같은 형식(상세 JSONL/CSV + 요약 JSON)으로
결과를 남기도록 이 모듈에 모았습니다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LIST_SEPARATOR = " | "


def write_detail_rows(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    columns: Sequence[str],
    detail_format: str = "jsonl",
    list_separators: Mapping[str, str] | None = None,
) -> Path:
    """문항별 결과를 JSONL 또는 CSV 로 저장합니다.

    CSV 는 목록 값을 문자열로 펼칩니다. `list_separators` 로 열마다 구분자를
    지정할 수 있고, 지정하지 않은 열은 `" | "` 를 씁니다.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if detail_format == "jsonl":
        with out_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path

    if detail_format == "csv":
        separators = dict(list_separators or {})
        # Excel 에서 한글이 깨지지 않도록 BOM 을 붙입니다.
        with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _flatten(row.get(key), separators.get(key, DEFAULT_LIST_SEPARATOR))
                        for key in columns
                    }
                )
        return out_path

    raise ValueError(f"Unknown detail format: {detail_format}")


def _flatten(value: Any, separator: str) -> Any:
    if isinstance(value, (list, tuple)):
        return separator.join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    """요약 지표를 JSON 으로 저장합니다."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path
