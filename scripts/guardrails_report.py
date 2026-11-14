from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path("data/evaluation/guardrails/results")
REPORT_DIR = Path("data/evaluation/guardrails/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def safe_div(a, b):
    return a / b if b else 0.0


def calc_metrics(rows):
    total = len(rows)
    correct_count = sum(1 for row in rows if to_bool(row.get("correct", False)))
    wrong_count = total - correct_count

    tp = tn = fp = fn = 0

    for row in rows:
        gold = to_int(row.get("gold", 0))
        pred = to_int(row.get("predicted", 0))

        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    accuracy = safe_div(tp + tn, total)

    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "accuracy": round(accuracy * 100, 2),
    }


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


csv_files = sorted(RESULTS_DIR.glob("*.csv"))

summary_rows = []
error_rows = []

for csv_file in csv_files:
    rows = read_csv_rows(csv_file)
    metrics = calc_metrics(rows)

    summary_rows.append(
        {
            "file": csv_file.name,
            **metrics,
        }
    )

    for row in rows:
        if not to_bool(row.get("correct", False)):
            error_rows.append(
                {
                    "source_file": csv_file.name,
                    "number": row.get("number", ""),
                    "gold": row.get("gold", ""),
                    "predicted": row.get("predicted", ""),
                    "question": row.get("question", ""),
                    "reason": row.get("reason", ""),
                }
            )

summary_csv_path = REPORT_DIR / "summary_metrics.csv"
with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "total",
            "correct",
            "wrong",
            "tp",
            "tn",
            "fp",
            "fn",
            "precision",
            "recall",
            "fpr",
            "fnr",
            "accuracy",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

error_csv_path = REPORT_DIR / "error_cases.csv"
with error_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_file",
            "number",
            "gold",
            "predicted",
            "question",
            "reason",
        ],
    )
    writer.writeheader()
    writer.writerows(error_rows)

md_path = REPORT_DIR / "summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# Guardrails Evaluation Summary\n\n")
    f.write("## 1. 전체 요약\n\n")
    f.write(
        "| file | total | correct | wrong | precision | recall | fpr | fnr | accuracy |\n"
    )
    f.write(
        "|------|------:|--------:|------:|----------:|-------:|----:|----:|---------:|\n"
    )

    for row in summary_rows:
        f.write(
            f"| {row['file']} | {row['total']} | {row['correct']} | {row['wrong']} | "
            f"{row['precision']:.2f}% | {row['recall']:.2f}% | {row['fpr']:.2f}% | "
            f"{row['fnr']:.2f}% | {row['accuracy']:.2f}% |\n"
        )

    f.write("\n## 2. 오답 사례 수\n\n")
    f.write(f"- 총 오답 수: **{len(error_rows)}개**\n\n")

    if error_rows:
        f.write("## 3. 주요 오답 사례\n\n")
        f.write("| source_file | number | gold | predicted | question | reason |\n")
        f.write("|------------|-------:|-----:|----------:|----------|--------|\n")
        for row in error_rows:
            question = str(row["question"]).replace("\n", " ").replace("|", "/")
            reason = str(row["reason"]).replace("\n", " ").replace("|", "/")
            f.write(
                f"| {row['source_file']} | {row['number']} | {row['gold']} | "
                f"{row['predicted']} | {question} | {reason} |\n"
            )
    else:
        f.write("## 3. 주요 오답 사례\n\n")
        f.write("- 오답 사례가 없습니다.\n")

print("생성 완료")
print(f"- {summary_csv_path}")
print(f"- {error_csv_path}")
print(f"- {md_path}")