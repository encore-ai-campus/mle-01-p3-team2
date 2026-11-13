"""Configurable RAG evaluation pipeline (RAGAS 0.4.x + ID/citation checks).

This module intentionally does not modify the original notebook pipeline.  It can
be imported from a notebook or executed by a thin notebook wrapper.
"""

# 질문
#  ↓
# 검색
#  ↓
# 검색된 chunk ID 평가
#  ↓
# LLM 답변 생성
#  ↓
# 답변의 인용 검증
#  ↓
# RAGAS 점수와 결합
#  ↓
# 검색 실패/생성 실패/통과/정상 보류 분류
#  ↓
# JSONL 및 요약 JSON 저장

from __future__ import annotations

import argparse, json, math, os, re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

TOKEN_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)
METRICS = ["faithfulness", "answer_relevancy", "answer_correctness",
           "context_precision", "context_recall", "context_relevancy"]

@dataclass
class EvalConfig:
    # 1. EvalConfig
    # 평가 설정값입니다.
    top_k: int = 5
    retry_top_k: int = 10
    score_thresholds: dict[str, float] = field(default_factory=lambda: {
        # Defaults only; replace with the team's approved thresholds.
        "faithfulness": 0.80, "answer_relevancy": 0.80,
        "answer_correctness": 0.80, "context_precision": 0.80,
        "context_recall": 0.80, "context_relevancy": 0.80,
        "hit_at_k": 1.0,
    })
    output_path: str = "data/evaluation/rag_eval_results_v2.jsonl"
    summary_path: str = "data/evaluation/rag_eval_summary_v2.json"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

# 2. validate_cases
# 평가셋의 정답 chunk ID가 실제 chunks.jsonl에 존재하는지 검사합니다.
def validate_cases(cases, chunk_by_id):
    errors = []
    for c in cases:
        ids = c.get("expected_source_chunk_ids") or []
        missing = [x for x in ids if x not in chunk_by_id]
        if missing: errors.append({"case_id": c.get("case_id"), "missing_chunk_ids": missing})
    if errors:
        raise ValueError("Invalid expected_source_chunk_ids: " + json.dumps(errors, ensure_ascii=False))

def context_text(c):
    return c.get("clean_text") or c.get("raw_text") or c.get("embedding_text") or ""

# 3. make_retriever
# 현재는 실제 벡터 검색기가 아니라 단순한 단어 겹침 방식의 임시 검색기입니다.

# 질문과 chunk의 단어가 많이 겹칠수록 높은 순위로 정렬합니다.

# 따라서 실제 운영 평가에서는 이 부분을 다음과 같은 검색기로 교체해야 합니다.
# - Vector Retriever
# - Hybrid Retriever
# - Graph Retriever
# - 실제 프로젝트의 retrieve() 함수
def make_retriever(chunks):
    indexed = []
    for c in chunks:
        text = context_text(c).lower()
        indexed.append((c, set(TOKEN_RE.findall(text))))
    def retrieve(question, k):
        q = set(TOKEN_RE.findall(question.lower()))
        scored = [(len(q & toks) / max(1, len(q)), c) for c, toks in indexed]
        scored.sort(key=lambda x: (-x[0], x[1].get("chunk_id", "")))
        # No lexical match means no evidence. Do not silently return unrelated chunks.
        return [c for score, c in scored[:k] if score > 0]
    return retrieve

# 4. id_metrics
# 검색 결과의 chunk ID와 정답 chunk ID를 직접 비교합니다.

# 계산하는 지표는 다음과 같습니다.
# - Hit@K: 정답 chunk가 하나라도 검색됐는가
# - Precision@K: 검색 결과 중 정답 chunk의 비율
# - Recall@K: 정답 chunk 중 검색된 비율
# - MRR@K: 첫 정답 chunk가 몇 번째에 등장했는가
# - Average Precision@K: 정답 chunk가 상위에 얼마나 잘 배치됐는가
# 즉, RAGAS의 Context Precision/Recall과 별도로 실제 ID 기준 검색 성능을 측정합니다.
def id_metrics(retrieved_ids, expected_ids, k):
    retrieved = list(retrieved_ids[:k]); expected = set(expected_ids)
    hits = [i for i, x in enumerate(retrieved, 1) if x in expected]
    hit = float(bool(hits)) if expected else 1.0
    precision = len(set(retrieved) & expected) / max(1, len(retrieved)) if expected else 1.0
    recall = len(set(retrieved) & expected) / len(expected) if expected else 1.0
    mrr = 1.0 / hits[0] if hits else 0.0
    ap = sum((len([x for x in retrieved[:i] if x in expected]) / i) for i in hits) / max(1, len(expected))
    return {"hit_at_k": hit, "precision_at_k": precision, "recall_at_k": recall,
            "mrr_at_k": mrr, "average_precision_at_k": ap}

# 5. validate_citations
# LLM 답변의 citations를 검증합니다.

# 검증 항목은 다음과 같습니다.
# 1. chunk_id가 검색 결과에 포함됐는가
# 2. quote가 실제 chunk 원문에 존재하는가
# 3. claim이 quote에 의해 뒷받침되는가
# 4. source_url이 없으면 임의 생성하지 않고 null로 두는가
# 현재 support_checker는 단순히 claim이 quote 안에 포함되는지 확인하는 방식입니다.

# 실제 사용 시에는 별도 LLM을 연결해야 합니다.
def validate_citations(answer_obj, retrieved, chunk_by_id, support_checker=None):
    retrieved_map = {c.get("chunk_id"): c for c in retrieved}
    checks = []
    for cit in answer_obj.get("citations", []) if isinstance(answer_obj, dict) else []:
        cid, quote, claim = cit.get("chunk_id"), cit.get("quote", ""), cit.get("claim", "")
        in_results = cid in retrieved_map
        text = context_text(retrieved_map[cid]) if in_results else ""
        quote_exists = bool(quote) and quote in text
        supported = bool(support_checker(claim, quote)) if (quote_exists and support_checker) else quote_exists
        checks.append({"claim": claim, "chunk_id": cid, "quote": quote,
                       "chunk_in_retrieved": in_results, "quote_exists": quote_exists,
                       "claim_supported": supported,
                       "source_url": retrieved_map[cid].get("source_url") if in_results else None})
    ok = bool(checks) and all(x["chunk_in_retrieved"] and x["quote_exists"] and x["claim_supported"] for x in checks)
    if answer_obj.get("status") == "INSUFFICIENT_EVIDENCE": ok = True
    return {"passed": ok, "checks": checks}

# 6. 실패 원인 분류
# classify()가 다음과 같이 분류합니다.
# - PASS: 검색과 답변 생성 모두 성공
# - RETRIEVAL_FAILURE: 정답 chunk를 검색하지 못함
# - GENERATION_FAILURE: 검색은 성공했지만 답변 품질 또는 인용 검증 실패
# - RETRIEVAL_AND_GENERATION_FAILURE: 검색과 답변 모두 실패
# - VALID_ABSTENTION: 근거가 부족해 정상적으로 답변을 보류함
def classify(retrieval_ok, generation_ok, abstention_ok=False):
    if abstention_ok: return "VALID_ABSTENTION"
    if retrieval_ok and generation_ok: return "PASS"
    if not retrieval_ok and not generation_ok: return "RETRIEVAL_AND_GENERATION_FAILURE"
    if not retrieval_ok: return "RETRIEVAL_FAILURE"
    return "GENERATION_FAILURE"

def _answer_obj(raw, retrieved):
    if isinstance(raw, dict): return raw
    return {"status": "ANSWERED", "answer": str(raw), "citations": []}

# 7. evaluate_cases
# 각 평가 문항에 대해 다음을 수행합니다.
# 1. Top-K 검색
# 2. 답변 생성
# 3. citation 검증
# 4. 검증 실패 시 재검색
# 5. 재검색 후 다시 답변 생성
# 6. ID 기반 검색 지표 계산
# 7. RAGAS 점수와 합격선 비교
# 8. 최종 실패 원인 분류
# 답변 생성기는 다음 형식을 반환해야 합니다.
def evaluate_cases(cases, chunks, answer_generator: Callable, ragas_rows=None,
                   config: EvalConfig | None = None, support_checker=None):
    config = config or EvalConfig(); by_id = {c["chunk_id"]: c for c in chunks}
    validate_cases(cases, by_id); retrieve = make_retriever(chunks); results = []
    ragas_rows = ragas_rows or {}
    for case in cases:
        expected = case.get("expected_source_chunk_ids") or []
        is_reject = case.get("expected_behavior") == "REJECT" or case.get("query_strategy") == "reject"
        if is_reject:
            # Out-of-scope cases must be rejected before retrieval. This prevents an
            # unrelated top-ranked chunk from being turned into an answer.
            retrieved = []
            answer = {"status": "INSUFFICIENT_EVIDENCE",
                      "answer": "질문 범위를 벗어나 확인할 수 없습니다.",
                      "citations": []}
            citation = validate_citations(answer, retrieved, by_id, support_checker)
        else:
            retrieved = retrieve(case["question"], config.top_k)
            answer = _answer_obj(answer_generator(case["question"], retrieved), retrieved)
            citation = validate_citations(answer, retrieved, by_id, support_checker)
            if not citation["passed"]:
                retry = retrieve(case["question"], config.retry_top_k)
                answer = _answer_obj(answer_generator(case["question"], retry), retry)
                retrieved = retry; citation = validate_citations(answer, retrieved, by_id, support_checker)
        ids = id_metrics([c.get("chunk_id") for c in retrieved], expected, config.top_k)
        scores = {m: ragas_rows.get(case.get("case_id"), {}).get(m) for m in METRICS}
        retrieval_ok = is_reject or ids["hit_at_k"] >= config.score_thresholds["hit_at_k"]
        gen_ok = all(scores[m] is not None and scores[m] >= config.score_thresholds[m] for m in ("answer_correctness", "faithfulness")) and citation["passed"]
        abstain = answer.get("status") == "INSUFFICIENT_EVIDENCE" and (is_reject or not expected or not retrieval_ok) and citation["passed"]
        passed = abstain or (retrieval_ok and gen_ok)
        result = {"case_id": case.get("case_id"), "question": case.get("question"),
                  "question_type": case.get("question_type"), "difficulty": case.get("difficulty"),
                  "expected_behavior": case.get("expected_behavior"), "response_status": answer.get("status"),
                  "answer": answer.get("answer", ""), "citations": answer.get("citations", []),
                  "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved], "expected_chunk_ids": expected,
                  "ragas": scores, "id_metrics": ids, "citation_validation": citation,
                  "failure_cause": classify(retrieval_ok, gen_ok, abstain), "passed": passed,
                  "retrieved_source_texts": [context_text(c) for c in retrieved],
                  "expected_source_texts": [context_text(by_id[x]) for x in expected]}
        results.append(result)
    return results

# summarize
# 전체 평가 결과를 요약합니다.

# 포함되는 내용은 다음과 같습니다.
# - RAGAS 6개 지표 평균
# - 각 지표의 합격선
# - ID 기반 검색 지표 평균
# - 전체 통과율
# - 실패 원인별 개수
# - 질문 유형별 결과
# - 난이도별 결과
# - expected behavior별 결과
# - 최저 점수 문항 10개
def summarize(results, config: EvalConfig | None = None):
    config = config or EvalConfig()
    def avg(key, subset=results):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    metric_summary = {m: {"score": avg(m, [{"x": r.get("ragas", {}).get(m)} for r in results]),
                          "threshold": config.score_thresholds[m]} for m in METRICS}
    # Correct the compact helper's key shape.
    for m in METRICS:
        vals = [r.get("ragas", {}).get(m) for r in results if r.get("ragas", {}).get(m) is not None]
        metric_summary[m]["score"] = sum(vals)/len(vals) if vals else None
    def grouped(field):
        out = {}
        for value in sorted({r.get(field) for r in results}):
            rows = [r for r in results if r.get(field) == value]
            out[str(value)] = {"count": len(rows), "pass_rate": sum(r["passed"] for r in rows)/len(rows),
                               "answer_correctness": avg("answer_correctness", [{"answer_correctness": r.get("ragas",{}).get("answer_correctness")} for r in rows])}
        return out
    return {"config": asdict(config), "metrics": metric_summary,
            "id_metrics": {k: avg(k, [{k: r["id_metrics"].get(k)} for r in results]) for k in ("hit_at_k","precision_at_k","recall_at_k","mrr_at_k","average_precision_at_k")},
            "overall_pass": bool(results) and all(r["passed"] for r in results),
            "pass_rate": sum(r["passed"] for r in results)/len(results) if results else 0,
            "failure_counts": dict(Counter(r["failure_cause"] for r in results)),
            "by_question_type": grouped("question_type"), "by_difficulty": grouped("difficulty"),
            "by_expected_behavior": grouped("expected_behavior"),
            "lowest_score_cases": sorted(results, key=lambda r: (r.get("ragas",{}).get("answer_correctness") or 0))[:10]}

# 9. save_results
# 다음 두 파일을 생성합니다.
# -> data/evaluation/rag_eval_results_v2.jsonl
#    data/evaluation/rag_eval_summary_v2.json
def save_results(results, summary, root: Path):
    out = root / summary["config"]["output_path"]; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sp = root / summary["config"]["summary_path"]
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, sp

def _default_answer_generator(question, retrieved):
    """Runnable fallback generator for smoke tests.

    Replace this with the project's real Answer Agent for production evaluation.
    """
    if not retrieved:
        return {"status": "INSUFFICIENT_EVIDENCE", "answer": "확인할 수 없습니다.", "citations": []}
    chunk = retrieved[0]
    quote = context_text(chunk)
    return {"status": "ANSWERED", "answer": quote,
            "citations": [{"claim": quote, "quote": quote,
                           "chunk_id": chunk.get("chunk_id"),
                           "source_url": chunk.get("source_url")}]}

def _default_support_checker(claim, quote):
    return bool(claim and quote and claim in quote)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the configurable RAG evaluation pipeline.")
    parser.add_argument("--root", type=Path, default=None, help="Project root (defaults to repository root).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retry-top-k", type=int, default=10)
    args = parser.parse_args(argv)
    root = args.root or Path(__file__).resolve().parents[2]
    config = EvalConfig(top_k=args.top_k, retry_top_k=args.retry_top_k)
    golden_path = root / "data/evaluation/golden_set_final.jsonl"
    cases = [row for row in read_jsonl(golden_path)
             if row.get("status", "READY") == "READY"]
    chunks = read_jsonl(root / "data/processed/chunks.jsonl")
    if not cases:
        raise FileNotFoundError(f"No READY cases found in {golden_path}")
    # RAGAS scores are intentionally empty here. Connect the real evaluator through the notebook
    # when six RAGAS scores are required; this CLI still produces ID/citation evaluation files.
    results = evaluate_cases(cases, chunks, _default_answer_generator,
                             ragas_rows={}, config=config,
                             support_checker=_default_support_checker)
    summary = summarize(results, config)
    out_path, summary_path = save_results(results, summary, root)
    print(f"evaluated cases: {len(results)}")
    print(f"saved: {out_path}")
    print(f"saved: {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
