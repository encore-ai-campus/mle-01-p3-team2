
"""질문 하나를 입력해 GraphRAG Supervisor 전체 흐름을 실행합니다."""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.graph_retriever import (  # noqa: E402
    GraphRetriever,
    graph_search,
)
from src.retrievers.answer import neo4j_lookup  # noqa: E402
from src.retrievers.vector_retriever import (  # noqa: E402
    VectorIndex,
    vector_search,
)
from src.retrievers.router import JevQuestionRouter  # noqa: E402
from src.supervisor.graph import build_supervisor  # noqa: E402
from src.supervisor.answer_agent import AnswerAgent  # noqa: E402


MODEL_NAME = "BAAI/bge-m3"


def load_config() -> dict[str, str]:
    """환경변수를 로드하고 필수 설정을 검증합니다."""
    load_dotenv(PROJECT_ROOT / ".env")

    required = ("NEO4J_URI", "NEO4J_PASSWORD")
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError(
            f"필수 환경변수가 없습니다: {', '.join(missing)}"
        )

    return {
        "uri": os.environ["NEO4J_URI"],
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.environ["NEO4J_PASSWORD"],
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """동일 프로세스에서는 임베딩 모델을 재사용합니다."""
    return SentenceTransformer(MODEL_NAME)


def build_app():
    """Neo4j·검색기·Supervisor를 연결합니다."""
    config = load_config()

    driver = GraphDatabase.driver(
        config["uri"],
        auth=(config["username"], config["password"]),
    )

    try:
        driver.verify_connectivity()

        graph_retriever = GraphRetriever(driver, database=config["database"])
        vector_retriever = VectorIndex(
            driver,
            get_embedder(),
            database=config["database"],
        )

        def run_graph_search(question: str, **kwargs: Any):
            return graph_search(
                question,
                retriever=graph_retriever,
                **kwargs,
            )

        def run_vector_search(question: str, **kwargs: Any):
            return vector_search(
                question,
                index=vector_retriever,
                **kwargs,
            )

        answer_agent = AnswerAgent.from_env()
        app = build_supervisor(
            graph_search_fn=run_graph_search,
            vector_search_fn=run_vector_search,
            router=JevQuestionRouter.from_env(),
            evidence_lookup=lambda evidence_ids: neo4j_lookup(
                driver,
                evidence_ids,
                database=config["database"],
            ),
            cited_answer_generator=answer_agent.generate_cited,
            citation_support_checker=answer_agent.review_support,
        )

        return app, driver

    except Exception:
        driver.close()
        raise


def print_result(result: dict[str, Any]) -> None:
    """Supervisor의 실행 결과를 안전하게 출력합니다."""

    print("\n[라우팅]")
    print(f"경로: {result.get('route') or '미확인'}")
    print(f"이유: {result.get('route_reason') or '없음'}")

    print("\n[검증]")
    print(f"근거 상태: {result.get('evidence_status') or '미확인'}")
    print(f"재검색 횟수: {result.get('retry_count', 0)}")
    print(f"최종 상태: {result.get('final_status') or '미확인'}")

    for warning in result.get("warnings") or []:
        print(f"경고: {warning}")

    print("\n[답변]")
    print(result.get("final_answer") or "답변이 없습니다.")

    print("\n[검색 근거]")
    results = result.get("results") or []

    if not results:
        print("검색된 근거가 없습니다.")
        return

    for number, item in enumerate(results, start=1):
        print(f"\n{number}. [{item.source}] {item.text}")

        if item.path:
            print("   경로:", " → ".join(map(str, item.path)))

        if item.relation_types:
            print(
                "   관계:",
                " → ".join(map(str, item.relation_types)),
            )

        for edge in item.path_edges or []:
            print(
                f"   hop {edge.get('hop', '?')}: "
                f"{edge.get('relation_type', '미확인')}"
            )

            for chunk in edge.get("original_chunks") or []:
                print(
                    f"      원문: {chunk.get('text', '')} "
                    f"(chunk_id={chunk.get('chunk_id')}, "
                    f"evidence_id={chunk.get('evidence_id')})"
                )

        print("   evidence_ids:", item.evidence_ids or [])
        print("   source_ids:", item.source_ids or [])
        print("   chunk_ids:", item.chunk_ids or [])


def main() -> None:
    """질문을 받아 Supervisor를 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        type=str,
        help="실행할 질문. 생략하면 직접 입력합니다.",
    )
    args = parser.parse_args()

    question = (
        args.question
        if args.question is not None
        else input("질문을 입력하세요: ")
    ).strip()

    if not question:
        parser.error("질문이 비어 있습니다.")

    driver = None

    try:
        app, driver = build_app()
        result = app.invoke({"question": question})
        print_result(result)

    except (ValueError, KeyError) as exc:
        print(f"[설정 오류] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    except Exception as exc:
        print(
            f"[실행 오류] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
