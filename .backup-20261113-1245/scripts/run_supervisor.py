"""질문 하나를 입력해 GraphRAG Supervisor 전체 흐름을 실행합니다."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrievers.graph_retriever import GraphRetriever, graph_search  # noqa: E402
from retrievers.vector_retriever import VectorIndex  # noqa: E402
from supervisor.graph import build_supervisor  # noqa: E402


def build_app():
    """Neo4j·임베딩 모델·Supervisor를 연결한 실행 앱을 만듭니다."""
    load_dotenv(PROJECT_ROOT / ".env")
    uri = os.environ["NEO4J_URI"]
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    graph_retriever = GraphRetriever(driver)
    embedder = SentenceTransformer("BAAI/bge-m3")
    vector_retriever = VectorIndex(driver, embedder)

    def run_graph_search(question, **kwargs):
        return graph_search(question, retriever=graph_retriever, **kwargs)

    def run_vector_search(question, **kwargs):
        return vector_retriever.search(question, **kwargs)

    app = build_supervisor(
        graph_search_fn=run_graph_search,
        vector_search_fn=run_vector_search,
    )
    return app, driver


def print_result(result: dict) -> None:
    """Supervisor 실행 결과에서 라우팅·근거·답변 정보를 출력합니다."""
    print("\n[라우팅]")
    print(f"경로: {result.get('route')}")
    print(f"이유: {result.get('route_reason')}")

    print("\n[검증]")
    print(f"근거 상태: {result.get('evidence_status')}")
    print(f"재검색 횟수: {result.get('retry_count', 0)}")
    print(f"최종 상태: {result.get('final_status')}")

    print("\n[답변]")
    print(result.get("final_answer", "답변이 없습니다."))

    print("\n[검색 근거]")
    for number, item in enumerate(result.get("results", []), 1):
        print(f"\n{number}. [{item.source}] {item.text}")
        if item.path:
            print("   경로:", " → ".join(item.path))
        if item.relation_types:
            print("   관계:", " → ".join(item.relation_types))
        print("   evidence_ids:", item.evidence_ids)
        print("   source_ids:", item.source_ids)
        print("   chunk_ids:", item.chunk_ids)


def main() -> None:
    """질문을 받아 Supervisor를 한 번 실행합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="실행할 질문. 생략하면 대화형으로 입력합니다.")
    args = parser.parse_args()
    question = args.question or input("질문을 입력하세요: ").strip()
    if not question:
        raise SystemExit("질문이 비어 있습니다.")

    app, driver = build_app()
    try:
        result = app.invoke({"question": question})
        print_result(result)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
