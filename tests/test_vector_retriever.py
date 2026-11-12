import unittest

import numpy as np

from vector_retriever import VectorIndex
from search_result import SearchResult


class FakeDriver:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.rows, None, None


class RecordingEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray(self.vectors, dtype=np.float32)


class VectorIndexContractTests(unittest.TestCase):
    def test_vector_index_class_is_available(self):
        try:
            from vector_retriever import VectorIndex
        except ModuleNotFoundError:
            VectorIndex = None

        self.assertIsNotNone(VectorIndex)

    def test_verify_ready_accepts_matching_online_index(self):
        driver = FakeDriver([{
            "name": "chunk_embedding",
            "state": "ONLINE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["embedding"],
            "options": {"indexConfig": {
                "vector.dimensions": 1024,
                "vector.similarity_function": "COSINE",
            }},
        }])
        try:
            index = VectorIndex(driver, object())
        except TypeError as exc:
            self.fail(f"VectorIndex constructor is not implemented: {exc}")

        status = index.verify_ready()

        self.assertEqual(status, {
            "name": "chunk_embedding", "state": "ONLINE",
            "model_name": "BAAI/bge-m3", "dimension": 1024,
            "similarity": "cosine",
        })
        self.assertEqual(driver.calls[0][1], {
            "name": "chunk_embedding", "database_": "neo4j",
        })

    def test_verify_ready_rejects_offline_index(self):
        driver = FakeDriver([{
            "name": "chunk_embedding",
            "state": "POPULATING",
            "labelsOrTypes": ["Chunk"],
            "properties": ["embedding"],
            "options": {"indexConfig": {
                "vector.dimensions": 1024,
                "vector.similarity_function": "COSINE",
            }},
        }])
        index = VectorIndex(driver, object())

        with self.assertRaisesRegex(ValueError, "ONLINE"):
            index.verify_ready()

    def test_verify_ready_rejects_incompatible_index_configuration(self):
        driver = FakeDriver([{
            "name": "chunk_embedding",
            "state": "ONLINE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["embedding"],
            "options": {"indexConfig": {
                "vector.dimensions": 768,
                "vector.similarity_function": "COSINE",
            }},
        }])
        index = VectorIndex(driver, object(), dimension=1024)

        with self.assertRaisesRegex(ValueError, "dimension"):
            index.verify_ready()

    def test_verify_ready_reports_missing_index(self):
        index = VectorIndex(FakeDriver([]), object())

        try:
            index.verify_ready()
        except ValueError as exc:
            self.assertIn("not found", str(exc))
        except Exception as exc:
            self.fail(f"Missing index raised the wrong error: {type(exc).__name__}: {exc}")
        else:
            self.fail("Missing vector index was accepted")

    def test_verify_ready_rejects_wrong_schema_or_similarity(self):
        base = {
            "name": "chunk_embedding", "state": "ONLINE",
            "labelsOrTypes": ["Chunk"], "properties": ["embedding"],
            "options": {"indexConfig": {
                "vector.dimensions": 1024,
                "vector.similarity_function": "COSINE",
            }},
        }
        bad_rows = [
            {**base, "labelsOrTypes": ["Document"]},
            {**base, "options": {"indexConfig": {
                "vector.dimensions": 1024,
                "vector.similarity_function": "EUCLIDEAN",
            }}},
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    VectorIndex(FakeDriver([row]), object()).verify_ready()

    def test_embed_question_normalizes_one_embedding(self):
        embedder = RecordingEmbedder([[3.0, 4.0]])
        index = VectorIndex(FakeDriver([]), embedder, dimension=2)

        self.assertTrue(hasattr(index, "embed_question"), "embed_question is not implemented")
        vector = index.embed_question("마늘 재배 방법")

        np.testing.assert_allclose(vector, [0.6, 0.8])
        self.assertEqual(embedder.calls, [(["마늘 재배 방법"], {})])

    def test_embed_question_rejects_blank_question_before_encoding(self):
        embedder = RecordingEmbedder([[1.0, 0.0]])
        index = VectorIndex(FakeDriver([]), embedder, dimension=2)

        with self.assertRaisesRegex(ValueError, "question"):
            index.embed_question(" \t ")
        self.assertEqual(embedder.calls, [])

    def test_embed_question_rejects_wrong_dimension(self):
        index = VectorIndex(
            FakeDriver([]), RecordingEmbedder([[1.0, 0.0, 0.0]]), dimension=2,
        )

        with self.assertRaisesRegex(ValueError, "dimension"):
            index.embed_question("마늘")

    def test_embed_question_rejects_nonfinite_or_zero_vector(self):
        for vector in ([[0.0, 0.0]], [[np.nan, 1.0]], [[np.inf, 1.0]]):
            with self.subTest(vector=vector):
                index = VectorIndex(
                    FakeDriver([]), RecordingEmbedder(vector), dimension=2,
                )
                with self.assertRaisesRegex(ValueError, "finite, nonzero"):
                    index.embed_question("마늘")

    def test_vector_search_returns_shared_search_result_contract(self):
        driver = FakeDriver([{
            "chunk_id": "CHK_1",
            "source_id": "SRC_1",
            "evidence_id": "EVD_1",
            "text": "마늘은 적합 지역에서 재배한다.",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": 2,
            "source_official": True,
            "score": 0.91,
        }])
        index = VectorIndex(
            driver,
            RecordingEmbedder([[3.0, 4.0]]),
            dimension=2,
        )

        result = index._search_verified("마늘 재배", k=1)[0]

        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.source, "vector")
        self.assertEqual(result.score, 0.91)
        self.assertIsNone(result.path)
        self.assertIsNone(result.relation_types)
        self.assertEqual(result.chunk_ids, ["CHK_1"])


if __name__ == "__main__":
    unittest.main()
