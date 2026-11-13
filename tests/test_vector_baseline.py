import json
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.indexing.build_vector_index import (
    DEFAULT_MODEL,
    VectorIndex,
    build_index,
    build_neo4j_index,
    evaluate,
    iter_embedding_batches,
    load_chunks,
    load_source_official,
    main,
    store_chunk_embeddings,
    store_chunk_metadata,
)


class FakeEmbedder:
    def __init__(self):
        self.vectors = {
            "apple": [1.0, 0.0],
            "banana": [0.0, 1.0],
            "apple question": [1.0, 0.0],
            "banana question": [0.0, 1.0],
        }

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.vectors["pear"] = [2.0, 0.0]

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return super().encode(texts, **kwargs)


class VectorBaselineTests(unittest.TestCase):
    def test_embedding_batches_keep_order_and_normalize(self):
        rows = [
            {"chunk_id": "A", "embedding_text": "apple"},
            {"chunk_id": "B", "embedding_text": "banana"},
            {"chunk_id": "C", "embedding_text": "pear"},
        ]
        embedder = RecordingEmbedder()
        batches = list(iter_embedding_batches(rows, embedder, batch_size=2))
        self.assertEqual(embedder.calls, [["apple", "banana"], ["pear"]])
        self.assertEqual([[row["chunk_id"] for row in chunk_rows] for chunk_rows, _ in batches], [["A", "B"], ["C"]])
        np.testing.assert_allclose(batches[1][1], [[1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            list(iter_embedding_batches(rows, embedder, batch_size=0))

    def test_embedding_batches_reject_dimension_change(self):
        class ChangingEmbedder:
            def encode(self, texts, **kwargs):
                width = 2 if texts[0] == "apple" else 3
                return np.ones((len(texts), width), dtype=np.float32)

        rows = [{"chunk_id": "A", "embedding_text": "apple"},
                {"chunk_id": "B", "embedding_text": "banana"}]
        with self.assertRaisesRegex(ValueError, "dimension"):
            list(iter_embedding_batches(rows, ChangingEmbedder(), batch_size=1))

    def test_default_embedding_model_is_bge_m3(self):
        self.assertEqual(DEFAULT_MODEL, "BAAI/bge-m3")

    def test_load_chunks_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            path.write_text(
                '{"chunk_id":"A","embedding_text":"apple"}\n\n'
                '{"chunk_id":"B","embedding_text":"banana"}\n',
                encoding="utf-8",
            )
            self.assertEqual([row["chunk_id"] for row in load_chunks(path)], ["A", "B"])
            path.write_text('{"chunk_id":"A"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":2"):
                load_chunks(path)
            path.write_text('[]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object"):
                load_chunks(path)

    def test_load_chunks_requires_nonempty_string_fields(self):
        invalid_rows = (
            ({"embedding_text": "apple"}, "chunk_id"),
            ({"chunk_id": "A"}, "embedding_text"),
            ({"chunk_id": "  ", "embedding_text": "apple"}, "chunk_id"),
            ({"chunk_id": 123, "embedding_text": "apple"}, "chunk_id"),
            ({"chunk_id": "A", "embedding_text": " \t "}, "embedding_text"),
            ({"chunk_id": "A", "embedding_text": None}, "embedding_text"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            for row, field in invalid_rows:
                with self.subTest(row=row):
                    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        load_chunks(path)

    def test_load_chunks_rejects_empty_data_and_duplicate_chunk_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            for content in ("", " \n\t\n"):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "No chunks"):
                        load_chunks(path)
            path.write_text(
                '{"chunk_id":"A","embedding_text":"apple"}\n'
                '{"chunk_id":"A","embedding_text":"banana"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate chunk_id.*A"):
                load_chunks(path)

    def make_index(self, root):
        chunks = root / "chunks.jsonl"
        chunks.write_text("\n".join(json.dumps(row) for row in [
            {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1"},
            {"chunk_id": "B", "embedding_text": "banana", "source_id": "S2"},
        ]) + "\n", encoding="utf-8")
        build_index(chunks, root / "index", FakeEmbedder(), "fake")
        return chunks, root / "index"

    def test_build_search_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                "\n".join(json.dumps(row) for row in [
                    {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1"},
                    {"chunk_id": "B", "embedding_text": "banana", "source_id": "S2"},
                ]) + "\n", encoding="utf-8"
            )
            build_index(chunks, root / "index", FakeEmbedder(), "fake", batch_size=1)
            index = VectorIndex.load(root / "index")
            hits = index.search("apple question", FakeEmbedder(), k=2)
            self.assertEqual([hit["chunk_id"] for hit in hits], ["A", "B"])
            self.assertEqual(hits[0]["source_id"], "S1")
            self.assertAlmostEqual(hits[0]["score"], 1.0)

    def test_evaluation_metrics_and_strategy_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                '\n'.join(json.dumps(row) for row in [
                    {"chunk_id": "A", "embedding_text": "apple"},
                    {"chunk_id": "B", "embedding_text": "banana"},
                ]) + '\n', encoding="utf-8"
            )
            build_index(chunks, root / "index", FakeEmbedder(), "fake")
            golden = root / "golden.jsonl"
            golden.write_text('\n'.join(json.dumps(row) for row in [
                {"case_id": "G1", "question": "apple question", "status": "READY", "query_strategy": "vector", "expected_source_chunk_ids": ["A", "B"]},
                {"case_id": "G2", "question": "banana question", "status": "READY", "query_strategy": "graph", "expected_source_chunk_ids": ["A"]},
                {"case_id": "G3", "question": "apple question", "status": "DRAFT", "query_strategy": "vector", "expected_source_chunk_ids": ["A"]},
            ]) + '\n', encoding="utf-8")
            report = evaluate(VectorIndex.load(root / "index"), golden, FakeEmbedder(), k=1, strategy="vector")
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["hit_at_k"], 1.0)
            self.assertEqual(report["recall_at_k"], 0.5)
            self.assertEqual(report["mrr_at_k"], 1.0)

    def test_no_relevant_result_has_zero_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, index_dir = self.make_index(root)
            golden = root / "golden.jsonl"
            golden.write_text(json.dumps({
                "case_id": "G1", "question": "apple question", "status": "READY",
                "query_strategy": "vector", "expected_source_chunk_ids": ["B"],
            }) + "\n", encoding="utf-8")
            report = evaluate(VectorIndex.load(index_dir), golden, FakeEmbedder(), k=1)
            self.assertEqual(report["cases"][0]["retrieved_chunk_ids"], ["A"])
            for metric in ("hit_at_k", "recall_at_k", "mrr_at_k"):
                self.assertEqual(report[metric], 0.0)

    def test_empty_question_and_invalid_search_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, index_dir = self.make_index(root)
            index = VectorIndex.load(index_dir)
            for question, k in (("", 1), (" \t ", 1), ("apple question", 0)):
                with self.subTest(question=question, k=k), self.assertRaises(ValueError):
                    index.search(question, FakeEmbedder(), k=k)
            with self.assertRaisesRegex(ValueError, "dimension"):
                index.search("apple question", WrongDimensionEmbedder(), k=1)

    def test_invalid_build_and_empty_gold_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, index_dir = self.make_index(root)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                build_index(chunks, root / "other", FakeEmbedder(), "fake", batch_size=0)
            chunks.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No chunks"):
                build_index(chunks, root / "other", FakeEmbedder(), "fake")
            golden = root / "golden.jsonl"
            golden.write_text(json.dumps({
                "case_id": "G1", "question": "apple question", "status": "READY",
                "query_strategy": "vector", "expected_source_chunk_ids": [],
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no expected source chunks"):
                evaluate(VectorIndex.load(index_dir), golden, FakeEmbedder())

    def test_index_metadata_and_manifest_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, index_dir = self.make_index(root)
            metadata = index_dir / "metadata.jsonl"
            rows = metadata.read_text(encoding="utf-8").splitlines()
            metadata.write_text(rows[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row counts"):
                VectorIndex.load(index_dir)
            metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
            metadata.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                VectorIndex.load(index_dir)
            metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
            manifest_path = index_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dimension"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dimension"):
                VectorIndex.load(index_dir)
            manifest["dimension"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            vectors_path = index_dir / "vectors.npy"
            vectors = np.load(vectors_path)
            vectors[0, 0] = 0.5
            np.save(vectors_path, vectors)
            with self.assertRaisesRegex(ValueError, "integrity"):
                VectorIndex.load(index_dir)


class WrongDimensionEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 3), dtype=np.float32)


class RecordingNeo4jDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))

    def verify_connectivity(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Neo4jIndexTests(unittest.TestCase):
    def test_source_official_metadata_uses_registry_mapping(self):
        rows = [{"chunk_id": "A", "embedding_text": "apple", "source_id": "S1",
                 "evidence_id": "E1", "raw_text": "Original A"}]
        driver = RecordingNeo4jDriver()
        store_chunk_metadata(rows, driver, source_official={"S1": False})
        self.assertIn("c.source_official = row.source_official", driver.calls[1][0])
        self.assertIs(driver.calls[1][1]["rows"][0]["source_official"], False)
        missing_driver = RecordingNeo4jDriver()
        with self.assertRaisesRegex(ValueError, "source_id"):
            store_chunk_metadata(rows, missing_driver, source_official={})
        self.assertEqual(missing_driver.calls, [])

    def test_load_source_official_validates_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            path.write_text('{"source_id":"S1","official":true}\n', encoding="utf-8")
            self.assertEqual(load_source_official(path), {"S1": True})
            path.write_text('{"source_id":"S1","official":"true"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official"):
                load_source_official(path)

    def test_store_chunk_metadata_preserves_embeddings(self):
        rows = [
            {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1", "evidence_id": "E1", "raw_text": "Original A"},
            {"chunk_id": "B", "embedding_text": "banana", "source_id": "S2", "evidence_id": "E2", "raw_text": "Original B"},
        ]
        driver = RecordingNeo4jDriver()
        result = store_chunk_metadata(rows, driver, batch_size=1)
        self.assertEqual(result, {"count": 2})
        self.assertEqual(len(driver.calls), 3)
        for query, kwargs in driver.calls[1:]:
            self.assertIn("MERGE (c:Chunk {chunk_id: row.chunk_id})", query)
            self.assertIn("c.source_id = row.source_id", query)
            self.assertIn("c.evidence_id = row.evidence_id", query)
            self.assertIn("c.raw_text = row.raw_text", query)
            self.assertIn("c.embedding_dimension = size(c.embedding)", query)
            self.assertNotIn("SET c.embedding =", query)
            self.assertEqual(kwargs["database_"], "neo4j")
        self.assertEqual(driver.calls[1][1]["rows"], [
            {"chunk_id": "A", "source_id": "S1", "evidence_id": "E1", "raw_text": "Original A"}])

    def test_store_chunk_metadata_rejects_missing_required_fields_before_writes(self):
        for missing in ("source_id", "evidence_id", "raw_text"):
            with self.subTest(missing=missing):
                row = {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1",
                       "evidence_id": "E1", "raw_text": "original"}
                row.pop(missing)
                driver = RecordingNeo4jDriver()
                with self.assertRaisesRegex(ValueError, missing):
                    store_chunk_metadata([row], driver)
                self.assertEqual(driver.calls, [])

    def test_neo4j_store_cli_resume_skips_matching_model(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.jsonl"
            chunks.write_text(
                '{"chunk_id":"A","embedding_text":"apple"}\n'
                '{"chunk_id":"B","embedding_text":"banana"}\n', encoding="utf-8")

            class ResumeDriver(RecordingNeo4jDriver):
                def execute_query(self, query, **kwargs):
                    super().execute_query(query, **kwargs)
                    if "RETURN c.chunk_id AS chunk_id" in query:
                        return ([{"chunk_id": "A"}], None, None)

            driver = ResumeDriver()
            modules = {
                "dotenv": types.SimpleNamespace(load_dotenv=lambda: None),
                "neo4j": types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=lambda uri, auth: driver)),
                "sentence_transformers": types.SimpleNamespace(SentenceTransformer=lambda model: FakeEmbedder()),
            }
            env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "secret"}
            argv = ["vector_baseline.py", "neo4j-store", "--chunks", str(chunks),
                    "--model", "fake", "--resume"]
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()) as output:
                    main()
            self.assertEqual(json.loads(output.getvalue())["count"], 1)
            writes = [kwargs["rows"] for query, kwargs in driver.calls if "UNWIND $rows" in query]
            self.assertEqual([row["chunk_id"] for batch in writes for row in batch], ["B"])

    def test_neo4j_store_cli_only_upserts_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.jsonl"
            chunks.write_text('{"chunk_id":"A","embedding_text":"apple"}\n', encoding="utf-8")
            driver = RecordingNeo4jDriver()
            modules = {
                "dotenv": types.SimpleNamespace(load_dotenv=lambda: None),
                "neo4j": types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=lambda uri, auth: driver)),
                "sentence_transformers": types.SimpleNamespace(SentenceTransformer=lambda model: FakeEmbedder()),
            }
            env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "secret"}
            argv = ["vector_baseline.py", "neo4j-store", "--chunks", str(chunks), "--model", "fake"]
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()) as output:
                    main()
            self.assertEqual(json.loads(output.getvalue())["count"], 1)
            self.assertEqual(len(driver.calls), 2)
            self.assertFalse(any("CREATE VECTOR INDEX" in query for query, _ in driver.calls))

    def test_store_chunk_embeddings_upserts_without_creating_index(self):
        rows = [
            {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1", "evidence_id": "E1"},
            {"chunk_id": "B", "embedding_text": "banana", "source_id": "S2"},
        ]
        driver = RecordingNeo4jDriver()
        result = store_chunk_embeddings(rows, driver, FakeEmbedder(), "fake", batch_size=1)
        self.assertEqual(result, {"count": 2, "dimension": 2})
        self.assertEqual(len(driver.calls), 3)
        self.assertIn("CREATE CONSTRAINT", driver.calls[0][0])
        for query, kwargs in driver.calls[1:]:
            self.assertIn("MERGE (c:Chunk {chunk_id: row.chunk_id})", query)
            self.assertIn("c.embedding = row.embedding", query)
            self.assertIn("c.embedding_dimension = $dimension", query)
            self.assertEqual(kwargs["model_name"], "fake")
            self.assertEqual(kwargs["dimension"], 2)
            self.assertEqual(kwargs["database_"], "neo4j")
        self.assertEqual(driver.calls[1][1]["rows"][0]["chunk_id"], "A")
        self.assertEqual(driver.calls[1][1]["rows"][0]["properties"]["evidence_id"], "E1")
        self.assertEqual(driver.calls[2][1]["rows"][0]["chunk_id"], "B")
        self.assertFalse(any("CREATE VECTOR INDEX" in query for query, _ in driver.calls))

    def test_neo4j_build_cli_uses_environment_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.jsonl"
            chunks.write_text('{"chunk_id":"A","embedding_text":"apple"}\n', encoding="utf-8")
            driver = RecordingNeo4jDriver()
            connections = []
            def connect(uri, auth):
                connections.append((uri, auth))
                return driver
            modules = {
                "dotenv": types.SimpleNamespace(load_dotenv=lambda: None),
                "neo4j": types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=connect)),
                "sentence_transformers": types.SimpleNamespace(SentenceTransformer=lambda model: FakeEmbedder()),
            }
            env = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "secret"}
            argv = ["vector_baseline.py", "neo4j-build", "--chunks", str(chunks), "--model", "fake"]
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()) as output:
                    main()
            self.assertEqual(connections, [(env["NEO4J_URI"], (env["NEO4J_USER"], env["NEO4J_PASSWORD"]))])
            self.assertEqual(json.loads(output.getvalue())["count"], 1)

    def test_builds_chunk_nodes_and_vector_index_in_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.jsonl"
            chunks.write_text("\n".join(json.dumps(row) for row in [
                {"chunk_id": "A", "embedding_text": "apple", "source_id": "S1"},
                {"chunk_id": "B", "embedding_text": "banana", "source_id": "S2"},
            ]) + "\n", encoding="utf-8")
            driver = RecordingNeo4jDriver()
            result = build_neo4j_index(chunks, driver, FakeEmbedder(), "fake", batch_size=1)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["dimension"], 2)
            self.assertEqual(len(driver.calls), 4)
            self.assertIn("CREATE CONSTRAINT", driver.calls[0][0])
            for query, kwargs in driver.calls[1:3]:
                self.assertIn("MERGE (c:Chunk {chunk_id: row.chunk_id})", query)
                self.assertEqual(kwargs["database_"], "neo4j")
                self.assertEqual(len(kwargs["rows"]), 1)
            self.assertEqual(driver.calls[1][1]["rows"][0]["properties"]["source_id"], "S1")
            self.assertEqual(driver.calls[1][1]["rows"][0]["embedding"], [1.0, 0.0])
            self.assertIn("CREATE VECTOR INDEX", driver.calls[3][0])
            self.assertIn("`vector.dimensions`: 2", driver.calls[3][0])
            self.assertIn("'cosine'", driver.calls[3][0])

    def test_rejects_bad_batch_size_before_database_write(self):
        driver = RecordingNeo4jDriver()
        with self.assertRaisesRegex(ValueError, "batch_size"):
            build_neo4j_index("unused.jsonl", driver, FakeEmbedder(), "fake", batch_size=0)
        self.assertEqual(driver.calls, [])


if __name__ == "__main__":
    unittest.main()
