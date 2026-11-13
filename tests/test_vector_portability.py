import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.maintenance.vector_portability import (
    iter_portable_rows,
    restore_portable,
    verify_portable,
    write_portable,
)


class PortableVectorTests(unittest.TestCase):
    def sample_rows(self):
        return [
            {"chunk_id": "A", "source_id": "S1", "source_official": True,
             "evidence_id": "E1", "raw_text": "원문 A", "embedding": [1.0, 0.0],
             "embedding_model": "test-model", "embedding_dimension": 2},
            {"chunk_id": "B", "source_id": "S2", "source_official": False,
             "evidence_id": "E2", "raw_text": "원문 B", "embedding": [0.0, 1.0],
             "embedding_model": "test-model", "embedding_dimension": 2},
            {"chunk_id": "C", "source_id": "S3", "source_official": True,
             "evidence_id": "E3", "raw_text": "원문 C", "embedding": [0.6, 0.8],
             "embedding_model": "test-model", "embedding_dimension": 2},
        ]

    def test_archive_round_trip_and_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "portable.zip"
            manifest = write_portable(self.sample_rows(), root, "test-model", 2)
            self.assertEqual(manifest["count"], 3)
            with ZipFile(root) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "metadata.jsonl", "vectors.f32"})
            self.assertEqual(verify_portable(root)["dimension"], 2)
            restored = list(iter_portable_rows(root))
            self.assertEqual([row["chunk_id"] for row in restored], ["A", "B", "C"])
            self.assertEqual(restored[1]["raw_text"], "원문 B")
            self.assertEqual(restored[0]["evidence_id"], "E1")
            self.assertIs(restored[1]["source_official"], False)
            self.assertEqual(restored[1]["embedding"], [0.0, 1.0])

            root.write_bytes(root.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_portable(root)

    def test_import_restores_chunk_properties_and_vector_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "portable.zip"
            write_portable(self.sample_rows(), root, "test-model", 2)

            class FakeDriver:
                def __init__(self):
                    self.calls = []
                    self.index_created = False

                def execute_query(self, query, **kwargs):
                    self.calls.append((query, kwargs))
                    if query.startswith("SHOW VECTOR INDEXES"):
                        if self.index_created:
                            return ([{"name": "chunk_embedding", "state": "ONLINE",
                                      "labelsOrTypes": ["Chunk"], "properties": ["embedding"],
                                      "options": {"indexConfig": {"vector.dimensions": 2,
                                                                  "vector.similarity_function": "COSINE"}}}], None, None)
                        return ([], None, None)
                    if query.startswith("CREATE VECTOR INDEX"):
                        self.index_created = True
                    return ([], None, None)

            driver = FakeDriver()
            result = restore_portable(root, driver, batch_size=2)
            self.assertEqual(result["count"], 3)
            writes = [kwargs["rows"] for query, kwargs in driver.calls if query.startswith("UNWIND $rows")]
            self.assertEqual([row["chunk_id"] for batch in writes for row in batch], ["A", "B", "C"])
            self.assertEqual(writes[0][0]["properties"]["raw_text"], "원문 A")
            self.assertIs(writes[0][0]["properties"]["source_official"], True)
            self.assertEqual(writes[0][0]["embedding"], [1.0, 0.0])
            self.assertTrue(any(query.startswith("CREATE VECTOR INDEX") for query, _ in driver.calls))

    def test_export_requires_evidence_id(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.sample_rows()[0]
            del row["evidence_id"]
            with self.assertRaisesRegex(ValueError, "evidence_id"):
                write_portable([row], Path(directory) / "portable.zip", "test-model", 2)

    def test_export_requires_source_official(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.sample_rows()[0]
            del row["source_official"]
            with self.assertRaisesRegex(ValueError, "source_official"):
                write_portable([row], Path(directory) / "portable.zip", "test-model", 2)


if __name__ == "__main__":
    unittest.main()
