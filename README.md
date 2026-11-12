# mle-01-p3-team2

## Vector search baseline

`vector_baseline.py` embeds `data/processed/chunks.jsonl` (`embedding_text`) and
stores an exact cosine-search index. It evaluates retrieved `chunk_id` values
against `expected_source_chunk_ids` in `golden_set.jsonl`. The default multilingual
model is `BAAI/bge-m3` (1024-dimensional dense embeddings); the model
is downloaded on first use. The project dependencies include `numpy` and
`sentence-transformers`.

```powershell
python vector_baseline.py build --index-dir data/vector_index
python vector_baseline.py search "마늘 병해충 관리 방법은?" --k 5
python vector_baseline.py evaluate --k 5 --output data/vector_index/evaluation.json
```

`build` writes `vectors.npy`, `metadata.jsonl`, and `manifest.json`. Rebuild the
index when chunks or the embedding model change. `search` and `evaluate` load the
model recorded in the manifest; an explicitly supplied `--model` must match it.
Loading checks row counts, vector dimension, and file hashes against the manifest.
The index uses exact cosine similarity over all chunks, so no approximate-index
settings affect this baseline.
Chunk texts are encoded in ordered batches (default `--batch-size 4`) and
normalized before storage; Neo4j writes each completed batch without retaining
the full embedding matrix in memory.

Evaluation defaults to READY cases with `query_strategy="vector"`; use
`--strategy all` to include every strategy, or select `graph` or `hybrid`.
`Hit@k` is the fraction of cases with at least
one expected chunk in the top k. `Recall@k` is the mean fraction of each case's
expected chunks retrieved. `MRR@k` is the mean reciprocal rank of the first
expected chunk. The same filtering applies when calling `evaluate()` directly;
`all` also includes graph and hybrid cases for comparison. The report contains
per-case rankings. This measures retrieval, not answer quality or safety.

Run the dependency-light tests with `python -m unittest discover -s tests -v`.

## Neo4j vector index

With Neo4j running and `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` set
(for example in the project's `.env`), build the database index with:

```powershell
$env:HF_HOME = (Join-Path (Get-Location) '.hf-cache')
python vector_baseline.py neo4j-store --chunks data/processed/chunks.jsonl --batch-size 32 --resume
python vector_baseline.py neo4j-build --chunks data/processed/chunks.jsonl --batch-size 32
```

`neo4j-store` validates chunk IDs and text, embeds the chunks in batches, and
upserts `:Chunk` nodes by `chunk_id` with their `embedding`, `embedding_model`,
and `embedding_dimension`
properties. `neo4j-build` also creates `chunk_embedding` on the `embedding`
property with cosine similarity. These commands can be rerun for the same
embedding model; they do not
delete existing graph relationships. `--model`, `--batch-size`, and `--database`
can be supplied explicitly. `neo4j-store --resume` skips chunks that already
have an embedding from the selected model, so an interrupted run can continue.
To add source IDs and original chunk text without rerunning the embedding model:

```powershell
python vector_baseline.py neo4j-metadata --chunks data/processed/chunks.jsonl --source-registry data/processed/source_registry.jsonl
```

This stores `chunk_id`, `source_id`, `evidence_id`, and the JSONL `raw_text` as
Neo4j Chunk properties. It also records `source_official` from the source registry
and `embedding_dimension` from each stored vector. It leaves
existing embeddings and the vector index intact.
The Neo4j command requires the project's `neo4j`,
`python-dotenv`, and `sentence-transformers` dependencies. The local NumPy
baseline above remains available for exact-search comparison.

## Share the Neo4j Chunk vectors as a file

With Neo4j running and the same `.env` connection settings, export the project's
`Chunk` nodes and BGE-M3 vectors into one ZIP file:

```powershell
python vector_portability.py export
python vector_portability.py verify
```

The archive is `data/portable/chunk_vectors_bge_m3.zip`. Copy the ZIP under
`data/portable/` together with `vector_portability.py` to another computer.
Install `numpy`, `neo4j`, and
`python-dotenv`; set that computer's `NEO4J_URI`, `NEO4J_USER`, and
`NEO4J_PASSWORD` in its own `.env`. With Neo4j running, verify the file, then
restore the nodes and recreate the vector index:

```powershell
python -m pip install numpy neo4j python-dotenv
python vector_portability.py verify --archive data/portable/chunk_vectors_bge_m3.zip
python vector_portability.py restore --archive data/portable/chunk_vectors_bge_m3.zip
```

The archive includes `chunk_id`, `source_id`, `evidence_id`, `source_official`,
original `raw_text`, other stored
Chunk properties, and the 1024-dimensional vectors. Restoring these vectors
does not require downloading or running the embedding model. It does not include
unrelated Neo4j nodes or relationships. The archive is ignored by Git because
it is shared directly as a file.
공식 농업 지식 기반 식물 재배 의사결정 지원을 위한 하이브리드 GraphRAG 질의응답 시스템
