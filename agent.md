# Agentic Coding Prompt — Implement BGE-M3 Embedding Stage (PoliceGPT)

Paste everything below into your agentic coding tool (Claude Code, etc.) run from the repo root.

---

## Context

You are working in the **PoliceGPT-RAG** repository. The ingestion + legal-structure-aware chunking stage is already complete and working (`src/policegpt_rag/ingestion/`, `src/policegpt_rag/preprocessing/chunker.py`, `scripts/run_ingestion.py`). Chunks are already being produced as `LegalChunk` pydantic objects and saved to `data/processed/<doc_id>_chunks.json` via `--save-chunks`.

The `embedding/` module is currently a placeholder:
- `src/policegpt_rag/embedding/embed.py`
- `src/policegpt_rag/embedding/batch_runner.py`
- `configs/embedding.yaml`

Your task is to implement this stage: **read the chunk JSON files, embed each chunk with BGE-M3, and persist embeddings to disk in a format ready for FAISS indexing** (the indexing stage itself is out of scope — don't implement it, just make sure your output format is clean and documented).

## Hardware constraints — must respect these

- CPU: i3 6th-gen
- GPU: GTX 1050 Ti, **4GB VRAM**
- RAM: 8GB

This means:
- Use `BGEM3FlagModel` from the `FlagEmbedding` library with `use_fp16=True` on CUDA (BGE-M3 fp16 weights are ~1.1GB, so this fits comfortably, but batch activation memory must stay bounded).
- Default batch size must be **small** (start at 4, make it configurable) — do not assume large-batch throughput.
- Must gracefully fall back to CPU (`use_fp16=False` on CPU) if CUDA is unavailable or a CUDA OOM is raised — catch the OOM, log a warning, halve the batch size, and retry rather than crashing the whole run.
- Follow the project's existing principle: **run this as a separate script invocation**, not chained in-memory with ingestion/chunking, to keep peak RAM bounded. Don't try to import and call the chunking pipeline directly in the same process as embedding.
- Load the model **lazily** (on first `.encode()` call or explicit `.load()`), never at import time — earlier stages of this repo need to run without pulling in torch/FlagEmbedding as a hard import cost.

## What to build

### 1. `configs/embedding.yaml`
Replace the current placeholder with real config, e.g.:
```yaml
model_name: "BAAI/bge-m3"
device: "cuda"          # falls back to cpu automatically if unavailable
use_fp16: true
batch_size: 4
max_length: 512          # token truncation length, align with chunking target_chunk_size
return_dense: true
return_sparse: true
return_colbert_vecs: false   # ColBERT vectors are heavy; keep off by default given VRAM budget
normalize_embeddings: true
paths:
  input_dir: "data/processed"
  output_dir: "data/processed/embeddings"
```

### 2. `src/policegpt_rag/embedding/embed.py`
Implement a `BGEM3Embedder` class:
- Constructor takes the config fields above (or a config dict/pydantic settings object — match the style already used in `DocumentQualityChecker`/`OCRFallbackEngine`, i.e. plain `__init__` kwargs with sane defaults, not a new config framework).
- Lazy-loads `BGEM3FlagModel` on first use; if `device="cuda"` but `torch.cuda.is_available()` is False, log a warning and fall back to `"cpu"` (and force `use_fp16=False` on CPU, matching known BGE-M3 constraints).
- `embed_chunks(chunks: List[LegalChunk]) -> EmbeddingBatch` (or similar) — takes `LegalChunk` objects (import from `policegpt_rag.preprocessing.chunker`), extracts `.content` as the text to embed, and returns dense vectors (and sparse lexical weights, if enabled) aligned to `chunk_id`.
- Must batch internally according to `batch_size`, not embed one giant list at once.
- On `torch.cuda.OutOfMemoryError`: catch it, call `torch.cuda.empty_cache()`, halve the effective batch size for the remainder of the run, log via `loguru`, and continue rather than aborting.
- Return a pydantic model (mirror the `LegalChunk` style) capturing per-chunk: `chunk_id`, `dense_vector: List[float]`, `sparse_weights: Optional[Dict[str, float]]`, `embedding_model: str`, `embedding_dim: int`.

### 3. `src/policegpt_rag/embedding/batch_runner.py`
Implement an `EmbeddingBatchRunner` that:
- Reads one `data/processed/<doc_id>_chunks.json` file, parses it back into `LegalChunk` objects.
- Streams chunks through `BGEM3Embedder` in batches (don't hold all embeddings for a huge document in memory longer than needed — write incrementally if the chunk count is large).
- Writes output to `data/processed/embeddings/<doc_id>_embeddings.parquet` (use `pyarrow`, already a dependency) with columns: `chunk_id, doc_id, dense_vector, sparse_weights_json, section_number, section_title, embedding_model`. Parquet over JSON here because dense float vectors get large fast and this will feed FAISS next.
- Supports resuming: if the output parquet already has embeddings for a given `chunk_id`, skip re-embedding it (idempotent reruns — important on this hardware where a run might get interrupted).

### 4. `scripts/build_embeddings.py`
New CLI script, following the same Rich-console UX pattern as `scripts/run_ingestion.py` (reuse `rich.console.Console`, `rich.progress.Progress`, a summary table at the end). Should support:
```bash
python scripts/build_embeddings.py --doc police_act_1861
python scripts/build_embeddings.py --all
python scripts/build_embeddings.py --all --batch-size 2   # override config for lower-VRAM runs
```
It should read chunk JSON files from `data/processed/`, discover `doc_id`s automatically for `--all` (glob `*_chunks.json`), and print a per-document summary: chunk count, embedding dim, time elapsed, any OOM-triggered batch-size reductions that occurred.

### 5. `tests/test_embedding.py`
- Unit test that mocks `BGEM3FlagModel.encode` (do **not** actually download/run the real model in CI — mock it, same pattern as `tests/test_ocr_fallback.py` uses `unittest.mock.patch`).
- Test: batching logic splits N chunks into correctly-sized batches.
- Test: CPU fallback logic sets `use_fp16=False` when device resolves to `"cpu"`.
- Test: resume logic skips chunk_ids already present in an existing output parquet.

### 6. Dependencies
Add to `requirements.txt` (or a new `embedding` optional-dependencies group in `pyproject.toml`, matching the existing `ocr`/`eval`/`dev` extras pattern):
```
FlagEmbedding>=1.2.0
torch>=2.1.0
pyarrow>=15.0.0   # already present
```

## Do NOT

- Do not modify `preprocessing/chunker.py`, `ingestion/*`, or any files explicitly scoped as "decoupled from this repository" placeholders (indexing, retrieval, generation, api, finetuning) — this task is embedding only.
- Do not hardcode CUDA as a requirement anywhere — CPU must always work, just slower.
- Do not load the full BGE-M3 model at module import time.
- Do not implement FAISS indexing — stop at producing the embeddings parquet.

## Acceptance criteria

- `python scripts/build_embeddings.py --doc police_act_1861` runs end-to-end on a `_chunks.json` file produced by the existing ingestion script and writes a valid parquet file.
- Running it twice in a row (without deleting output) does not re-embed already-embedded chunks.
- Forcing `device="cpu"` in the config still works correctly with `use_fp16` disabled.
- `pytest tests/test_embedding.py -v` passes without requiring GPU or network access (mocked model).