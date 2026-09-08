# PoliceGPT — Legal Document RAG Pipeline

A specialized multilingual (Bengali & English) legal document processing and retrieval pipeline tailored for Bangladesh Police regulations, penal codes, criminal procedure, special acts, and regulatory documents.

> **Hardware Target:** Intel i3 6th-gen / NVIDIA GTX 1050 Ti (4 GB VRAM) / 8 GB RAM

---

## 🏛️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PDF Extraction (PyMuPDF)                                 │
│    • Page-by-page text extraction from legal statute PDFs    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Document Quality Check                                   │
│    • Bengali Unicode density (≥15% for expected BN docs)    │
│    • Valid character ratio (≥80%)                            │
│    • Minimum text length per page (≥50 chars)               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                   GOOD              POOR / SCANNED
                    │                     │
                    │                     ▼
                    │     ┌───────────────────────────────┐
                    │     │ 3. OCR Fallback (Tesseract 5) │
                    │     │    • CPU-only, bn + en        │
                    │     │    • 300 DPI page rendering    │
                    │     └───────────────┬───────────────┘
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Text Normalization & Cleaning                            │
│    • Unicode NFC + ZWJ/ZWNJ cleanup                        │
│    • Bangladesh Gazette header/footer stripping             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Legal Structure-Aware Chunking                           │
│    • Detects ধারা/Section/বিধি/Rule/Article boundaries      │
│    • Rich metadata: section_number, title, act, page refs   │
│    • Sliding window fallback for non-section text           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. BGE-M3 Embedding (CUDA fp16)                             │
│    • 1024-dim dense vectors + sparse lexical weights        │
│    • Batched with OOM recovery & CPU fallback               │
│    • Output: Parquet per document                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. FAISS & Lexical Indexing                                 │
│    • FAISS IndexFlatIP for 1024-dim dense vectors           │
│    • Inverted index for BGE-M3 sparse lexical weights       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Hybrid Retrieval + Reciprocal Rank Fusion (RRF)          │
│    • Dense cosine search + Sparse term dot-product search   │
│    • RRF(d) = Σ 1/(60 + rank_m) blends both rankings       │
│    • Returns top-5 fused statutory chunks with citations    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Reranker (Cross-Encoder) [code-complete, disabled]       │
│    • BGE-Reranker-Base, disabled by default (VRAM budget)   │
│    • Enable via use_reranker: true in retrieval.yaml        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. LLM Generation + Legal Guardrails (Gemini 3.5 Flash)    │
│     • Gemini 3.5 Flash Client, Bengali Prompt Templates     │
│     • Evidence Pre-Flight, Citation & Claim Guardrails      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Police-GPT/
├── README.md
├── agent.md                        # Embedding stage specification
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── ingestion.yaml              # PDF parser, quality thresholds, OCR (Tesseract 5)
│   ├── chunking.yaml               # Target chunk size, overlap, legal regex patterns
│   ├── embedding.yaml              # BGE-M3 model, batch size, CUDA/CPU, fp16
│   ├── retrieval.yaml              # FAISS + Sparse hybrid, RRF k=60, reranker toggle
│   └── generation.yaml             # Gemini 3.5 Flash model, temperature, guardrails
│
├── data/
│   ├── Police_Law/                 # Source statutes (Police Act 1861, PRB 1943, Penal Code 1860)
│   ├── interim/                    # Intermediate normalized/cleaned JSON per document
│   └── processed/
│       ├── *_chunks.json           # Chunked legal sections with metadata
│       ├── embeddings/             # Parquet files with dense + sparse vectors
│       └── indexes/                # FAISS binary index + sparse JSON + metadata
│
├── src/policegpt_rag/
│   ├── ingestion/
│   │   ├── parse_pdf.py            # PyMuPDF page-by-page extraction with quality gate
│   │   ├── quality_check.py        # Bengali Unicode density & corrupt font detection
│   │   └── ocr_fallback.py         # Tesseract 5 OCR (CPU-only, auto-discovery)
│   │
│   ├── preprocessing/
│   │   ├── normalize.py            # NFC normalization, ZWJ/ZWNJ cleanup, digit conversion
│   │   ├── boilerplate.py          # Bangladesh Gazette & header/footer remover
│   │   └── chunker.py              # LegalSectionChunker (ধারা/Section boundary chunking)
│   │
│   ├── embedding/
│   │   ├── embed.py                # BGEM3Embedder — dense + sparse, CUDA fp16, OOM recovery
│   │   └── batch_runner.py         # EmbeddingBatchRunner — Parquet I/O, resume support
│   │
│   ├── indexing/
│   │   ├── faiss_index.py          # FAISSVectorIndex — IndexFlatIP, L2-normalized cosine
│   │   └── sparse_index.py         # SparseLexicalIndex — inverted term dot-product index
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py        # HybridRetriever — Dense + Sparse + RRF fusion
│   │   └── rerank.py               # BGEReranker — cross-encoder (lazy-loaded, optional)
│   │
│   ├── generation/                 # Stage 10: Grounded Statutory LLM Generation
│   │   ├── llm_client.py           # Gemini 3.5 Flash REST client + MockLLMClient
│   │   ├── context_builder.py      # ContextAssembler (budget packing, [S1]/[S2] tags)
│   │   ├── prompt_templates.py     # Bengali statutory prompt builder & templates
│   │   ├── guardrails.py           # Pre-flight evidence, citations & claim guardrails
│   │   └── generator.py            # LegalGenerator end-to-end orchestrator
│   │
│   ├── evaluation/                 # ⚠️ Placeholders — not yet implemented
│   │   ├── retrieval_metrics.py
│   │   └── generation_metrics.py
│   │
│   └── pipeline.py                 # PoliceGPTChunkingPipeline orchestrator
│
├── scripts/
│   ├── run_ingestion.py            # Rich CLI: PDF → Quality → OCR → Clean → Chunk
│   ├── build_embeddings.py         # Rich CLI: Chunks → BGE-M3 → Parquet embeddings
│   ├── build_index.py              # Rich CLI: Parquet → FAISS + Sparse indexes
│   └── search_cli.py              # Rich CLI: Interactive hybrid legal search
│
├── tests/
│   ├── test_chunker.py             # Section-aware chunking tests
│   ├── test_quality_check.py       # Quality evaluator & encoding test cases (3 tests)
│   ├── test_ocr_fallback.py        # Tesseract 5 OCR engine tests (6 tests)
│   ├── test_embedding.py           # BGE-M3 embedder tests (mocked, 5+ tests)
│   └── test_indexing.py            # FAISS + Sparse + Hybrid RRF tests (5 tests)
│
├── api/                            # Production FastAPI REST Service (/query, /search, /health)
├── finetuning/                     # ⚠️ Placeholder — not yet implemented
└── venv/
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies (includes PyTorch CUDA 11.8 for GTX 1050 Ti)
pip install -r requirements.txt
```

**System Dependencies:**

| Dependency | Purpose | Install |
|---|---|---|
| **Tesseract 5** | OCR fallback for scanned pages | [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) or `sudo apt install tesseract-ocr tesseract-ocr-ben` |
| **ben.traineddata** | Bengali OCR language data | Included in `tesseract-ocr-ben` package or [download manually](https://github.com/tesseract-ocr/tessdata) |

> Tesseract is auto-discovered from PATH, standard Windows install directories, or the `TESSERACT_CMD` env var. If unavailable, OCR fallback degrades gracefully (logs a warning, returns partial PyMuPDF text).

### 2. Ingestion & Chunking

```bash
# Process all default law documents
python scripts/run_ingestion.py --all --save-chunks --report

# Process a single document and inspect chunks
python scripts/run_ingestion.py --doc policeAct --inspect 5

# Process an external PDF
python scripts/run_ingestion.py --pdf path/to/law_document.pdf --inspect 10
```

**Output:** `data/processed/<doc_id>_chunks.json`

### 3. Build Embeddings

```bash
# Embed all chunked documents (uses CUDA fp16 if available)
python scripts/build_embeddings.py --all

# Embed a single document
python scripts/build_embeddings.py --doc police_act_1861

# Override batch size for lower VRAM
python scripts/build_embeddings.py --all --batch-size 2

# Check GPU availability
python scripts/build_embeddings.py --check-gpu
```

**Output:** `data/processed/embeddings/<doc_id>_embeddings.parquet`

### 4. Build Hybrid Indexes

```bash
# Compile FAISS dense + Sparse lexical indexes from Parquet embeddings
python scripts/build_index.py
```

**Output:**
- `data/processed/indexes/dense_index.faiss` — FAISS IndexFlatIP binary
- `data/processed/indexes/sparse_index.json` — Inverted lexical index
- `data/processed/indexes/chunks_metadata.json` — Chunk text & metadata registry

### 5. Search Legal Statutes

```bash
# Single query (Bengali)
python scripts/search_cli.py --query "চুরির শাস্তি কি?"

# Single query (English)
python scripts/search_cli.py --query "powers of police officer to arrest without warrant"

# Interactive search session
python scripts/search_cli.py --interactive

# Custom top-k
python scripts/search_cli.py --query "ধারা ৩৭৮" --top-k 10
```

---

## 🛠️ Key Components

### Ingestion & Quality Gate (`ingestion/`)

- **`parse_pdf.py`** — PyMuPDF page-by-page extraction. Each page passes through a quality gate; poor pages trigger OCR fallback automatically.
- **`quality_check.py`** — Evaluates Bengali Unicode density (≥15%), valid character ratio (≥80%), and minimum text length (≥50 chars). Detects garbled legacy fonts (e.g., Bijoy ASCII encoding errors).
- **`ocr_fallback.py`** — Tesseract 5 via `pytesseract`. CPU-only by design (leaves GPU for embedding). Auto-discovers the Tesseract binary on Windows/Linux. Renders pages at 300 DPI via PyMuPDF. Graceful degradation if Tesseract is not installed.

### Preprocessing (`preprocessing/`)

- **`normalize.py`** — Unicode NFC normalization, ZWJ/ZWNJ cleanup (preserving valid Bengali Hasanta conjuncts), Bengali Dari (`।`) spacing.
- **`boilerplate.py`** — Strips recurring Bangladesh Gazette headers, ministry notices, and page number footers.
- **`chunker.py`** — Legal structure-aware chunking. Detects `ধারা`, `Section`, `বিধি`, `Rule`, `Article` boundaries. Produces `LegalChunk` pydantic models with `chunk_id`, `doc_id`, `section_number`, `section_title`, `act_name_bn/en`, `act_year`, `page_numbers`. Falls back to sliding window (512 tokens, 64 overlap) for non-section text.

### Embedding (`embedding/`)

- **`embed.py`** — `BGEM3Embedder` wrapping `BAAI/bge-m3`. Produces 1024-dim dense vectors and sparse lexical weights. Lazy-loads model on first use. CUDA fp16 on GPU, auto-fallback to CPU fp32. On `torch.cuda.OutOfMemoryError`: catches OOM, halves batch size, retries.
- **`batch_runner.py`** — `EmbeddingBatchRunner` reads `*_chunks.json`, streams through the embedder, writes `*_embeddings.parquet` with resume support (skips already-embedded chunk_ids).

### Indexing (`indexing/`)

- **`faiss_index.py`** — `FAISSVectorIndex` wrapping `faiss.IndexFlatIP`. L2-normalizes vectors for exact cosine similarity. Builds from Parquet files, enriches metadata from raw chunk JSON. Persists to `dense_index.faiss` + `chunks_metadata.json`.
- **`sparse_index.py`** — `SparseLexicalIndex` inverted index. Maps BGE-M3 learned lexical token keys to `(chunk_id, weight)` postings. Computes query-document dot-product scores for exact statutory term matching (e.g., `"ধারা ৩৭৮"`, `"arrest"`).

### Retrieval (`retrieval/`)

- **`hybrid_search.py`** — `HybridRetriever` runs parallel dense FAISS search + sparse inverted index search. Applies Reciprocal Rank Fusion: `RRF(d) = Σ 1/(60 + rank_m(d))`. Returns `RetrievedChunk` pydantic objects with section citations, act names, text content, and score breakdowns.
- **`rerank.py`** — `BGEReranker` using `BAAI/bge-reranker-base`. Lazy-loaded, disabled by default (`use_reranker: false` in `retrieval.yaml`) to conserve VRAM on the GTX 1050 Ti. Enable by setting `use_reranker: true`.

### Generation & Guardrails (`generation/`) — Stage 10

Production grounded statutory generation powered by **Google Gemini 3.5 Flash** with multi-tier legal guardrails:
- **`llm_client.py`** — `GeminiClient` implementing `BaseLLMClient` with exponential backoff on HTTP 429 / 503 retries for `gemini-3.5-flash`. Includes `MockLLMClient` for reproducible offline testing.
- **`context_builder.py`** — `ContextAssembler` assigns `[S1]`, `[S2]` source identifiers, deduplicates sections, and packs statutory text within a character budget (default 12,000 chars).
- **`prompt_templates.py`** — Bengali legal system instructions and statutory injection prompt enforcing mandatory citation tags and refusal on insufficient evidence.
- **`guardrails.py`** — Pre-flight `EvidenceChecker` (relevance score floors, ambiguous query detection), `CitationValidator` (resolves citations to statutory metadata), and `AnswerValidator` (lexical claim support & hallucination prevention).
- **`generator.py`** — `LegalGenerator` orchestrates the complete generation pipeline and returns structured `LegalAnswerResponse` models.

---

## 🧪 Running Tests

```bash
# Run all unit tests
pytest -v

# Individual test suites
pytest tests/test_chunker.py -v         # Legal section chunking
pytest tests/test_quality_check.py -v   # Quality checker (3 tests)
pytest tests/test_ocr_fallback.py -v    # Tesseract 5 OCR (6 tests)
pytest tests/test_embedding.py -v       # BGE-M3 embedder (mocked)
pytest tests/test_indexing.py -v        # FAISS + Sparse + Hybrid RRF (5 tests)
pytest tests/test_retrieval.py -v       # Retrieval hardening & IR metrics (12 tests)
pytest tests/test_rerank.py -v          # Cross-encoder reranker + VRAM guard (8 tests)
pytest tests/test_guardrails.py -v      # Context assembly, evidence, citations, claims (14 tests)
pytest tests/test_api.py -v             # FastAPI REST service & endpoints (9 tests)
```

All tests run without GPU or network access (models are mocked).

---

## 🌐 FastAPI REST Service

Start the production RESTful API server:

```bash
# Launch server with default settings (http://localhost:8000)
python scripts/run_api.py

# Launch with hot-reloading on a specific port
python scripts/run_api.py --port 8000 --reload
```

Interactive API documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Diagnostics:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### Core Endpoints:
- `POST /api/v1/query`: Full legal RAG question answering (Gemini 3.5 Flash generation + statutory citations + claim guardrails).
- `POST /api/v1/search`: Pure hybrid statutory retrieval (Dense FAISS + Sparse Inverted Index + RRF).
- `GET /api/v1/chunks/{chunk_id}`: Statutory chunk inspection and legal metadata lookup.
- `GET /api/v1/health`: Subsystem readiness, index vector counts, and reranker status.

---

## 📊 Benchmarking

Run the automated retrieval and reranker benchmarking suite:

```bash
# Fast validation / CI mode (synthetic mock models)
python scripts/benchmark_retrieval.py --synthetic

# Live benchmark on compiled indexes
python scripts/benchmark_retrieval.py --live --index-dir data/processed/indexes
```

---

## ⚙️ Configuration

| Config | Purpose |
|--------|---------|
| [`configs/ingestion.yaml`](configs/ingestion.yaml) | PDF parser, quality thresholds, Tesseract 5 OCR settings (DPI, PSM, OEM, binary path) |
| [`configs/chunking.yaml`](configs/chunking.yaml) | Target chunk size (512), overlap (64), min chunk size, legal regex patterns |
| [`configs/embedding.yaml`](configs/embedding.yaml) | BGE-M3 model, batch size (4), device (cuda/cpu), fp16, max_length, return_dense/sparse |
| [`configs/retrieval.yaml`](configs/retrieval.yaml) | FAISS dimension (1024), RRF k=60, dense/sparse top-k, reranker toggle, auto-disable on no-CUDA, min VRAM threshold |
| [`configs/generation.yaml`](configs/generation.yaml) | Gemini 3.5 Flash model (`gemini-3.5-flash`), temperature (0.1), context budget (12000 chars), evidence floors, support threshold |

---

## 📋 Implementation Status

| Stage | Component                             | Status                                      |
|-------|---------------------------------------|---------------------------------------------|
| 1     | PDF Extraction (PyMuPDF)              | ✅ Complete                                  |
| 2     | Document Quality Check                | ✅ Complete + Tests                         |
| 3     | OCR Fallback (Tesseract 5)            | ✅ Complete + Tests                         |
| 4     | Unicode Normalization & Cleaning      | ✅ Complete                                  |
| 5     | Legal Section Chunking                | ✅ Complete + Tests                         |
| 6     | BGE-M3 Embedding (CUDA fp16)          | ✅ Complete + Tests                         |
| 7     | FAISS Dense + Sparse Lexical Indexing | ✅ Complete + Tests                         |
| 8     | Hybrid Retrieval (RRF Fusion)         | ✅ Complete + Tests                         |
| 8.5   | Retrieval Validation & Hardening      | ✅ Complete + Tests                         |
| 9     | Cross-Encoder Reranker & Benchmarking | ✅ Complete + Tests (VRAM/CUDA Gated)       |
| 10    | LLM Generation + Guardrails (Gemini 3.5 Flash)  | ✅ Complete + Tests                         |
| —     | FastAPI Service                       | ✅ Complete + Tests                         |
| —     | Evaluation Metrics                    | ✅ Complete (`retrieval_metrics.py`)        |
| —     | Fine-tuning (LoRA)                    | 🔴 Placeholder                              |



