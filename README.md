# PoliceGPT — Legal Document Ingestion & Section Chunking

A specialized multilingual (Bengali & English) legal document processing and structure-aware chunking pipeline tailored for Bangladesh Police regulations, penal codes, criminal procedure, special acts, and regulatory documents.

> **Design Principle:** This repository is focused strictly on the **Document Ingestion & Chunking Layer** (PyMuPDF → Quality Detection → OCR Fallback → Text Cleaning → Legal Structure Detection → Section Chunking). Downstream components (embeddings, vector databases, retrieval, and LLMs) are decoupled to ensure the extraction and chunking quality is robust, verified, and transparent.

---

## 🏛️ Pipeline Architecture

```
[Raw Legal PDFs (Police Act, PRB, Penal Code)]
                      │
                      ▼
              ┌───────────────┐
              │    PyMuPDF    │
              └───────┬───────┘
                      │
                      ▼
              Extract Page Text
                      │
                      ▼
           ┌──────────────────────┐
           │ DocumentQualityCheck │
           └──────────┬───────────┘
                      │
               ┌──────┴──────┐
              GOOD          POOR / SCANNED / CORRUPT
               │             │
               │             ▼
               │        Render Page Image
               │             │
               │             ▼
               │     OCR Fallback (EasyOCR / PaddleOCR)
               │             │
               └──────┬──────┘
                      ▼
           Unicode NFC & ZWJ/ZWNJ Normalization
                      │
                      ▼
           Gazette & Header Boilerplate Cleaning
                      │
                      ▼
           Legal Structure Detection (ধারা / Section / বিধি / Rule)
                      │
                      ▼
           Legal Chunking with Rich Metadata
                      │
                      ▼
           CLI Inspection & JSON / Parquet Export
```

---

## 📂 Project Structure

```
Police-GPT/
├── README.md                       # Project documentation & CLI reference
├── agent.md                        # Architectural guidelines & processing flow
├── approach.md                     # Chunker flow summary
├── pyproject.toml                  # Packaging specification
├── requirements.txt                # Lightweight chunking dependencies
│
├── configs/
│   ├── ingestion.yaml              # PDF parser, quality thresholds, OCR settings
│   └── chunking.yaml               # Target chunk size, overlap %, legal regex patterns
│
├── data/
│   ├── Police_Law/                 # Source statutes (Police Act 1861, PRB 1943, Penal Code 1860)
│   └── processed/                  # Generated chunk JSONs and quality reports
│
├── src/policegpt_rag/
│   ├── ingestion/
│   │   ├── parse_pdf.py            # Page-by-page extraction with quality gate
│   │   ├── quality_check.py        # Bengali Unicode density & corrupt font detection
│   │   └── ocr_fallback.py         # EasyOCR & PaddleOCR fallback engine
│   │
│   ├── preprocessing/
│   │   ├── normalize.py            # NFC normalization, ZWJ/ZWNJ cleanup, digit conversion
│   │   ├── boilerplate.py          # Bangladesh Gazette & header/footer remover
│   │   └── chunker.py              # LegalSectionChunker (ধারা/Section boundary chunking)
│   │
│   └── pipeline.py                 # Core PoliceGPTChunkingPipeline orchestrator
│
├── scripts/
│   └── run_ingestion.py            # Rich CLI runner for document chunking & inspection
│
└── tests/
    ├── test_chunker.py             # Section-aware chunking tests
    └── test_quality_check.py       # Quality evaluator & encoding test cases
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Ingestion & Chunking CLI

```bash
# Process all default law documents
python scripts/run_ingestion.py

# Process only The Police Act, 1861 and inspect first 5 chunks
python scripts/run_ingestion.py --doc policeAct --inspect 5

# Process Police Regulations of Bengal
python scripts/run_ingestion.py --doc policeRegulations

# Process The Penal Code, 1860
python scripts/run_ingestion.py --doc penalCode

# Process an arbitrary external PDF
python scripts/run_ingestion.py --pdf path/to/law_document.pdf --inspect 10

# Save all chunks to JSON and generate a quality report
python scripts/run_ingestion.py --all --save-chunks --report
```

Output chunks and quality reports are saved to `data/processed/`:
* `data/processed/<doc_id>_chunks.json`
* `data/processed/quality_report.json`

---

## 🛠️ Key Components & Capabilities

### 1. Document Quality Checker (`quality_check.py`)
Evaluates each extracted page to ensure:
* **Character Length**: Rejects pages below `min_text_length_per_page=50` (flags scanned pages).
* **Bengali Unicode Density**: Checks for proper Bengali script (`[\u0980-\u09FF]`). Detects garbled legacy fonts (e.g. Bijoy ASCII encoding errors).
* **Valid Character Ratio**: Ensures character readability exceeds `min_valid_char_ratio=0.80`.
* **Automatic OCR Routing**: Automatically triggers OCR fallback when a page fails text layer checks.

### 2. OCR Fallback Engine (`ocr_fallback.py`)
* Automatically renders page images at 200–300 DPI.
* Supports **EasyOCR** and **PaddleOCR** with Bengali (`bn`) and English (`en`) support.
* Falls back gracefully if OCR packages are not locally installed.

### 3. Unicode Normalization & Cleaning (`normalize.py`, `boilerplate.py`)
* Performs Unicode **NFC** canonical decomposition and composition.
* Cleans unneeded Zero-Width Joiners (`\u200D`), Zero-Width Non-Joiners (`\u200C`), and BOM characters while preserving valid Bengali conjuncts (*Hasanta*).
* Regularizes Bengali Dari (`।`) punctuation spacing.
* Strips recurring Bangladesh Gazette headers, ministry notices, and page number footers.

### 4. Legal Structure-Aware Chunker (`chunker.py`)
* Detects statutory boundaries:
  * Bengali: `ধারা`, `দণ্ডবিধি`, `বিধি`, `অনুচ্ছেদ`
  * English: `Section`, `Rule`, `Article`, `Order`
* Produces structured `LegalChunk` models with:
  * `chunk_id`, `doc_id`, `act_name_bn`, `act_name_en`, `act_year`
  * `section_number` (e.g., `"৩৭৮"` or `"378"`)
  * `section_title` (e.g., `"চুরি"` or `"Theft"`)
  * `page_numbers` and sub-chunk tracking for large sections.

---

## 🧪 Running Automated Tests

Run unit tests via `pytest`:

```bash
# Run all unit tests
pytest -v

# Test section-aware chunker
pytest tests/test_chunker.py -v

# Test quality checker and encoding detection
pytest tests/test_quality_check.py -v
```

---

## ⚙️ Configuration

* [`configs/ingestion.yaml`](file:///k:/Police-GPT/configs/ingestion.yaml): Configure minimum text lengths, Bengali Unicode thresholds, and OCR engine settings.
* [`configs/chunking.yaml`](file:///k:/Police-GPT/configs/chunking.yaml): Configure target chunk size (default: 512), overlap (default: 64), and boundary regex patterns.
