there is two file location :
1. data\Police_Law\policeAct.pdf
2. data\Police_Law\policeRegulations.pdf
                         │
                         ▼
                    ┌─────────┐
                    │ PyMuPDF │
                    └────┬────┘
                         │
                         ▼
                 Extract page text
                         │
                         ▼
                ┌────────────────┐
                │ Quality Checker│
                └───────┬────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
             GOOD                POOR
              │                   │
              │                   ▼
              │              Render page
              │                   │
              │                   ▼
              │              PaddleOCR
              │                   │
              └────────┬──────────┘
                       ▼
                 Clean text
                       │
                       ▼
              Structure detection
                       │
                       ▼
                    Chunking
                       │
                       ▼
                 Test / Inspect

PyMuPDF → quality detection(Report Generation for all the pdfs) → OCR fallback → chunking.
for now No embeddings, no vector DB, no LLM, no LangChain/LlamaIndex yet. That makes it much easier to understand whether your document-processing layer is actually good.
