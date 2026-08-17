"""
CLI Script: Batch Ingestion from data/raw -> data/interim
Scans raw directories, extracts text layers with quality validation & OCR fallback, saves interim JSONs.
"""

from pathlib import Path
import json
import argparse
from loguru import logger
import sys

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policegpt_rag.ingestion.parse_pdf import PDFParser
from policegpt_rag.ingestion.quality_check import DocumentQualityChecker
from policegpt_rag.ingestion.ocr_fallback import OCRFallbackEngine


def run_ingestion(raw_dir: str = "data/raw", interim_dir: str = "data/interim"):
    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    parser = PDFParser(
        quality_checker=DocumentQualityChecker(),
        ocr_engine=OCRFallbackEngine(),
    )

    pdf_files = list(raw_path.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found under {raw_path}. Add files to data/raw/ to ingest.")
        return

    logger.info(f"Discovered {len(pdf_files)} PDF documents to ingest.")
    for pdf in pdf_files:
        try:
            pages = parser.parse_pdf(pdf)
            output_file = interim_path / f"{pdf.stem}_interim.json"
            data = {
                "source_file": pdf.name,
                "pages": [p.to_dict() for p in pages],
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved parsed interim data: {output_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoliceGPT Raw Document Ingestion")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw PDFs")
    parser.add_argument("--interim-dir", default="data/interim", help="Path for parsed output")
    args = parser.parse_args()

    run_ingestion(args.raw_dir, args.interim_dir)
