"""
PDF parsing module with PyMuPDF / pdfplumber integration, quality checks, and OCR fallback.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from loguru import logger

from .quality_check import DocumentQualityChecker, QualityCheckResult
from .ocr_fallback import OCRFallbackEngine


class ParsedPage:
    def __init__(self, page_number: int, text: str, quality: QualityCheckResult, metadata: Dict[str, Any]):
        self.page_number = page_number
        self.text = text
        self.quality = quality
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "quality": self.quality.model_dump(),
            "metadata": self.metadata,
        }


class PDFParser:
    def __init__(
        self,
        quality_checker: Optional[DocumentQualityChecker] = None,
        ocr_engine: Optional[OCRFallbackEngine] = None,
    ):
        self.quality_checker = quality_checker or DocumentQualityChecker()
        self.ocr_engine = ocr_engine or OCRFallbackEngine()

    def parse_pdf(self, file_path: str | Path, is_expected_bangla: bool = True) -> List[ParsedPage]:
        """
        Extract text page-by-page. If quality check fails (e.g. scanned/garbled), trigger OCR fallback.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        logger.info(f"Parsing document: {path.name}")
        parsed_pages: List[ParsedPage] = []

        doc = fitz.open(str(path))
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            extractedText= page.get_text("text")

            # Check quality
            quality = self.quality_checker.evaluate_text(
                extractedText, is_expected_bangla=is_expected_bangla
            )

            # Trigger OCR if required
            final_text = extractedText
            ocr_attempted = False
            ocr_recovered = False
            ocr_chars = 0

            if quality.needs_ocr:
                logger.warning(
                    f"Page {page_num} in '{path.name}' failed text-layer quality check ({quality.reason}). Triggering OCR..."
                )
                ocr_attempted = True
                dpi = getattr(self.ocr_engine, "dpi", 300)
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                ocr_text = self.ocr_engine.ocr_image_or_page(img_bytes)
                if ocr_text.strip():
                    final_text = ocr_text
                    ocr_recovered = True
                    ocr_chars = len(ocr_text.strip())

            page_meta = {
                "source_file": path.name,
                "file_path": str(path.resolve()),
                "total_pages": len(doc),
                "ocr_attempted": ocr_attempted,
                "ocr_recovered": ocr_recovered,
                "ocr_chars": ocr_chars,
            }

            parsed_pages.append(
                ParsedPage(
                    page_number=page_num,
                    text=final_text,
                    quality=quality,
                    metadata=page_meta,
                )
            )

        doc.close()
        return parsed_pages
