"""
Ingestion module: Document parsing, quality validation, and OCR fallback.
"""

from .quality_check import DocumentQualityChecker, QualityCheckResult
from .ocr_fallback import OCRFallbackEngine
from .parse_pdf import PDFParser

__all__ = ["DocumentQualityChecker", "QualityCheckResult", "OCRFallbackEngine", "PDFParser"]
