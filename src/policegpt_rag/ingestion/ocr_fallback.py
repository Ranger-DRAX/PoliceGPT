"""
OCR fallback engine for scanned legal documents and damaged text layers.
Supports EasyOCR, PaddleOCR, and Tesseract wrappers with multilingual (bn + en) recognition.
"""

from typing import List, Optional
from pathlib import Path
from loguru import logger


class OCRFallbackEngine:
    def __init__(
        self,
        engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        dpi: int = 300,
    ):
        self.engine = engine.lower()
        self.languages = languages or ["bn", "en"]
        self.use_gpu = use_gpu
        self.dpi = dpi
        self._reader = None

    def _init_reader(self):
        if self._reader is not None:
            return

        if self.engine == "easyocr":
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR reader for languages: {self.languages} (gpu={self.use_gpu})")
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except ImportError:
                logger.warning("EasyOCR is not installed. Install via `pip install easyocr`.")
        elif self.engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR reader")
                self._reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_gpu)
            except ImportError:
                logger.warning("PaddleOCR is not installed. Install via `pip install paddleocr`.")
        else:
            logger.warning(f"Unsupported OCR engine '{self.engine}', defaulting to stub mode.")

    def ocr_image_or_page(self, image_path_or_bytes) -> str:
        """
        Run OCR on an image file or in-memory byte buffer and return extracted text.
        """
        self._init_reader()
        if self._reader is None:
            logger.warning("OCR reader unavailable; returning empty string.")
            return ""

        if self.engine == "easyocr":
            results = self._reader.readtext(image_path_or_bytes, detail=0)
            return "\n".join([str(t).strip() for t in results if str(t).strip()])

        if self.engine == "paddleocr":
            # PaddleOCR returns list of [[box, [text, confidence]]]
            results = self._reader.ocr(image_path_or_bytes, cls=True)
            extracted_lines = []
            if results and isinstance(results, list):
                for page_res in results:
                    if page_res:
                        for line in page_res:
                            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                extracted_lines.append(str(line[1][0]).strip())
            return "\n".join(extracted_lines)

        return ""
