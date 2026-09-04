"""
OCR fallback engine for scanned legal documents and damaged text layers.
Uses Tesseract 5 (via pytesseract) for multilingual (Bengali + English) recognition.

CPU-only by design -- no GPU/VRAM usage. This keeps ingestion fully CPU-bound and
leaves the embedding stage as the only point where VRAM becomes a constraint,
per the project's hardware-scoping principle (i3 6th-gen / GTX 1050 Ti 4GB / 8GB RAM).
"""

import os
import shutil
from typing import List, Optional, Union
from pathlib import Path
from io import BytesIO
from loguru import logger

# Maps common ISO codes to Tesseract traineddata names
_LANG_MAP = {
    "bn": "ben",
    "ben": "ben",
    "en": "eng",
    "eng": "eng",
}

# Standard candidate paths on Windows where Tesseract-OCR is installed
_WINDOWS_TESSERACT_CANDIDATES = [
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def _discover_tesseract_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect the Tesseract executable across Windows and Linux.
    Precedence:
      1. Explicit argument
      2. TESSERACT_CMD environment variable
      3. System PATH (shutil.which)
      4. Common Windows installation directories
    """
    if explicit_path and Path(explicit_path).is_file():
        return str(Path(explicit_path).resolve())

    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return str(Path(env_cmd).resolve())

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return str(Path(which_cmd).resolve())

    if os.name == "nt":
        for candidate in _WINDOWS_TESSERACT_CANDIDATES:
            if candidate.is_file():
                return str(candidate.resolve())

    return None


class OCRFallbackEngine:
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        dpi: int = 300,
        psm: int = 6,
        oem: int = 3,
        tesseract_cmd: Optional[str] = None,
        tessdata_dir: Optional[str] = None,
        *args,
        engine: Optional[str] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Args:
            languages: ISO or Tesseract codes, e.g. ["bn", "en"] or ["ben", "eng"].
            dpi: Rendering DPI when converting PDF pages to images. 300 is recommended
                 for statute-quality print scans.
            psm: Page segmentation mode. 6 = uniform block of text (standard statute pages).
            oem: OCR engine mode. 3 = default (LSTM + legacy combined).
            tesseract_cmd: Optional path to tesseract binary.
            tessdata_dir: Optional path to traineddata directory.
            engine, use_gpu, *args, **kwargs: Accepted for backward compatibility with
                 earlier configs/pipeline callers. Tesseract 5 runs CPU-only.
        """
        self.languages = languages or ["bn", "en"]
        self.dpi = dpi
        self.psm = psm
        self.oem = oem
        self.tesseract_cmd = tesseract_cmd
        self.tessdata_dir = tessdata_dir
        self._tess_lang = self._build_lang_string(self.languages)
        self._checked = False
        self._available = False
        self._warned_unavailable = False

        if engine and engine.lower() != "tesseract":
            logger.debug(
                f"OCRFallbackEngine: Config requested '{engine}', using Tesseract 5 (CPU-only)."
            )

    @property
    def is_available(self) -> bool:
        """Check whether Tesseract binary and pytesseract are ready for OCR."""
        if not self._checked:
            self._check_tesseract()
        return self._available

    def _build_lang_string(self, languages: List[str]) -> str:
        mapped = [_LANG_MAP.get(lang.lower(), lang.lower()) for lang in languages]
        return "+".join(mapped)

    def _check_tesseract(self) -> None:
        """Verify the Tesseract binary and required traineddata are available. Runs once."""
        if self._checked:
            return
        self._checked = True
        self._available = False

        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract is not installed. Install via `pip install pytesseract pillow`.")
            return

        # Discover or configure binary
        bin_path = _discover_tesseract_binary(self.tesseract_cmd)
        if bin_path:
            pytesseract.pytesseract.tesseract_cmd = bin_path
        elif self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        if self.tessdata_dir and Path(self.tessdata_dir).is_dir():
            os.environ["TESSDATA_PREFIX"] = str(Path(self.tessdata_dir).resolve())

        try:
            version = pytesseract.get_tesseract_version()
            self._available = True
            logger.info(
                f"Tesseract OCR {version} initialized (cmd='{pytesseract.pytesseract.tesseract_cmd}', lang='{self._tess_lang}')"
            )

            available_langs = set(pytesseract.get_languages(config=""))
            missing = [l for l in self._tess_lang.split("+") if l not in available_langs]
            if missing:
                logger.warning(
                    f"Tesseract is available, but missing traineddata for: {missing}. "
                    f"Install Bengali via `sudo apt install tesseract-ocr-ben` (Linux) or "
                    f"download ben.traineddata into your Tesseract 'tessdata' folder (Windows)."
                )
        except Exception as e:
            self._available = False
            logger.warning(
                f"Tesseract binary not found or inaccessible: {e}. "
                f"Ensure Tesseract 5 is installed and in PATH, or set TESSERACT_CMD."
            )

    def ocr_image_or_page(self, image_path_or_bytes: Union[str, Path, bytes]) -> str:
        """
        Run OCR on an image file, path, or in-memory byte buffer and return extracted text.
        CPU-only -- safe to run alongside or before GPU-bound pipeline stages.
        """
        self._check_tesseract()
        if not self._available:
            if not self._warned_unavailable:
                logger.warning(
                    "Tesseract OCR is not available. Skipping OCR fallback (returning empty string)."
                )
                self._warned_unavailable = True
            return ""

        try:
            import pytesseract
        except ImportError:
            return ""

        try:
            image = self._load_image(image_path_or_bytes)
        except Exception as e:
            logger.warning(f"Failed to load image for OCR: {e}")
            return ""

        config = f"--psm {self.psm} --oem {self.oem}"
        try:
            text = pytesseract.image_to_string(image, lang=self._tess_lang, config=config)
        except pytesseract.TesseractError as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""

        return text.strip()

    def _load_image(self, image_path_or_bytes: Union[str, Path, bytes]):
        from PIL import Image

        if isinstance(image_path_or_bytes, (str, Path)):
            return Image.open(image_path_or_bytes)
        if isinstance(image_path_or_bytes, bytes):
            return Image.open(BytesIO(image_path_or_bytes))
        # Already a PIL Image or compatible buffer
        return image_path_or_bytes

    def render_pdf_page_to_image(self, pdf_path: Union[str, Path], page_number: int):
        """
        Render a single PDF page to a PIL Image at self.dpi using PyMuPDF (already a
        pipeline dependency, so no extra system package like poppler is needed).
        Use this to feed a specific low-quality page straight into ocr_image_or_page().
        """
        import fitz  # PyMuPDF
        from PIL import Image

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_number)
        zoom = self.dpi / 72  # PyMuPDF renders at 72 DPI by default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return image