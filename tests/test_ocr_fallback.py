"""
Unit tests for OCRFallbackEngine (Tesseract 5 integration).
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
from io import BytesIO
import pytest
from PIL import Image

try:
    from policegpt_rag.ingestion.ocr_fallback import (
        OCRFallbackEngine,
        _discover_tesseract_binary,
    )
except ImportError:
    from src.policegpt_rag.ingestion.ocr_fallback import (
        OCRFallbackEngine,
        _discover_tesseract_binary,
    )


def test_language_string_mapping():
    engine = OCRFallbackEngine(languages=["bn", "en"])
    assert engine._tess_lang == "ben+eng"

    engine_custom = OCRFallbackEngine(languages=["ben", "eng"])
    assert engine_custom._tess_lang == "ben+eng"


def test_backward_compatibility_init():
    """Ensure older callers passing legacy kwargs do not throw TypeError."""
    engine = OCRFallbackEngine(
        engine="easyocr",
        use_gpu=True,
        dpi=300,
        some_future_kwarg="ignored",
    )
    assert engine.dpi == 300
    assert engine.languages == ["bn", "en"]


def test_binary_discovery_custom_path(tmp_path):
    fake_exe = tmp_path / "tesseract.exe"
    fake_exe.write_text("fake binary")

    discovered = _discover_tesseract_binary(str(fake_exe))
    assert discovered == str(fake_exe.resolve())


def test_binary_discovery_env_var(tmp_path, monkeypatch):
    fake_exe = tmp_path / "custom_tesseract.exe"
    fake_exe.write_text("fake binary")

    monkeypatch.setenv("TESSERACT_CMD", str(fake_exe))
    discovered = _discover_tesseract_binary()
    assert discovered == str(fake_exe.resolve())


def test_ocr_unavailable_graceful_fallback():
    """If Tesseract is not available, ocr_image_or_page returns empty string without error."""
    engine = OCRFallbackEngine()
    engine._checked = True
    engine._available = False

    result = engine.ocr_image_or_page(b"fake image bytes")
    assert result == ""
    assert engine.is_available is False


def test_ocr_successful_extraction_mocked():
    """Mock pytesseract to verify image_to_string execution and output stripping."""
    engine = OCRFallbackEngine(languages=["bn", "en"], psm=6, oem=3)
    engine._checked = True
    engine._available = True

    # Create a small dummy in-memory image
    img = Image.new("RGB", (60, 30), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with patch("pytesseract.image_to_string", return_value="   বাংলাদেশ পুলিশ আইন ১৮৬১   \n"):
        text = engine.ocr_image_or_page(img_bytes)
        assert text == "বাংলাদেশ পুলিশ আইন ১৮৬১"
