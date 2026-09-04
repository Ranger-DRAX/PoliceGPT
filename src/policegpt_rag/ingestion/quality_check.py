"""
Bengali / English text layer quality checker for extracted PDFs.
Evaluates Bengali Unicode ratios, valid character density, and flags corrupted or scanned pages.
"""

from typing import Dict, Any
from pydantic import BaseModel
import re
from loguru import logger


class QualityCheckResult(BaseModel):
    passed: bool
    text_length: int
    bangla_ratio: float
    valid_char_ratio: float
    needs_ocr: bool
    reason: str


class DocumentQualityChecker:
    def __init__(
        self,
        min_text_length_per_page: int = 50,
        min_bangla_unicode_ratio: float = 0.15,
        min_valid_char_ratio: float = 0.80,
    ):
        self.min_text_length = min_text_length_per_page
        self.min_bangla_ratio = min_bangla_unicode_ratio
        self.min_valid_char_ratio = min_valid_char_ratio

        # Bengali Unicode block: U+0980 to U+09FF
        self.bangla_regex = re.compile(r"[\u0980-\u09FF]")
        # Valid printable characters (Bengali, English, numbers, standard punctuation)
        self.valid_regex = re.compile(r"[\u0980-\u09FFa-zA-Z0-9\s.,;:?!()\[\]\"\'\-\/—।]")

    def evaluate_text(self, text: str, is_expected_bangla: bool = True) -> QualityCheckResult:
        """
        Evaluate text layer quality of a single page or document chunk.
        """
        if not text or len(text.strip()) < self.min_text_length:
            return QualityCheckResult(
                passed=False,
                text_length=len(text.strip()) if text else 0,
                bangla_ratio=0.0,
                valid_char_ratio=0.0,
                needs_ocr=True,
                reason="Text layer empty or below minimum character threshold (likely scanned image).",
            )

        clean_text = text.replace(" ", "").replace("\n", "")
        total_chars = max(len(clean_text), 1)

        bangla_chars = len(self.bangla_regex.findall(clean_text))
        valid_chars = len(self.valid_regex.findall(clean_text))

        bangla_ratio = bangla_chars / total_chars
        valid_char_ratio = valid_chars / total_chars

        needs_ocr = False
        reasons = []

        if valid_char_ratio < self.min_valid_char_ratio:
            needs_ocr = True
            reasons.append(f"Low valid character ratio ({valid_char_ratio:.2f} < {self.min_valid_char_ratio}). Possible font encoding corruption.")

        if is_expected_bangla and bangla_ratio < self.min_bangla_ratio:
            # If doc is expected to be in Bengali but has nearly zero Bengali glyphs (e.g. Bijoy ASCII encoding garbled)
            needs_ocr = True
            reasons.append(f"Expected Bengali document but detected low Bengali Unicode ratio ({bangla_ratio:.2f}).")

        passed = not needs_ocr
        reason_str = "; ".join(reasons) if reasons else "Text layer passed quality checks."

        return QualityCheckResult(
            passed=passed,
            text_length=len(text),
            bangla_ratio=bangla_ratio,
            valid_char_ratio=valid_char_ratio,
            needs_ocr=needs_ocr,
            reason=reason_str,
        )
