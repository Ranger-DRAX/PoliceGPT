"""
Unicode Normalization and cleanup for Bengali and English legal text.
Handles NFC normalization, Zero-Width Joiner (ZWJ), Zero-Width Non-Joiner (ZWNJ), and whitespace normalization.
"""

import unicodedata
import re


class UnicodeNormalizer:
    def __init__(self, remove_extra_whitespace: bool = True, normalize_digits: bool = False):
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_digits = normalize_digits

        # Bengali digit to English mapping
        self.bn_to_en_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
        self.en_to_bn_digits = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

    def clean_zwj_zwnj(self, text: str) -> str:
        """
        Clean unneeded Zero-Width Joiners (\u200D) and Zero-Width Non-Joiners (\u200C)
        while preserving valid Bengali conjuncts (e.g. Hasanta combinations).
        """
        # Remove standalone or repeated ZWJ/ZWNJ
        text = re.sub(r"[\u200B\u200E\u200F\uFEFF]", "", text)  # Zero width spaces, marks, BOM
        text = re.sub(r"[\u200C\u200D]{2,}", "", text)          # Consecutive ZWJ/ZWNJ
        return text

    def normalize(self, text: str) -> str:
        """
        Perform standard NFC normalization, ZWJ/ZWNJ cleanup, and whitespace regularization.
        """
        if not text:
            return ""

        # 1. Unicode NFC Normalization (canonical composition)
        text = unicodedata.normalize("NFC", text)

        # 2. Clean unneeded invisible control characters
        text = self.clean_zwj_zwnj(text)

        # 3. Regularize Bengali Dari (।) and standard punctuation spaces
        text = re.sub(r"\s+([।,;:?!])", r"\1", text)
        text = re.sub(r"([।,;:?!])(?=[^\s।,;:?!0-9])", r"\1 ", text)

        # 4. Whitespace cleanup
        if self.remove_extra_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            text = text.strip()

        return text

    def to_english_digits(self, text: str) -> str:
        return text.translate(self.bn_to_en_digits)

    def to_bengali_digits(self, text: str) -> str:
        return text.translate(self.en_to_bn_digits)
