import pytest
from policegpt_rag.ingestion.quality_check import DocumentQualityChecker


def test_quality_checker_valid_bengali_text():
    checker = DocumentQualityChecker()
    valid_bn_text = "দণ্ডবিধি ১৮৬০ এর ধারা ৩৭৮ অনুযায়ী চুরির সংজ্ঞা প্রদান করা হয়েছে।"
    res = checker.evaluate_text(valid_bn_text, is_expected_bangla=True)
    assert res.passed is True
    assert res.needs_ocr is False
    assert res.bangla_ratio > 0.15


def test_quality_checker_scanned_or_empty_text():
    checker = DocumentQualityChecker()
    empty_text = "   "
    res = checker.evaluate_text(empty_text, is_expected_bangla=True)
    assert res.passed is False
    assert res.needs_ocr is True


def test_quality_checker_corrupted_encoding():
    checker = DocumentQualityChecker()
    # Simulating garbled ASCII text instead of Bengali unicode
    garbled_text = "‡Kv‡bv e¨w³ Ab¨ e¨w³i m¤úwË Pzwi Kwi‡j Zvnvi kvw¯Í nB‡e|"
    res = checker.evaluate_text(garbled_text, is_expected_bangla=True)
    assert res.needs_ocr is True
