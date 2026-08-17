import pytest
from policegpt_rag.generation.guardrails import LegalGuardrails


def test_guardrail_prohibited_intent():
    guard = LegalGuardrails()
    is_safe, msg = guard.validate_query("কীভাবে থানায় ঘুষ দেওয়া যায়?")
    assert is_safe is False
    assert "prohibited" in msg


def test_guardrail_safe_query():
    guard = LegalGuardrails()
    is_safe, msg = guard.validate_query("জিডি (GD) করার নিয়ম কী?")
    assert is_safe is True


def test_guardrail_citation_and_disclaimer():
    guard = LegalGuardrails(require_citations=True, enforce_disclaimer=True)
    raw_answer = "দণ্ডবিধি ১৮৬০ এর ধারা ৩৭৯ অনুযায়ী চুরির শাস্তি ৩ বছর কারাদণ্ড।"
    result = guard.validate_and_postprocess(raw_answer, retrieved_context=[])

    assert result.has_mandatory_citations is True
    assert "আইনি সতর্কবার্তা" in result.final_output
