"""
Unit and hardening tests for Context Assembly, Evidence Checking, Citation Validation,
Answer Validation, and End-to-End LegalGenerator.
All LLM and network calls are mocked: zero API keys or external services required.
"""

import pytest
from unittest.mock import MagicMock, patch

try:
    from policegpt_rag.generation.context_builder import ContextAssembler, SourceContext, AssembledContext
    from policegpt_rag.generation.guardrails import (
        EvidenceChecker,
        CitationValidator,
        AnswerValidator,
        SourceReference,
    )
    from policegpt_rag.generation.llm_client import MockLLMClient, LLMResponse
    from policegpt_rag.generation.generator import LegalGenerator, LegalAnswerResponse
except ImportError:
    from src.policegpt_rag.generation.context_builder import ContextAssembler, SourceContext, AssembledContext
    from src.policegpt_rag.generation.guardrails import (
        EvidenceChecker,
        CitationValidator,
        AnswerValidator,
        SourceReference,
    )
    from src.policegpt_rag.generation.llm_client import MockLLMClient, LLMResponse
    from src.policegpt_rag.generation.generator import LegalGenerator, LegalAnswerResponse


def _create_sample_chunks():
    """Helper to create sample retrieved statutory chunks."""
    return [
        {
            "chunk_id": "penal_378",
            "doc_id": "penal_code_1860",
            "act_name_en": "The Penal Code, 1860",
            "act_name_bn": "দণ্ডবিধি, ১৮৬০",
            "section_number": "378",
            "section_title": "Theft",
            "page_numbers": [124],
            "content": "Whoever, intending to take dishonestly any moveable property out of the possession of any person...",
            "rrf_score": 0.032,
            "dense_score": 0.85,
        },
        {
            "chunk_id": "penal_379",
            "doc_id": "penal_code_1860",
            "act_name_en": "The Penal Code, 1860",
            "act_name_bn": "দণ্ডবিধি, ১৮৬০",
            "section_number": "379",
            "section_title": "Punishment for theft",
            "page_numbers": [125],
            "content": "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
            "rrf_score": 0.030,
            "dense_score": 0.82,
        },
    ]


# ---------------------------------------------------------------------------
# 1. Context Assembly Tests
# ---------------------------------------------------------------------------

def test_context_assembler_source_tagging():
    """Verify context assembler assigns [S1], [S2] tags and renders statutory metadata."""
    assembler = ContextAssembler()
    chunks = _create_sample_chunks()

    assembled = assembler.assemble(chunks)
    assert len(assembled.sources) == 2
    assert assembled.sources[0].source_id == "S1"
    assert assembled.sources[1].source_id == "S2"
    assert "[উৎস S1]" in assembled.formatted_context
    assert "দণ্ডবিধি, ১৮৬০" in assembled.formatted_context
    assert "ধারা 378" in assembled.formatted_context


def test_context_assembler_deduplication():
    """Verify duplicate chunks with identical IDs or text are filtered out."""
    assembler = ContextAssembler()
    chunks = _create_sample_chunks()
    # Add an exact duplicate
    chunks.append(chunks[0])

    assembled = assembler.assemble(chunks)
    assert len(assembled.sources) == 2
    assert assembled.num_deduplicated == 1


def test_context_assembler_budget_enforcement():
    """Verify context assembly halts when the character budget is reached."""
    assembler = ContextAssembler(max_context_chars=120)
    chunks = _create_sample_chunks()

    assembled = assembler.assemble(chunks, max_chars=120)
    # First chunk is ~98 chars; second chunk would exceed 120 chars -> only 1 accepted
    assert len(assembled.sources) == 1
    assert assembled.sources[0].source_id == "S1"


# ---------------------------------------------------------------------------
# 2. Evidence Checking Tests
# ---------------------------------------------------------------------------

def test_evidence_checker_empty_chunks():
    """Verify empty chunk list triggers insufficient_evidence status."""
    checker = EvidenceChecker()
    result = checker.check("চুরির শাস্তি কি?", [])

    assert result.can_generate is False
    assert result.status == "insufficient_evidence"


def test_evidence_checker_ambiguous_query():
    """Verify single generic word triggers needs_clarification status."""
    checker = EvidenceChecker()
    chunks = _create_sample_chunks()

    res1 = checker.check("আইন", chunks)
    assert res1.can_generate is False
    assert res1.status == "needs_clarification"

    res2 = checker.check("police", chunks)
    assert res2.can_generate is False
    assert res2.status == "needs_clarification"


def test_evidence_checker_low_relevance_scores():
    """Verify extremely low RRF and dense scores trigger insufficient_evidence."""
    checker = EvidenceChecker(min_rrf_score=0.02, min_dense_score=0.50)
    low_chunks = [
        {
            "chunk_id": "c1",
            "content": "random",
            "rrf_score": 0.005,
            "dense_score": 0.20,
        }
    ]
    result = checker.check("rare query", low_chunks)
    assert result.can_generate is False
    assert result.status == "insufficient_evidence"


def test_evidence_checker_valid_query_passes():
    """Verify clear query with good chunks passes to generation."""
    checker = EvidenceChecker()
    chunks = _create_sample_chunks()

    result = checker.check("চুরির শাস্তি কি?", chunks)
    assert result.can_generate is True
    assert result.status == "answered"


# ---------------------------------------------------------------------------
# 3. Citation Validation Tests
# ---------------------------------------------------------------------------

def test_citation_validator_valid_citations():
    """Verify valid citations [S1] resolve to full statutory metadata."""
    validator = CitationValidator()
    assembler = ContextAssembler()
    assembled = assembler.assemble(_create_sample_chunks())

    answer = "চুরির শাস্তি হলো অনধিক ৩ বছর কারাদণ্ড [S2] এবং সংজ্ঞা ধার্য হয়েছে ধারা ৩৭৮ এ [S1]।"
    result = validator.validate(answer, assembled)

    assert result.is_valid is True
    assert len(result.hallucinated_citations) == 0
    assert len(result.valid_references) == 2
    assert result.valid_references[0].source_id == "S2"
    assert result.valid_references[0].act_name == "দণ্ডবিধি, ১৮৬০ (The Penal Code, 1860)"
    assert result.valid_references[0].section == "ধারা 379"


def test_citation_validator_hallucinated_citations():
    """Verify citing an invalid source tag [S9] is caught and flagged."""
    validator = CitationValidator()
    assembler = ContextAssembler()
    assembled = assembler.assemble(_create_sample_chunks())

    answer = "দণ্ডবিধির বিধান অনুসারে অপরাধের শাস্তি প্রদান করা হবে [S9]।"
    result = validator.validate(answer, assembled)

    assert result.is_valid is False
    assert "S9" in result.hallucinated_citations


# ---------------------------------------------------------------------------
# 4. Answer Claim Validation Tests
# ---------------------------------------------------------------------------

def test_answer_validator_grounded_sections():
    """Verify mentioned sections that exist in context pass validation."""
    validator = AnswerValidator()
    assembler = ContextAssembler()
    assembled = assembler.assemble(_create_sample_chunks())

    answer = "দণ্ডবিধির ধারা ৩৭৯ অনুসারে চুরির শাস্তি অনধিক ৩ বছর কারাদণ্ড।"
    result = validator.validate(answer, assembled)

    assert result.is_supported is True
    assert len(result.hallucinated_sections) == 0


def test_answer_validator_hallucinated_sections():
    """Verify mentioned sections absent from context are caught and flagged."""
    validator = AnswerValidator()
    assembler = ContextAssembler()
    assembled = assembler.assemble(_create_sample_chunks())

    # Section 420 (Cheating) is NOT in our sample chunks (which only have 378 and 379)
    answer = "প্রতারণার অপরাধে ধারা ৪২০ অনুযায়ী সাত বছর কারাদণ্ড দেওয়া হবে।"
    result = validator.validate(answer, assembled)

    assert result.is_supported is False
    assert "৪২০" in result.hallucinated_sections


# ---------------------------------------------------------------------------
# 5. End-to-End LegalGenerator Tests
# ---------------------------------------------------------------------------

def test_legal_generator_full_pipeline_success():
    """Verify end-to-end LegalGenerator flow producing an answered LegalAnswerResponse."""
    mock_llm = MockLLMClient(
        default_response="দণ্ডবিধি, ১৮৬০ এর ধারা ৩৭৯ অনুসারে চুরির শাস্তি ৩ বছর কারাদণ্ড [S2]।"
    )
    generator = LegalGenerator(llm_client=mock_llm)
    chunks = _create_sample_chunks()

    response = generator.generate_answer("চুরির শাস্তি কি?", chunks)

    assert isinstance(response, LegalAnswerResponse)
    assert response.status == "answered"
    assert response.is_grounded is True
    assert len(response.sources) >= 1
    assert response.sources[0].source_id == "S2"
    assert len(response.hallucinated_citations) == 0
    assert len(response.hallucinated_sections) == 0
    assert response.latency_ms > 0


def test_legal_generator_ambiguous_query_short_circuit():
    """Verify ambiguous query immediately short-circuits without calling LLM."""
    mock_llm = MockLLMClient()
    generator = LegalGenerator(llm_client=mock_llm)

    response = generator.generate_answer("আইন", _create_sample_chunks())

    assert response.status == "needs_clarification"
    assert "অত্যন্ত সংক্ষিপ্ত বা অস্পষ্ট" in response.answer
    assert len(mock_llm.calls) == 0  # LLM was never called!


def test_legal_generator_empty_evidence_short_circuit():
    """Verify empty evidence immediately short-circuits without calling LLM."""
    mock_llm = MockLLMClient()
    generator = LegalGenerator(llm_client=mock_llm)

    response = generator.generate_answer("চুরির শাস্তি কি?", [])

    assert response.status == "insufficient_evidence"
    assert "পর্যাপ্ত প্রমাণ বা বিধান পাওয়া যায়নি" in response.answer
    assert len(mock_llm.calls) == 0  # LLM was never called!
