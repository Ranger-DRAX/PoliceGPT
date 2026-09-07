"""
PoliceGPT Generation & Legal Guardrails Package.
Provides context assembly, evidence checking, Gemini LLM generation,
citation validation, and claim verification.
"""

from .llm_client import BaseLLMClient, GeminiClient, MockLLMClient, LLMResponse
from .context_builder import ContextAssembler, AssembledContext, SourceContext
from .prompt_templates import (
    BENGALI_LEGAL_SYSTEM_INSTRUCTION,
    build_legal_prompt,
    build_clarification_response,
    build_insufficient_evidence_response,
)
from .guardrails import (
    EvidenceChecker,
    CitationValidator,
    AnswerValidator,
    EvidenceCheckResult,
    CitationValidationResult,
    AnswerValidationResult,
    SourceReference,
)
from .generator import LegalGenerator, LegalAnswerResponse

__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "MockLLMClient",
    "LLMResponse",
    "ContextAssembler",
    "AssembledContext",
    "SourceContext",
    "BENGALI_LEGAL_SYSTEM_INSTRUCTION",
    "build_legal_prompt",
    "build_clarification_response",
    "build_insufficient_evidence_response",
    "EvidenceChecker",
    "CitationValidator",
    "AnswerValidator",
    "EvidenceCheckResult",
    "CitationValidationResult",
    "AnswerValidationResult",
    "SourceReference",
    "LegalGenerator",
    "LegalAnswerResponse",
]
