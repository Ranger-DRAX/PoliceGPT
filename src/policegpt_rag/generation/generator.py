"""
End-to-End Legal Generation Orchestrator for PoliceGPT.
Coordinates Context Assembly, Pre-flight Evidence Gating, Replaceable LLM Inference,
Citation Validation, and Claim Verification.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import time
import yaml
from pydantic import BaseModel, Field
from loguru import logger

try:
    from policegpt_rag.generation.context_builder import ContextAssembler, AssembledContext
    from policegpt_rag.generation.llm_client import BaseLLMClient, GeminiClient, MockLLMClient, LLMResponse
    from policegpt_rag.generation.prompt_templates import (
        BENGALI_LEGAL_SYSTEM_INSTRUCTION,
        build_legal_prompt,
        build_clarification_response,
        build_insufficient_evidence_response,
    )
    from policegpt_rag.generation.guardrails import (
        EvidenceChecker,
        CitationValidator,
        AnswerValidator,
        SourceReference,
    )
except ImportError:
    from src.policegpt_rag.generation.context_builder import ContextAssembler, AssembledContext
    from src.policegpt_rag.generation.llm_client import BaseLLMClient, GeminiClient, MockLLMClient, LLMResponse
    from src.policegpt_rag.generation.prompt_templates import (
        BENGALI_LEGAL_SYSTEM_INSTRUCTION,
        build_legal_prompt,
        build_clarification_response,
        build_insufficient_evidence_response,
    )
    from src.policegpt_rag.generation.guardrails import (
        EvidenceChecker,
        CitationValidator,
        AnswerValidator,
        SourceReference,
    )


class LegalAnswerResponse(BaseModel):
    """Rich, validated legal answer response returned to the caller or API layer."""
    query: str
    status: Literal["answered", "needs_clarification", "insufficient_evidence"]
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    hallucinated_citations: List[str] = Field(default_factory=list)
    hallucinated_sections: List[str] = Field(default_factory=list)
    is_grounded: bool = True
    claim_support_score: float = 1.0
    clarification_prompt: Optional[str] = None
    latency_ms: float = 0.0
    model_name: str = "gemini-3.5-flash"
    validation_notes: List[str] = Field(default_factory=list)


class LegalGenerator:
    """
    Production RAG Generation Orchestrator:
      1. Context Assembly (Deduplication & [S1], [S2] source ID assignment)
      2. Evidence Check (Relevance floor & ambiguity check)
      3. Prompt Building (Bengali statutory templates)
      4. LLM Call (Gemini 3.5 Flash client with retry backoff)
      5. Citation Validation (Verifying source IDs against context)
      6. Answer Validation (Claim support & hallucinated section checks)
      7. Packaging structured API response
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        config_path: Optional[Union[str, Path]] = None,
        model_name: str = "gemini-3.5-flash",
        max_context_chars: int = 12000,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_context_chars = max_context_chars

        if config_path:
            self._load_config(config_path)

        self.context_assembler = ContextAssembler(max_context_chars=self.max_context_chars)
        self.evidence_checker = EvidenceChecker()
        self.citation_validator = CitationValidator()
        self.answer_validator = AnswerValidator()

        # LLM client setup
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # Default to GeminiClient if API key present, otherwise MockLLMClient
            try:
                self.llm_client = GeminiClient(model_name=self.model_name)
            except ValueError:
                logger.warning("No Gemini API key detected. Initializing with MockLLMClient for offline use.")
                self.llm_client = MockLLMClient(model_name=self.model_name)

    def _load_config(self, config_path: Union[str, Path]):
        path = Path(config_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gen_cfg = cfg.get("generation", {})
            self.model_name = gen_cfg.get("model_name", self.model_name)
            self.temperature = gen_cfg.get("temperature", self.temperature)
            self.max_output_tokens = gen_cfg.get("max_output_tokens", self.max_output_tokens)
            self.max_context_chars = gen_cfg.get("context_budget_chars", self.max_context_chars)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Any],
    ) -> LegalAnswerResponse:
        """
        Execute full generation and guardrail validation pipeline.
        Returns a structured LegalAnswerResponse.
        """
        t_start = time.perf_counter()

        # Step 1: Pre-flight Evidence Check
        evidence_eval = self.evidence_checker.check(query, retrieved_chunks)

        if not evidence_eval.can_generate:
            t_elapsed = (time.perf_counter() - t_start) * 1000.0

            if evidence_eval.status == "needs_clarification":
                clarify_msg = build_clarification_response(query)
                return LegalAnswerResponse(
                    query=query,
                    status="needs_clarification",
                    answer=clarify_msg,
                    clarification_prompt=clarify_msg,
                    latency_ms=round(t_elapsed, 2),
                    model_name=getattr(self.llm_client, "model_name", "pre-flight-gate"),
                    validation_notes=[evidence_eval.reason],
                )
            else:  # insufficient_evidence
                refusal_msg = build_insufficient_evidence_response(query)
                return LegalAnswerResponse(
                    query=query,
                    status="insufficient_evidence",
                    answer=refusal_msg,
                    latency_ms=round(t_elapsed, 2),
                    model_name=getattr(self.llm_client, "model_name", "pre-flight-gate"),
                    validation_notes=[evidence_eval.reason],
                )

        # Step 2: Context Assembly (Deduplication, Source Tagging, Budget Packing)
        assembled_ctx = self.context_assembler.assemble(
            retrieved_chunks,
            max_chars=self.max_context_chars,
        )

        # Step 3: Prompt Building
        prompt = build_legal_prompt(query, assembled_ctx.formatted_context)

        # Step 4: LLM Generation
        llm_resp: LLMResponse = self.llm_client.generate(
            prompt=prompt,
            system_instruction=BENGALI_LEGAL_SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        # Step 5: Citation Validation
        citation_eval = self.citation_validator.validate(llm_resp.text, assembled_ctx)

        # Step 6: Answer Claim Validation
        answer_eval = self.answer_validator.validate(llm_resp.text, assembled_ctx)

        t_total = (time.perf_counter() - t_start) * 1000.0
        all_notes = list(answer_eval.notes)
        if citation_eval.hallucinated_citations:
            all_notes.append(f"অপ্রমাণিত সূত্রের উদ্ধৃতি: {', '.join(citation_eval.hallucinated_citations)}")

        is_grounded = citation_eval.is_valid and answer_eval.is_supported

        return LegalAnswerResponse(
            query=query,
            status="answered",
            answer=llm_resp.text,
            sources=citation_eval.valid_references,
            hallucinated_citations=citation_eval.hallucinated_citations,
            hallucinated_sections=answer_eval.hallucinated_sections,
            is_grounded=is_grounded,
            claim_support_score=answer_eval.claim_support_score,
            latency_ms=round(t_total, 2),
            model_name=llm_resp.model_name,
            validation_notes=all_notes,
        )
