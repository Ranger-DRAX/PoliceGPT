"""
Legal Guardrails for PoliceGPT Generation Pipeline.
Implements:
  1. Evidence Checking (Pre-flight relevance & ambiguity gating)
  2. Citation Validation (Source tag verification & metadata resolution)
  3. Answer Validation (Claim support & statutory hallucination prevention)
"""

from typing import List, Dict, Set, Any, Optional, Tuple, Literal
import re
from pydantic import BaseModel, Field
from loguru import logger

try:
    from policegpt_rag.generation.context_builder import AssembledContext, SourceContext
except ImportError:
    from src.policegpt_rag.generation.context_builder import AssembledContext, SourceContext


class EvidenceCheckResult(BaseModel):
    """Outcome of pre-generation evidence evaluation."""
    status: Literal["answered", "needs_clarification", "insufficient_evidence"]
    reason: str
    can_generate: bool


class SourceReference(BaseModel):
    """Detailed human-readable citation resolved from source metadata."""
    source_id: str
    act_name: str
    section: Optional[str] = None
    section_title: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    snippet: str


class CitationValidationResult(BaseModel):
    """Outcome of citation verification."""
    is_valid: bool
    cited_source_ids: List[str]
    valid_references: List[SourceReference]
    hallucinated_citations: List[str]


class AnswerValidationResult(BaseModel):
    """Outcome of claim support and hallucination check."""
    is_supported: bool
    hallucinated_sections: List[str]
    claim_support_score: float
    notes: List[str] = Field(default_factory=list)


class EvidenceChecker:
    """
    Pre-flight guardrail checking whether retrieved evidence is sufficient
    or if user query requires clarification.
    """

    AMBIGUOUS_TERMS = {
        "আইন", "ধারা", "শাস্তি", "পুলিশ", "পুলিশের", "গ্রেপ্তার", "মামলা",
        "law", "act", "police", "arrest", "punishment", "section", "crime"
    }

    def __init__(
        self,
        min_chunks: int = 1,
        min_rrf_score: float = 0.012,
        min_dense_score: float = 0.30,
    ):
        self.min_chunks = min_chunks
        self.min_rrf_score = min_rrf_score
        self.min_dense_score = min_dense_score

    def check(self, query: str, chunks: List[Any]) -> EvidenceCheckResult:
        """Evaluate query clarity and candidate relevance."""
        clean_q = query.strip()

        # 1. Ambiguity / brevity check
        tokens = [t.lower() for t in clean_q.split() if len(t) > 1]
        if len(tokens) <= 1 and (not tokens or tokens[0] in self.AMBIGUOUS_TERMS):
            return EvidenceCheckResult(
                status="needs_clarification",
                reason=f"প্রশ্নটি অত্যন্ত অস্পষ্ট বা সংক্ষিপ্ত ('{clean_q}')।",
                can_generate=False,
            )

        # 2. Retrieved chunk existence check
        if not chunks or len(chunks) < self.min_chunks:
            return EvidenceCheckResult(
                status="insufficient_evidence",
                reason="কোন প্রাসঙ্গিক আইনি নথি বা ধারা পাওয়া যায়নি।",
                can_generate=False,
            )

        # 3. Minimum score thresholding
        top_chunk = chunks[0]
        rrf_score = getattr(top_chunk, "rrf_score", None)
        if rrf_score is None and isinstance(top_chunk, dict):
            rrf_score = top_chunk.get("rrf_score", 0.0)

        dense_score = getattr(top_chunk, "dense_score", None)
        if dense_score is None and isinstance(top_chunk, dict):
            dense_score = top_chunk.get("dense_score")

        # If both scores are present and below minimum floors
        if rrf_score is not None and rrf_score < self.min_rrf_score:
            if dense_score is not None and dense_score < self.min_dense_score:
                return EvidenceCheckResult(
                    status="insufficient_evidence",
                    reason=(
                        f"আইনি তথ্যের প্রাসঙ্গিকতা স্কোর অত্যন্ত কম "
                        f"(RRF: {rrf_score:.5f}, Dense: {dense_score:.4f})।"
                    ),
                    can_generate=False,
                )

        return EvidenceCheckResult(
            status="answered",
            reason="পর্যাপ্ত প্রাসঙ্গিক প্রমাণ পাওয়া গেছে।",
            can_generate=True,
        )


class CitationValidator:
    """
    Post-generation guardrail validating that every cited source tag ([S1], [S2])
    exists in the assembled context and rendering rich statutory metadata.
    """

    CITATION_PATTERN = re.compile(r"\[(?:উৎস\s*)?S?(\d+)\]", re.IGNORECASE)

    def validate(
        self,
        generated_text: str,
        assembled_context: AssembledContext,
    ) -> CitationValidationResult:
        """Extract citations, verify against sources, and construct SourceReference objects."""
        # Find all cited IDs (e.g. S1, S2)
        raw_matches = self.CITATION_PATTERN.findall(generated_text)
        cited_ids = [f"S{m}" for m in raw_matches]

        source_map: Dict[str, SourceContext] = {
            src.source_id: src for src in assembled_context.sources
        }

        valid_refs: List[SourceReference] = []
        hallucinated: List[str] = []
        seen_cited: Set[str] = set()

        for cid in cited_ids:
            if cid in seen_cited:
                continue
            seen_cited.add(cid)

            if cid in source_map:
                src = source_map[cid]
                act_label = src.act_name_bn or src.act_name_en or src.doc_id
                sec_label = f"ধারা {src.section_number}" if src.section_number else None
                snippet = src.content[:200] + "..." if len(src.content) > 200 else src.content

                valid_refs.append(
                    SourceReference(
                        source_id=cid,
                        act_name=act_label,
                        section=sec_label,
                        section_title=src.section_title,
                        page_numbers=src.page_numbers,
                        snippet=snippet,
                    )
                )
            else:
                hallucinated.append(cid)

        is_valid = len(hallucinated) == 0

        # If LLM didn't cite anything but sources were available, include all sources in references
        if not valid_refs and assembled_context.sources:
            for src in assembled_context.sources:
                act_label = src.act_name_bn or src.act_name_en or src.doc_id
                sec_label = f"ধারা {src.section_number}" if src.section_number else None
                valid_refs.append(
                    SourceReference(
                        source_id=src.source_id,
                        act_name=act_label,
                        section=sec_label,
                        section_title=src.section_title,
                        page_numbers=src.page_numbers,
                        snippet=src.content[:200] + "..." if len(src.content) > 200 else src.content,
                    )
                )

        return CitationValidationResult(
            is_valid=is_valid,
            cited_source_ids=cited_ids,
            valid_references=valid_refs,
            hallucinated_citations=hallucinated,
        )


class AnswerValidator:
    """
    Validates claim support and detects hallucinated section numbers.
    A valid source ID alone does not prove the answer is supported.
    """

    # Matches Bengali and English section mentions: e.g. "ধারা ৩৭৯", "Section 54", "ধারা 302"
    SECTION_REGEX = re.compile(r"(?:ধারা|section|sec\.?)\s*([০-৯\d]+[A-Za-z]*)", re.IGNORECASE)

    BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

    def _normalize_section_num(self, sec: str) -> str:
        """Normalize Bengali digits to English digits for comparison."""
        return sec.translate(self.BENGALI_DIGITS).strip().lower()

    def validate(
        self,
        answer_text: str,
        assembled_context: AssembledContext,
    ) -> AnswerValidationResult:
        """
        Verify that all mentioned section numbers exist in context metadata/content,
        and calculate lexical support overlap.
        """
        notes: List[str] = []

        # 1. Collect all valid section numbers from the assembled context
        valid_sections: Set[str] = set()
        context_corpus: List[str] = []
        for src in assembled_context.sources:
            context_corpus.append(src.content.lower())
            if src.section_number:
                valid_sections.add(self._normalize_section_num(src.section_number))

        full_context_text = " ".join(context_corpus)

        # 2. Extract sections mentioned in the generated answer
        mentioned_raw = self.SECTION_REGEX.findall(answer_text)
        hallucinated_sections: List[str] = []

        for raw_sec in mentioned_raw:
            norm_sec = self._normalize_section_num(raw_sec)
            # Check if section number is present in valid sections OR appears directly in context text
            if norm_sec not in valid_sections and norm_sec not in full_context_text:
                hallucinated_sections.append(raw_sec)

        # 3. Compute simple lexical claim support overlap
        answer_words = [w.strip(".,;:?!'\"()[]") for w in answer_text.split() if len(w) > 2]
        if not answer_words:
            support_score = 0.0
        else:
            supported_words = sum(1 for w in answer_words if w.lower() in full_context_text)
            support_score = round(supported_words / len(answer_words), 3)

        is_supported = len(hallucinated_sections) == 0 and (support_score >= 0.25 or len(answer_words) < 5)

        if hallucinated_sections:
            notes.append(f"উত্তরটিতে অপ্রমাণিত ধারা উল্লেখ রয়েছে: {', '.join(hallucinated_sections)}")
        if support_score < 0.25:
            notes.append(f"আইনি বিষয়ের সাথে শাব্দিক মিল কম (Support Score: {support_score})")

        return AnswerValidationResult(
            is_supported=is_supported,
            hallucinated_sections=hallucinated_sections,
            claim_support_score=support_score,
            notes=notes,
        )
