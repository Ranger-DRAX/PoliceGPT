"""
Context Assembly and Source Tagging for PoliceGPT Legal Generation.
Deduplicates statutory candidate chunks, enforces context budget, and assigns
deterministic source identifiers ([S1], [S2], ...) with full legal metadata.
"""

from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field
from loguru import logger
import unicodedata


class SourceContext(BaseModel):
    """Encapsulates a single verified statutory source with metadata and source ID."""
    source_id: str  # e.g., "S1", "S2"
    chunk_id: str
    doc_id: str
    act_name_en: Optional[str] = None
    act_name_bn: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    content: str
    rrf_score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AssembledContext(BaseModel):
    """Result of context assembly containing source items and formatted prompt text."""
    sources: List[SourceContext]
    formatted_context: str
    total_chars: int
    num_deduplicated: int


class ContextAssembler:
    """
    Context assembly pipeline:
      1. Deduplication (exact chunk_id + high-overlap content).
      2. Source tag assignment ([S1], [S2], ...).
      3. Token / character budget enforcement.
      4. Structured statutory markdown rendering.
    """

    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max(1000, max_context_chars)

    def _normalize_text(self, text: str) -> str:
        """Light normalization for duplicate detection."""
        return "".join(unicodedata.normalize("NFC", text).split()).lower()

    def assemble(
        self,
        chunks: List[Any],
        max_chars: Optional[int] = None,
    ) -> AssembledContext:
        """
        Assemble retrieved chunks into a budget-enforced, deduplicated context string.
        Accepts Pydantic RetrievedChunk or dict instances.
        """
        budget = max_chars or self.max_context_chars
        if not chunks:
            return AssembledContext(
                sources=[],
                formatted_context="",
                total_chars=0,
                num_deduplicated=0,
            )

        seen_ids: Set[str] = set()
        seen_texts: List[str] = []
        selected_sources: List[SourceContext] = []
        num_dedup = 0
        current_chars = 0

        source_index = 1
        for item in chunks:
            # Handle both Pydantic model and dict
            if hasattr(item, "model_dump"):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            else:
                data = getattr(item, "__dict__", {})

            cid = data.get("chunk_id", f"chunk_{source_index}")
            content = (data.get("content") or "").strip()

            if not content:
                continue

            # Check exact chunk ID duplication
            if cid in seen_ids:
                num_dedup += 1
                continue

            # Check text overlap duplication (exact or >90% prefix overlap)
            norm_content = self._normalize_text(content)
            is_dup = False
            for prev_norm in seen_texts:
                if norm_content in prev_norm or prev_norm in norm_content:
                    is_dup = True
                    break
            if is_dup:
                num_dedup += 1
                continue

            # Check budget
            content_len = len(content)
            if current_chars + content_len > budget and selected_sources:
                logger.debug(
                    f"Context budget reached ({current_chars} + {content_len} > {budget}). "
                    f"Stopping context assembly at {len(selected_sources)} sources."
                )
                break

            source_id = f"S{source_index}"
            source_ctx = SourceContext(
                source_id=source_id,
                chunk_id=cid,
                doc_id=data.get("doc_id", ""),
                act_name_en=data.get("act_name_en"),
                act_name_bn=data.get("act_name_bn"),
                section_number=data.get("section_number"),
                section_title=data.get("section_title"),
                page_numbers=data.get("page_numbers") or [],
                content=content,
                rrf_score=data.get("rrf_score", 0.0),
                dense_score=data.get("dense_score"),
                sparse_score=data.get("sparse_score"),
                rerank_score=data.get("rerank_score"),
            )

            selected_sources.append(source_ctx)
            seen_ids.add(cid)
            seen_texts.append(norm_content)
            current_chars += content_len
            source_index += 1

        # Format context markdown for LLM
        formatted_blocks = []
        for src in selected_sources:
            act_label = src.act_name_bn or src.act_name_en or src.doc_id
            if src.act_name_en and src.act_name_bn and src.act_name_en != src.act_name_bn:
                act_label = f"{src.act_name_bn} ({src.act_name_en})"

            sec_label = f"ধারা {src.section_number}" if src.section_number else "সাধারণ বিধান"
            if src.section_title:
                sec_label += f" — {src.section_title}"

            pages_str = f"পৃষ্ঠা: {', '.join(map(str, src.page_numbers))}" if src.page_numbers else ""

            block = (
                f"[উৎস {src.source_id}]\n"
                f"আইন: {act_label}\n"
                f"ধারা: {sec_label}\n"
            )
            if pages_str:
                block += f"{pages_str}\n"
            block += f"মূল আইনি বিধান:\n{src.content}\n"
            formatted_blocks.append(block)

        formatted_context = "\n---\n".join(formatted_blocks)

        return AssembledContext(
            sources=selected_sources,
            formatted_context=formatted_context,
            total_chars=len(formatted_context),
            num_deduplicated=num_dedup,
        )
