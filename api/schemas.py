"""
Pydantic Request and Response Schemas for PoliceGPT FastAPI Service.
Defines strict API contracts for legal querying, statutory search, and health checks.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Legal Query (RAG Generation) Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for full legal question answering with generation and guardrails."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Legal question or query in Bengali or English.",
        examples=["চুরির শাস্তি কি?", "power of police to arrest without warrant"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of statutory chunks to retrieve and consider.",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to run the Cross-Encoder reranker (auto-gated by VRAM/CUDA).",
    )


class SourceReferenceSchema(BaseModel):
    """Citation metadata for a referenced statutory excerpt."""
    source_id: str = Field(..., description="Unique source tag, e.g. 'S1', 'S2'")
    act_name: str = Field(..., description="Full legal act name in Bengali/English")
    section: Optional[str] = Field(None, description="Specific statutory section, e.g. 'ধারা ৩৭৯'")
    section_title: Optional[str] = Field(None, description="Title or heading of the section")
    page_numbers: List[int] = Field(default_factory=list, description="Original PDF page numbers")
    snippet: str = Field(..., description="Excerpt of the statutory text")


class QueryResponse(BaseModel):
    """Structured legal answer response returned to API consumers."""
    query: str
    status: Literal["answered", "needs_clarification", "insufficient_evidence"]
    answer: str
    sources: List[SourceReferenceSchema] = Field(default_factory=list)
    is_grounded: bool = Field(..., description="True if citations and claims pass all legal guardrails")
    claim_support_score: float = Field(..., description="Lexical support ratio between context and claims")
    hallucinated_citations: List[str] = Field(default_factory=list, description="Invalid citation tags caught by guardrails")
    hallucinated_sections: List[str] = Field(default_factory=list, description="Mentioned sections unsupported by context")
    clarification_prompt: Optional[str] = None
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")
    model_name: str
    validation_notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Statutory Search (Retrieval-Only) Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Payload for statutory retrieval search without LLM generation."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Legal query string for statutory search.",
        examples=["দণ্ডবিধি ৩৭৯", "unlawful assembly punishment"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top statutory chunks to return.",
    )
    use_reranker: bool = Field(
        default=True,
        description="Apply Cross-Encoder reranking to top candidates.",
    )


class RetrievedChunkItem(BaseModel):
    """Individual statutory chunk returned by hybrid search."""
    chunk_id: str
    doc_id: str
    act_name_en: Optional[str] = None
    act_name_bn: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    content: str
    page_numbers: List[int] = Field(default_factory=list)
    rrf_score: float
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Response container for pure statutory retrieval search."""
    query: str
    total_results: int
    results: List[RetrievedChunkItem]
    latency_ms: float
    profile: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. Chunk Inspection & Health Schemas
# ---------------------------------------------------------------------------

class ChunkDetailResponse(BaseModel):
    """Complete metadata and content for a single stored statutory chunk."""
    chunk_id: str
    doc_id: str
    act_name_en: Optional[str] = None
    act_name_bn: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    content: str
    page_numbers: List[int] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health and diagnostic status."""
    status: str = "healthy"
    service: str = "PoliceGPT Legal RAG API"
    version: str = "1.0.0"
    indexes_loaded: bool
    total_dense_vectors: int
    total_sparse_terms: int
    reranker_status: Dict[str, Any]
    generation_model: str
