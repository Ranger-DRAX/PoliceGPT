"""
Legal Query and Statutory Search Routers for PoliceGPT API.
Exposes endpoints for full legal RAG generation, statutory chunk search,
and specific chunk metadata inspection.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
import time
from loguru import logger

from api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceReferenceSchema,
    SearchRequest,
    SearchResponse,
    RetrievedChunkItem,
    ChunkDetailResponse,
)
from api.dependencies import get_retriever, get_generator

try:
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
    from policegpt_rag.generation.generator import LegalGenerator, LegalAnswerResponse
except ImportError:
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
    from src.policegpt_rag.generation.generator import LegalGenerator, LegalAnswerResponse


router = APIRouter(prefix="/api/v1", tags=["Legal Services"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Legal Question Answering (Full RAG Generation + Guardrails)",
    description=(
        "Executes the full PoliceGPT RAG pipeline: retrieves relevant statutory provisions "
        "using Dense FAISS + Sparse Lexical RRF search, performs pre-flight evidence checking, "
        "generates a grounded Bengali answer using Google Gemini 3.5 Flash, and validates citations & claims."
    ),
)
async def legal_query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    generator: LegalGenerator = Depends(get_generator),
) -> QueryResponse:
    """Execute end-to-end legal question answering with generation and citation guardrails."""
    t_start = time.perf_counter()

    # Configure reranker toggle per request
    original_rerank_flag = retriever.use_reranker
    retriever.use_reranker = request.use_reranker

    try:
        # Step 1: Retrieve statutory candidates
        retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)

        # Step 2: Generate grounded response through LegalGenerator orchestrator
        gen_response: LegalAnswerResponse = generator.generate_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
        )

    finally:
        retriever.use_reranker = original_rerank_flag

    t_total = (time.perf_counter() - t_start) * 1000.0

    # Map sources to schema
    source_schemas = [
        SourceReferenceSchema(
            source_id=s.source_id,
            act_name=s.act_name,
            section=s.section,
            section_title=s.section_title,
            page_numbers=s.page_numbers,
            snippet=s.snippet,
        )
        for s in gen_response.sources
    ]

    return QueryResponse(
        query=gen_response.query,
        status=gen_response.status,
        answer=gen_response.answer,
        sources=source_schemas,
        is_grounded=gen_response.is_grounded,
        claim_support_score=gen_response.claim_support_score,
        hallucinated_citations=gen_response.hallucinated_citations,
        hallucinated_sections=gen_response.hallucinated_sections,
        clarification_prompt=gen_response.clarification_prompt,
        latency_ms=round(t_total, 2),
        model_name=gen_response.model_name,
        validation_notes=gen_response.validation_notes,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Statutory Retrieval Search (Dense + Sparse + RRF)",
    description=(
        "Direct statutory search across Bangladesh legal codes without LLM generation. "
        "Returns ranked candidate chunks combining dense cosine and sparse lexical matches."
    ),
)
async def statutory_search(
    request: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> SearchResponse:
    """Execute hybrid statutory search returning ranked legal chunk candidates."""
    t_start = time.perf_counter()

    original_rerank_flag = retriever.use_reranker
    retriever.use_reranker = request.use_reranker

    try:
        results: List[RetrievedChunk] = retriever.retrieve(request.query, top_k=request.top_k)
    finally:
        retriever.use_reranker = original_rerank_flag

    t_total = (time.perf_counter() - t_start) * 1000.0

    chunk_items = [
        RetrievedChunkItem(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            act_name_en=r.act_name_en,
            act_name_bn=r.act_name_bn,
            section_number=r.section_number,
            section_title=r.section_title,
            content=r.content,
            page_numbers=r.page_numbers,
            rrf_score=round(r.rrf_score, 6),
            dense_score=round(r.dense_score, 4) if r.dense_score is not None else None,
            dense_rank=r.dense_rank,
            sparse_score=round(r.sparse_score, 4) if r.sparse_score is not None else None,
            sparse_rank=r.sparse_rank,
            rerank_score=round(r.rerank_score, 4) if r.rerank_score is not None else None,
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        total_results=len(chunk_items),
        results=chunk_items,
        latency_ms=round(t_total, 2),
        profile=retriever.last_profile,
    )


@router.get(
    "/chunks/{chunk_id}",
    response_model=ChunkDetailResponse,
    summary="Statutory Chunk Metadata Lookup",
    description="Retrieve the full statutory text and legal metadata for a specific chunk ID.",
)
async def get_chunk_detail(
    chunk_id: str,
    retriever: HybridRetriever = Depends(get_retriever),
) -> ChunkDetailResponse:
    """Inspect full metadata and text of a specific chunk by its chunk_id."""
    idx = retriever.dense_index._chunk_id_to_idx.get(chunk_id)
    if idx is None or idx >= len(retriever.dense_index.metadata_registry):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk '{chunk_id}' not found in legal index registry.",
        )

    meta = retriever.dense_index.metadata_registry[idx]
    return ChunkDetailResponse(
        chunk_id=meta.get("chunk_id", chunk_id),
        doc_id=meta.get("doc_id", ""),
        act_name_en=meta.get("act_name_en"),
        act_name_bn=meta.get("act_name_bn"),
        section_number=meta.get("section_number"),
        section_title=meta.get("section_title"),
        content=meta.get("content", ""),
        page_numbers=meta.get("page_numbers", []),
    )
