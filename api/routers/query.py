"""
Query router handling /query and /retrieve endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
import os

from ..schemas import QueryRequest, QueryResponse, RetrievedSource
from policegpt_rag.pipeline import PoliceGPTRAGPipeline

router = APIRouter(tags=["Legal Query"])

# Lazy singleton pipeline instance
_pipeline: PoliceGPTRAGPipeline | None = None


def get_pipeline() -> PoliceGPTRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PoliceGPTRAGPipeline(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", 6333)),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "policegpt_legal_corpus"),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model=os.getenv("LLM_MODEL_NAME", "qwen2.5:7b"),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        )
    return _pipeline


@router.post("/query", response_model=QueryResponse)
async def query_legal_corpus(
    request: QueryRequest, pipeline: PoliceGPTRAGPipeline = Depends(get_pipeline)
):
    try:
        filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else None

        result = pipeline.query(
            query_text=request.query,
            language=request.language,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
            filters=filter_dict,
        )

        sources = [
            RetrievedSource(
                chunk_id=s.get("chunk_id", ""),
                act_name_bn=s.get("act_name_bn"),
                act_name_en=s.get("act_name_en"),
                section_number=s.get("section_number"),
                section_title=s.get("section_title"),
                content=s.get("content", ""),
                score=s.get("rerank_score", s.get("score")),
                page_numbers=s.get("page_numbers", []),
            )
            for s in result.retrieved_sources
        ]

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            citations=result.citations,
            retrieved_sources=sources,
            confidence_score=result.confidence_score,
            is_grounded=result.is_grounded,
            status="success",
        )
    except Exception as e:
        logger.error(f"Error handling /query request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
