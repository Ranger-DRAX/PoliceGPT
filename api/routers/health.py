"""
Health Check and Subsystem Diagnostics Router for PoliceGPT API.
"""

from fastapi import APIRouter, Request
from api.schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Subsystem Health and Status Check",
    description="Inspect readiness of vector index, sparse lexical index, reranker VRAM status, and generation model.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of all PoliceGPT components."""
    retriever = getattr(request.app.state, "retriever", None)
    generator = getattr(request.app.state, "generator", None)

    indexes_loaded = False
    dense_vecs = 0
    sparse_terms = 0
    reranker_status = {}
    gen_model = "unknown"

    if retriever is not None:
        indexes_loaded = bool(
            getattr(retriever, "_indexes_loaded", False)
            or retriever.dense_index.total_vectors > 0
        )
        dense_vecs = retriever.dense_index.total_vectors
        sparse_terms = len(retriever.sparse_index.inverted_index)
        if retriever.reranker:
            reranker_status = retriever.reranker.get_status()

    if generator is not None and hasattr(generator, "llm_client"):
        gen_model = getattr(generator.llm_client, "model_name", "unknown")

    status_str = "healthy" if indexes_loaded else "degraded"

    return HealthResponse(
        status=status_str,
        service="PoliceGPT Legal RAG API",
        version="1.0.0",
        indexes_loaded=indexes_loaded,
        total_dense_vectors=dense_vecs,
        total_sparse_terms=sparse_terms,
        reranker_status=reranker_status,
        generation_model=gen_model,
    )
