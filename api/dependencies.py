"""
FastAPI Dependency Providers for PoliceGPT Service.
Provides singleton access to HybridRetriever and LegalGenerator with support
for dependency overrides during testing.
"""

from fastapi import Request, HTTPException, status
from typing import Optional

try:
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from policegpt_rag.generation.generator import LegalGenerator
except ImportError:
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from src.policegpt_rag.generation.generator import LegalGenerator


def get_retriever(request: Request) -> HybridRetriever:
    """
    Retrieve singleton HybridRetriever from FastAPI application state.
    Raises HTTP 503 if the index or retriever has not been loaded.
    """
    retriever: Optional[HybridRetriever] = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hybrid Retriever is not initialized. Server may still be starting up.",
        )

    # Guard: indexes must be loaded before serving queries
    indexes_ready = (
        getattr(retriever, "_indexes_loaded", False)
        or retriever.dense_index.total_vectors > 0
        or retriever.sparse_index.total_docs > 0
    )
    if not indexes_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Search indexes are not loaded. "
                "Run `python scripts/build_index.py` to compile FAISS & sparse indexes "
                "from embeddings before starting the API server."
            ),
        )
    return retriever


def get_generator(request: Request) -> LegalGenerator:
    """
    Retrieve singleton LegalGenerator from FastAPI application state.
    Raises HTTP 503 if the generation service is not initialized.
    """
    generator: Optional[LegalGenerator] = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Legal Generator is not initialized.",
        )
    return generator
