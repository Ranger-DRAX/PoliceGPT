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
            detail="Hybrid Retriever is not initialized or indexes are not loaded.",
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
