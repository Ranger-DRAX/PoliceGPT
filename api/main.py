"""
PoliceGPT Legal RAG — FastAPI Application Server
=================================================
Provides high-performance RESTful APIs for Bengali/English statutory question answering,
hybrid legal retrieval, and subsystem diagnostics.

Features:
  - Lifespan state management for fast in-memory query execution.
  - CORS middleware for Web and mobile client integrations.
  - Granular timing telemetry headers (X-Process-Time-Ms).
  - OpenAPI Swagger documentation at /docs and ReDoc at /redoc.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
import os

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers.query import router as query_router
from api.routers.health import router as health_router

try:
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from policegpt_rag.generation.generator import LegalGenerator
except ImportError:
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from src.policegpt_rag.generation.generator import LegalGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
      - Startup: initialize and load FAISS vector index, sparse lexical index,
        and the Gemini legal generator.
      - Shutdown: clean up GPU cache and unload models if necessary.
    """
    logger.info("Initializing PoliceGPT FastAPI Service...")
    index_dir = REPO_ROOT / "data" / "processed" / "indexes"
    retrieval_cfg = REPO_ROOT / "configs" / "retrieval.yaml"
    generation_cfg = REPO_ROOT / "configs" / "generation.yaml"

    # 1. Initialize Hybrid Retriever
    retriever = HybridRetriever(config_path=retrieval_cfg if retrieval_cfg.is_file() else None)
    if (index_dir / "dense_index.faiss").is_file():
        try:
            retriever.load_indexes(index_dir)
            logger.info(
                f"Successfully loaded hybrid indexes: {retriever.dense_index.total_vectors} chunks, "
                f"{len(retriever.sparse_index.inverted_index)} sparse lexical terms."
            )
        except Exception as e:
            logger.error(f"Failed to load indexes from {index_dir}: {e}")
    else:
        logger.warning(
            f"Index directory not found at {index_dir}. "
            f"Retriever started in uninitialized mode. Run `python scripts/build_index.py`."
        )

    # 2. Initialize Legal Generator
    generator = LegalGenerator(config_path=generation_cfg if generation_cfg.is_file() else None)
    logger.info(f"Legal Generator ready (Model: {getattr(generator.llm_client, 'model_name', 'default')}).")

    app.state.retriever = retriever
    app.state.generator = generator

    yield

    # Teardown logic
    logger.info("Shutting down PoliceGPT FastAPI Service...")
    if hasattr(app.state, "retriever") and app.state.retriever.reranker:
        app.state.retriever.reranker.unload()


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app = FastAPI(
        title="PoliceGPT Legal RAG API",
        description=(
            "Statutory Legal Question Answering and Retrieval API for Bangladesh Law Enforcement. "
            "Combines Dense FAISS (BGE-M3) + Sparse Lexical Inverted Index (RRF Fusion), "
            "Google Gemini (gemini-2.5-flash), and statutory citation guardrails."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. CORS Middleware (Allow web frontends)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Latency Measurement Middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # 3. Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    # 4. Include Routers
    app.include_router(query_router)
    app.include_router(health_router)

    # 5. Root Info Endpoint
    @app.get("/", tags=["System"])
    async def root_info():
        return {
            "service": "PoliceGPT Legal RAG API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "query_endpoint": "/api/v1/query",
            "search_endpoint": "/api/v1/search",
        }

    return app


app = create_app()
