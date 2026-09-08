"""
PoliceGPT Legal RAG — FastAPI Application Server
=================================================
Provides high-performance RESTful APIs for Bengali/English statutory question answering,
hybrid legal retrieval, and subsystem diagnostics.

Features:
  - Lifespan state management for fast in-memory query execution.
  - Auto-builds FAISS + Sparse indexes from Parquet embeddings on first start.
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

# Load .env early so all modules see env vars
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from api.routers.query import router as query_router
from api.routers.health import router as health_router

try:
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from policegpt_rag.generation.generator import LegalGenerator
    from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from policegpt_rag.indexing.sparse_index import SparseLexicalIndex
except ImportError:
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever
    from src.policegpt_rag.generation.generator import LegalGenerator
    from src.policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from src.policegpt_rag.indexing.sparse_index import SparseLexicalIndex

REPO_ROOT = Path(__file__).resolve().parent.parent


def _auto_build_indexes(index_dir: Path, embeddings_dir: Path, chunks_dir: Path) -> bool:
    """
    Compile FAISS dense index + Sparse lexical index from Parquet embeddings.
    Called automatically on first startup if index files are missing.
    Returns True if indexes were built successfully, False otherwise.
    """
    parquet_files = sorted(list(embeddings_dir.glob("*_embeddings.parquet")))
    if not parquet_files:
        logger.error(
            f"Cannot auto-build indexes: no Parquet embedding files in {embeddings_dir}. "
            f"Run `python scripts/build_embeddings.py --all` first."
        )
        return False

    logger.info(f"Auto-building indexes from {len(parquet_files)} Parquet file(s)...")
    try:
        # Build FAISS dense index
        dense_index = FAISSVectorIndex(dimension=1024)
        dense_index.build_from_parquet_files(parquet_files, chunks_json_dir=chunks_dir)
        dense_index.save(index_dir)
        logger.info(f"FAISS IndexFlatIP built: {dense_index.total_vectors} vectors (1024-d)")

        # Build Sparse lexical inverted index
        sparse_index = SparseLexicalIndex()
        sparse_index.build_from_parquet_files(parquet_files)
        sparse_index.save(index_dir)
        logger.info(f"Sparse lexical index built: {len(sparse_index.inverted_index)} terms, {sparse_index.total_docs} docs")

        return True
    except Exception as e:
        logger.error(f"Auto-build of indexes failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
      - Startup: load (or auto-build) FAISS + Sparse indexes, then init generator.
      - Shutdown: clean up GPU cache and unload models if necessary.
    """
    logger.info("Initializing PoliceGPT FastAPI Service...")
    index_dir = REPO_ROOT / "data" / "processed" / "indexes"
    embeddings_dir = REPO_ROOT / "data" / "processed" / "embeddings"
    chunks_dir = REPO_ROOT / "data" / "processed"
    retrieval_cfg = REPO_ROOT / "configs" / "retrieval.yaml"
    generation_cfg = REPO_ROOT / "configs" / "generation.yaml"

    # 1. Auto-build indexes from Parquet if they don't exist yet
    if not (index_dir / "dense_index.faiss").is_file():
        logger.warning("Index files not found. Attempting auto-build from Parquet embeddings...")
        _auto_build_indexes(index_dir, embeddings_dir, chunks_dir)

    # 2. Initialize Hybrid Retriever
    retriever = HybridRetriever(config_path=retrieval_cfg if retrieval_cfg.is_file() else None)
    if (index_dir / "dense_index.faiss").is_file():
        try:
            retriever.load_indexes(index_dir)
            logger.info(
                f"Hybrid indexes loaded: {retriever.dense_index.total_vectors} dense vectors, "
                f"{len(retriever.sparse_index.inverted_index)} sparse terms, "
                f"{retriever.sparse_index.total_docs} indexed docs."
            )
        except Exception as e:
            logger.error(f"Failed to load indexes from {index_dir}: {e}")
    else:
        logger.error(
            f"Index files still missing after auto-build at {index_dir}. "
            f"The /search and /query endpoints will return 503. "
            f"Run the full pipeline: build_embeddings.py → build_index.py"
        )

    # 3. Initialize Legal Generator
    generator = LegalGenerator(config_path=generation_cfg if generation_cfg.is_file() else None)
    logger.info(f"Legal Generator ready (Model: {getattr(generator.llm_client, 'model_name', 'default')}).")

    app.state.retriever = retriever
    app.state.generator = generator

    yield

    # Teardown logic
    logger.info("Shutting down PoliceGPT FastAPI Service...")
    if hasattr(app.state, "retriever") and getattr(app.state.retriever, "reranker", None):
        try:
            app.state.retriever.reranker.unload()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app = FastAPI(
        title="PoliceGPT Legal RAG API",
        description=(
            "Statutory Legal Question Answering and Retrieval API for Bangladesh Law Enforcement. "
            "Combines Dense FAISS (BGE-M3) + Sparse Lexical Inverted Index (RRF Fusion), "
            "Google Gemini (gemini-3.5-flash), and statutory citation guardrails."
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
