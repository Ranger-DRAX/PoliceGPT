"""
Main FastAPI entrypoint for PoliceGPT RAG Service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
import os

from .routers import query
from .schemas import HealthResponse

app = FastAPI(
    title="PoliceGPT RAG API",
    description="Specialized Multilingual Legal RAG API for Bangladesh Police & Laws",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(query.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        service="policegpt-rag",
        version="0.1.0",
    )


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    logger.info(f"Starting PoliceGPT RAG API on {host}:{port}...")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
