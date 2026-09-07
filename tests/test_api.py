"""
Unit and integration tests for PoliceGPT FastAPI Service.
Uses fastapi.testclient.TestClient with dependency overrides:
fast, reproducible, and zero network/GPU required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_retriever, get_generator
from api.schemas import (
    QueryResponse,
    SearchResponse,
    HealthResponse,
    ChunkDetailResponse,
)

try:
    from policegpt_rag.retrieval.hybrid_search import RetrievedChunk
    from policegpt_rag.generation.generator import LegalAnswerResponse
    from policegpt_rag.generation.guardrails import SourceReference
except ImportError:
    from src.policegpt_rag.retrieval.hybrid_search import RetrievedChunk
    from src.policegpt_rag.generation.generator import LegalAnswerResponse
    from src.policegpt_rag.generation.guardrails import SourceReference


@pytest.fixture
def mock_retriever():
    """Mock HybridRetriever returning synthetic statutory chunks."""
    retriever = MagicMock()
    retriever.use_reranker = True
    retriever.dense_index.total_vectors = 494
    retriever.dense_index._chunk_id_to_idx = {"chunk_379": 0}
    retriever.dense_index.metadata_registry = [
        {
            "chunk_id": "chunk_379",
            "doc_id": "penal_code_1860",
            "act_name_en": "The Penal Code, 1860",
            "act_name_bn": "দণ্ডবিধি, ১৮৬০",
            "section_number": "379",
            "section_title": "Theft punishment",
            "content": "Whoever commits theft shall be punished with imprisonment up to 3 years.",
            "page_numbers": [125],
        }
    ]
    retriever.sparse_index.inverted_index = {"চুরি": [("chunk_379", 3.0)]}
    retriever.reranker.get_status.return_value = {
        "is_enabled": True,
        "device": "cuda",
        "model_name": "BAAI/bge-reranker-base",
    }
    retriever.last_profile = {
        "t_encode_ms": 10.5,
        "t_dense_ms": 1.2,
        "t_sparse_ms": 0.8,
        "t_rrf_ms": 0.5,
        "t_rerank_ms": 2.1,
        "t_total_ms": 15.1,
    }

    # Mock retrieve method
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="chunk_379",
            doc_id="penal_code_1860",
            act_name_en="The Penal Code, 1860",
            act_name_bn="দণ্ডবিধি, ১৮৬০",
            section_number="379",
            section_title="Theft punishment",
            content="Whoever commits theft shall be punished with imprisonment up to 3 years.",
            page_numbers=[125],
            rrf_score=0.0325,
            dense_score=0.88,
            dense_rank=1,
            sparse_score=3.0,
            sparse_rank=1,
            rerank_score=0.92,
        )
    ]
    return retriever


@pytest.fixture
def mock_generator():
    """Mock LegalGenerator returning structured LegalAnswerResponse."""
    generator = MagicMock()
    generator.llm_client.model_name = "mock-gemini-flash"

    def _generate(query, retrieved_chunks):
        if query == "আইন":
            return LegalAnswerResponse(
                query=query,
                status="needs_clarification",
                answer="আপনার প্রশ্নটি অত্যন্ত সংক্ষিপ্ত বা অস্পষ্ট।",
                clarification_prompt="অনুগ্রহ করে বিস্তারিত লিখুন।",
                latency_ms=5.0,
                model_name="mock-gemini-flash",
            )
        return LegalAnswerResponse(
            query=query,
            status="answered",
            answer="দণ্ডবিধি, ১৮৬০ এর ধারা ৩৭৯ অনুসারে চুরির শাস্তি অনধিক ৩ বছর কারাদণ্ড [S1]।",
            sources=[
                SourceReference(
                    source_id="S1",
                    act_name="দণ্ডবিধি, ১৮৬০",
                    section="ধারা 379",
                    section_title="Theft punishment",
                    page_numbers=[125],
                    snippet="Whoever commits theft shall be punished...",
                )
            ],
            is_grounded=True,
            claim_support_score=0.85,
            latency_ms=25.0,
            model_name="mock-gemini-flash",
        )

    generator.generate_answer.side_effect = _generate
    return generator


@pytest.fixture
def client(mock_retriever, mock_generator):
    """FastAPI TestClient with overridden dependencies."""
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.state.retriever = mock_retriever
    app.state.generator = mock_generator

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root_endpoint(client):
    """Verify GET / returns API metadata and documentation links."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "PoliceGPT Legal RAG API"
    assert data["version"] == "1.0.0"
    assert "docs" in data
    assert "health" in data


def test_health_endpoint(client):
    """Verify GET /api/v1/health returns healthy status and index stats."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["indexes_loaded"] is True
    assert data["total_dense_vectors"] == 494
    assert data["total_sparse_terms"] == 1
    assert data["reranker_status"]["is_enabled"] is True


def test_search_endpoint_returns_chunks(client):
    """Verify POST /api/v1/search executes hybrid retrieval without LLM call."""
    resp = client.post(
        "/api/v1/search",
        json={"query": "চুরির শাস্তি", "top_k": 5, "use_reranker": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "চুরির শাস্তি"
    assert data["total_results"] == 1
    chunk = data["results"][0]
    assert chunk["chunk_id"] == "chunk_379"
    assert chunk["section_number"] == "379"
    assert chunk["rerank_score"] == 0.92
    assert "profile" in data


def test_query_endpoint_full_rag_answered(client):
    """Verify POST /api/v1/query returns complete grounded answer with sources."""
    resp = client.post(
        "/api/v1/query",
        json={"query": "চুরির শাস্তি কি?", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert "দণ্ডবিধি, ১৮৬০ এর ধারা ৩৭৯" in data["answer"]
    assert data["is_grounded"] is True
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "S1"
    assert data["sources"][0]["section"] == "ধারা 379"
    assert data["claim_support_score"] == pytest.approx(0.85)


def test_query_endpoint_ambiguous_short_circuit(client):
    """Verify POST /api/v1/query handles ambiguous query with needs_clarification."""
    resp = client.post(
        "/api/v1/query",
        json={"query": "আইন"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "needs_clarification"
    assert "সংক্ষিপ্ত বা অস্পষ্ট" in data["answer"]


def test_query_endpoint_validation_error_on_empty_query(client):
    """Verify POST /api/v1/query rejects empty query with HTTP 422."""
    resp = client.post(
        "/api/v1/query",
        json={"query": ""},
    )
    assert resp.status_code == 422


def test_query_endpoint_validation_error_on_invalid_top_k(client):
    """Verify POST /api/v1/query rejects top_k < 1 with HTTP 422."""
    resp = client.post(
        "/api/v1/query",
        json={"query": "valid query", "top_k": 0},
    )
    assert resp.status_code == 422


def test_chunk_detail_endpoint_found(client):
    """Verify GET /api/v1/chunks/{chunk_id} returns chunk details."""
    resp = client.get("/api/v1/chunks/chunk_379")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_id"] == "chunk_379"
    assert data["section_number"] == "379"
    assert "theft" in data["content"].lower()


def test_chunk_detail_endpoint_not_found(client):
    """Verify GET /api/v1/chunks/{chunk_id} returns 404 for unknown chunk."""
    resp = client.get("/api/v1/chunks/unknown_chunk_id")
    assert resp.status_code == 404
    data = resp.json()
    assert "not found" in data["detail"].lower()


def test_timing_middleware_header(client):
    """Verify custom X-Process-Time-Ms header is present in all responses."""
    resp = client.get("/")
    assert "x-process-time-ms" in resp.headers
    process_time = float(resp.headers["x-process-time-ms"])
    assert process_time >= 0.0
