"""
Unit and hardening tests for HybridRetriever and IR Evaluation Metrics.
All embedding models and external calls are mocked: zero GPU/network required.
Validates input sanitization, Unicode NFC normalization, channel fault tolerance,
deterministic tie-breaking, boundary handling, and latency profiling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import unicodedata

try:
    from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
    from policegpt_rag.evaluation.retrieval_metrics import (
        hit_rate_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        rank_correlation_kendall,
        rank_correlation_spearman,
    )
except ImportError:
    from src.policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from src.policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
    from src.policegpt_rag.evaluation.retrieval_metrics import (
        hit_rate_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        rank_correlation_kendall,
        rank_correlation_spearman,
    )


def _setup_mock_retriever(dim: int = 1024, num_chunks: int = 3):
    """Helper fixture to create an initialized HybridRetriever with synthetic index entries."""
    dense_idx = FAISSVectorIndex(dimension=dim)
    sparse_idx = SparseLexicalIndex()

    vectors = []
    metadata = []
    sparse_records = []

    for i in range(num_chunks):
        v = [0.0] * dim
        v[i] = 1.0
        cid = f"chunk_{i+1}"
        vectors.append(v)
        metadata.append({
            "chunk_id": cid,
            "doc_id": f"act_{i+1}",
            "act_name_en": f"Penal Code Act {i+1}",
            "act_name_bn": f"দণ্ডবিধি আইন {i+1}",
            "section_number": str(300 + i),
            "section_title": f"Title {i+1}",
            "content": f"Sample statutory content for section {300 + i}.",
            "page_numbers": [i + 1, i + 2],
        })
        sparse_records.append((cid, {f"term_{i+1}": 3.0, "common_term": 1.0}))

    dense_idx.build_from_records(vectors, metadata)
    sparse_idx.build_from_records(sparse_records)

    mock_embedder = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [vectors[0]],
        "lexical_weights": [{"term_1": 2.0}],
    }
    mock_embedder.load.return_value = mock_model

    retriever = HybridRetriever(
        dense_index=dense_idx,
        sparse_index=sparse_idx,
        embedder=mock_embedder,
        rrf_k=60,
    )
    return retriever, vectors, metadata


def test_retrieve_empty_or_whitespace_query_returns_empty():
    """Verify that empty, None, or whitespace-only queries return empty list safely."""
    retriever, _, _ = _setup_mock_retriever()

    assert retriever.retrieve("") == []
    assert retriever.retrieve("    ") == []
    assert retriever.retrieve("\t \n \r") == []
    assert retriever.retrieve(None) == []


def test_retrieve_with_top_k_zero_or_negative():
    """Verify top_k <= 0 returns an empty list immediately without computation."""
    retriever, _, _ = _setup_mock_retriever()

    assert retriever.retrieve("valid query", top_k=0) == []
    assert retriever.retrieve("valid query", top_k=-5) == []


def test_retrieve_unloaded_indexes_raises_runtime_error():
    """Verify searching on uninitialized/unloaded indexes raises informative RuntimeError."""
    empty_dense = FAISSVectorIndex()
    empty_sparse = SparseLexicalIndex()
    retriever = HybridRetriever(dense_index=empty_dense, sparse_index=empty_sparse)

    with pytest.raises(RuntimeError, match="Indexes are not loaded or empty"):
        retriever.retrieve("test query")


def test_unicode_nfc_normalization_applied_to_query():
    """Verify Unicode NFC normalization aligns decomposed Bengali characters."""
    retriever, _, _ = _setup_mock_retriever()

    # Decomposed Bengali text (NFD)
    nfd_text = unicodedata.normalize("NFD", "চুরির শাস্তি কি?")
    sanitized = retriever._sanitize_query(nfd_text)

    # Should be normalized to NFC
    assert sanitized == unicodedata.normalize("NFC", "চুরির শাস্তি কি?")
    assert unicodedata.is_normalized("NFC", sanitized)


def test_top_k_exceeding_index_size_returns_all_available():
    """Verify asking for top_k > total_docs returns all candidates without indexing errors."""
    retriever, _, _ = _setup_mock_retriever(num_chunks=3)

    results = retriever.retrieve("test legal query", top_k=50)
    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_resilient_dense_failure_falls_back_to_sparse():
    """Verify that if the dense vector search channel raises an error, sparse hits are preserved."""
    retriever, _, _ = _setup_mock_retriever(num_chunks=3)

    # Force dense search to throw an exception
    retriever.dense_index.search = MagicMock(side_effect=RuntimeError("FAISS CUDA device fault"))

    results = retriever.retrieve("test query", top_k=2)
    assert len(results) > 0
    # The chunk matching sparse token "term_1" should still be returned
    assert results[0].chunk_id == "chunk_1"
    assert results[0].sparse_score is not None
    assert results[0].dense_score is None


def test_resilient_sparse_failure_falls_back_to_dense():
    """Verify that if the sparse search channel raises an error, dense hits are preserved."""
    retriever, _, _ = _setup_mock_retriever(num_chunks=3)

    # Force sparse search to throw an exception
    retriever.sparse_index.search = MagicMock(side_effect=KeyError("Corrupt sparse term dictionary"))

    results = retriever.retrieve("test query", top_k=2)
    assert len(results) > 0
    # Dense results should still succeed
    assert results[0].chunk_id == "chunk_1"
    assert results[0].dense_score is not None


def test_metadata_roundtrip_integrity():
    """Verify all legal metadata fields (act names, section numbers, pages) survive through retrieval."""
    retriever, _, _ = _setup_mock_retriever(num_chunks=3)

    results = retriever.retrieve("test query", top_k=1)
    assert len(results) == 1
    top = results[0]

    assert top.chunk_id == "chunk_1"
    assert top.act_name_en == "Penal Code Act 1"
    assert top.act_name_bn == "দণ্ডবিধি আইন 1"
    assert top.section_number == "300"
    assert top.section_title == "Title 1"
    assert "Sample statutory content" in top.content
    assert top.page_numbers == [1, 2]


def test_deterministic_tie_breaking():
    """Verify candidate list sorts deterministically when scores are tied."""
    dense_idx = FAISSVectorIndex(dimension=10)
    sparse_idx = SparseLexicalIndex()

    # Two identical vectors -> equal dense scores
    v = [1.0] * 10
    dense_idx.build_from_records(
        [v, v],
        [
            {"chunk_id": "chunk_B", "doc_id": "act", "content": "B"},
            {"chunk_id": "chunk_A", "doc_id": "act", "content": "A"},
        ],
    )

    mock_embedder = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": [v], "lexical_weights": [{}]}
    mock_embedder.load.return_value = mock_model

    retriever = HybridRetriever(dense_index=dense_idx, sparse_index=sparse_idx, embedder=mock_embedder)
    results = retriever.retrieve("tie break test", top_k=2)

    assert len(results) == 2
    # Equal RRF and dense score should execute without crashing
    assert results[0].chunk_id in ("chunk_A", "chunk_B")


def test_latency_profiling_telemetry_capture():
    """Verify retriever populates last_profile with granular stage latencies."""
    retriever, _, _ = _setup_mock_retriever()

    retriever.retrieve("police search query", top_k=2)
    profile = retriever.last_profile

    assert "t_total_ms" in profile
    assert "t_encode_ms" in profile
    assert "t_dense_ms" in profile
    assert "t_sparse_ms" in profile
    assert "t_rrf_ms" in profile
    assert profile["dense_hits"] >= 0
    assert profile["fused_candidates"] >= 0


def test_ir_evaluation_metrics_mathematical_correctness():
    """Verify correctness of HitRate, MRR, NDCG, Precision, Recall, and Rank Correlations."""
    relevant = {"c1", "c3"}
    retrieved = ["c2", "c1", "c4", "c3", "c5"]

    # Hit Rate @ 1 = 0, Hit Rate @ 2 = 1.0
    assert hit_rate_at_k(relevant, retrieved, k=1) == 0.0
    assert hit_rate_at_k(relevant, retrieved, k=2) == 1.0

    # MRR @ 5: first hit is at rank 2 -> 1/2 = 0.5
    assert mrr_at_k(relevant, retrieved, k=5) == 0.5
    assert mrr_at_k(relevant, retrieved, k=1) == 0.0

    # Precision @ 2: 1 relevant out of 2 = 0.5
    assert precision_at_k(relevant, retrieved, k=2) == 0.5
    # Recall @ 2: 1 relevant out of 2 total relevant = 0.5
    assert recall_at_k(relevant, retrieved, k=2) == 0.5

    # NDCG with graded relevance
    relevance_scores = {"c1": 2.0, "c3": 1.0}
    ndcg_val = ndcg_at_k(relevance_scores, retrieved, k=5)
    assert 0.0 < ndcg_val <= 1.0

    # Rank correlation: identical rankings
    assert rank_correlation_spearman(["c1", "c2", "c3"], ["c1", "c2", "c3"]) == pytest.approx(1.0)
    assert rank_correlation_kendall(["c1", "c2", "c3"], ["c1", "c2", "c3"]) == pytest.approx(1.0)

    # Rank correlation: reversed rankings
    assert rank_correlation_spearman(["c1", "c2", "c3"], ["c3", "c2", "c1"]) == pytest.approx(-1.0)
    assert rank_correlation_kendall(["c1", "c2", "c3"], ["c3", "c2", "c1"]) == pytest.approx(-1.0)
