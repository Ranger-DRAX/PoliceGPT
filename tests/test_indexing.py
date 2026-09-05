"""
Unit tests for FAISS Vector Index, Sparse Lexical Index, and Hybrid RRF Retrieval.
All model calls are mocked — fast, reproducible, and zero network/GPU required.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

try:
    from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
except ImportError:
    from src.policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from src.policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from src.policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk


def test_faiss_vector_index_build_and_search():
    """Verify FAISS vector index builds and returns highest cosine similarity vector."""
    dim = 1024
    v_index = FAISSVectorIndex(dimension=dim)

    # 3 orthogonal-like unit vectors
    v1 = [0.0] * dim
    v1[0] = 1.0  # target vector

    v2 = [0.0] * dim
    v2[1] = 1.0

    v3 = [0.0] * dim
    v3[2] = 1.0

    metadata = [
        {"chunk_id": "c1", "doc_id": "penal_code", "section_number": "378", "content": "Theft definition"},
        {"chunk_id": "c2", "doc_id": "penal_code", "section_number": "379", "content": "Theft punishment"},
        {"chunk_id": "c3", "doc_id": "police_act", "section_number": "23", "content": "Police duties"},
    ]

    v_index.build_from_records([v1, v2, v3], metadata)
    assert v_index.total_vectors == 3

    # Query with vector matching v1
    hits = v_index.search(v1, top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["dense_rank"] == 1
    assert hits[0]["dense_score"] > 0.99  # Normalized dot product == 1.0


def test_faiss_vector_index_save_and_load(tmp_path):
    """Verify FAISS index and metadata persistence."""
    dim = 1024
    v_index = FAISSVectorIndex(dimension=dim)
    v1 = [0.5] * dim
    metadata = [{"chunk_id": "chunk_save", "doc_id": "act_1861", "content": "Sample content"}]
    v_index.build_from_records([v1], metadata)

    save_dir = tmp_path / "faiss_test"
    v_index.save(save_dir)

    loaded_index = FAISSVectorIndex(dimension=dim)
    loaded_index.load(save_dir)

    assert loaded_index.total_vectors == 1
    assert loaded_index.metadata_registry[0]["chunk_id"] == "chunk_save"


def test_sparse_lexical_index_search():
    """Verify sparse lexical inverted index computes term dot product."""
    s_index = SparseLexicalIndex()

    # Document 1 has token "churi" (3.0) and "shasti" (1.0)
    # Document 2 has token "police" (2.0)
    records = [
        ("c1", {"churi": 3.0, "shasti": 1.0}),
        ("c2", {"police": 2.0, "duty": 1.5}),
    ]
    s_index.build_from_records(records)
    assert s_index.total_docs == 2

    # Query with {"churi": 2.0} -> should match only c1 with score = 2.0 * 3.0 = 6.0
    results = s_index.search({"churi": 2.0}, top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["sparse_score"] == pytest.approx(6.0)


def test_sparse_lexical_index_save_and_load(tmp_path):
    """Verify sparse index serialization to and from JSON."""
    s_index = SparseLexicalIndex()
    s_index.build_from_records([("c1", {"term_a": 2.5})])

    save_dir = tmp_path / "sparse_test"
    s_index.save(save_dir)

    loaded_sparse = SparseLexicalIndex()
    loaded_sparse.load(save_dir)

    assert loaded_sparse.total_docs == 1
    results = loaded_sparse.search({"term_a": 1.0})
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_hybrid_rrf_retrieval():
    """Verify HybridRetriever fuses dense and sparse rankings with RRF formula."""
    dim = 1024
    dense_idx = FAISSVectorIndex(dimension=dim)
    sparse_idx = SparseLexicalIndex()

    v1 = [0.0] * dim; v1[0] = 1.0  # matches query dense
    v2 = [0.0] * dim; v2[1] = 1.0

    dense_idx.build_from_records(
        [v1, v2],
        [
            {"chunk_id": "c1", "doc_id": "act", "section_number": "1", "content": "Section 1 text"},
            {"chunk_id": "c2", "doc_id": "act", "section_number": "2", "content": "Section 2 text"},
        ]
    )

    # c2 has high sparse lexical match
    sparse_idx.build_from_records([
        ("c1", {"term_x": 1.0}),
        ("c2", {"target_term": 5.0}),
    ])

    # Mock embedder
    mock_embedder = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [v1],  # c1 will be dense #1
        "lexical_weights": [{"target_term": 2.0}],  # c2 will be sparse #1
    }
    mock_embedder.load.return_value = mock_model

    retriever = HybridRetriever(
        dense_index=dense_idx,
        sparse_index=sparse_idx,
        embedder=mock_embedder,
        rrf_k=60,
    )

    results = retriever.retrieve("test legal query", top_k=2)

    assert len(results) == 2
    # Dense search returns both: c1 rank 1, c2 rank 2 (dense_top_k=20 > 2 docs)
    # Sparse search returns: c2 rank 1 (only "target_term" match)
    # c2 RRF: 1/(60+2) [dense rank 2] + 1/(60+1) [sparse rank 1] = 1/62 + 1/61
    # c1 RRF: 1/(60+1) [dense rank 1 only]
    assert results[0].chunk_id == "c2"
    assert results[0].rrf_score == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert results[1].chunk_id == "c1"
    assert results[1].rrf_score == pytest.approx(1.0 / 61.0)
