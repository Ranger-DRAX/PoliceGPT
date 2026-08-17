import pytest
from policegpt_rag.indexing.faiss_index import LocalFaissIndex
from policegpt_rag.retrieval.hybrid_search import HybridSearchRetriever
from policegpt_rag.embedding.embed import BGEM3Embedder


def test_faiss_index_and_retrieval():
    dim = 64
    faiss_idx = LocalFaissIndex(dimension=dim)

    # Mock records
    import numpy as np

    v1 = np.random.randn(dim).tolist()
    v2 = np.random.randn(dim).tolist()

    records = [
        {"chunk_id": "c1", "content": "Section 378 Theft", "dense_vector": v1},
        {"chunk_id": "c2", "content": "Section 302 Murder", "dense_vector": v2},
    ]

    faiss_idx.build_index(records)
    results = faiss_idx.search(query_vector=v1, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"


def test_reciprocal_rank_fusion():
    embedder = BGEM3Embedder()
    retriever = HybridSearchRetriever(embedder=embedder)

    dense_res = [{"chunk_id": "doc_a"}, {"chunk_id": "doc_b"}]
    sparse_res = [{"chunk_id": "doc_b"}, {"chunk_id": "doc_c"}]

    fused = retriever.reciprocal_rank_fusion(dense_res, sparse_res, top_k=3)
    assert len(fused) == 3
    # doc_b is in both lists so it should rank highest
    assert fused[0]["chunk_id"] == "doc_b"
