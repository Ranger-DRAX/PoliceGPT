"""
PoliceGPT Retrieval Layer.
Hybrid Search (Dense FAISS + Sparse Lexical RRF) and Cross-Encoder Reranking.
"""

from .hybrid_search import HybridRetriever, RetrievedChunk
from .rerank import BGEReranker

__all__ = [
    "HybridRetriever",
    "RetrievedChunk",
    "BGEReranker",
]
