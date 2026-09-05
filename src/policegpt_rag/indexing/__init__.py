"""
PoliceGPT Indexing Layer.
Dense FAISS Index + Sparse Inverted Index for Hybrid Retrieval.
"""

from .faiss_index import FAISSVectorIndex
from .sparse_index import SparseLexicalIndex

__all__ = [
    "FAISSVectorIndex",
    "SparseLexicalIndex",
]
