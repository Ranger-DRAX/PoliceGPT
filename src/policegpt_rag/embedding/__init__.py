"""
PoliceGPT Embedding Layer.
BGE-M3 Multilingual Hybrid (Dense & Sparse) Embedding Engine.
"""

from .embed import BGEM3Embedder, ChunkEmbedding
from .batch_runner import EmbeddingBatchRunner, EMBEDDING_PARQUET_SCHEMA

__all__ = [
    "BGEM3Embedder",
    "ChunkEmbedding",
    "EmbeddingBatchRunner",
    "EMBEDDING_PARQUET_SCHEMA",
]
