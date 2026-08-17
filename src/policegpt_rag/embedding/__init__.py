"""
Embedding module: BGE-M3 Multilingual dense and sparse representation generator.
"""

from .embed import BGEM3Embedder, EmbeddingOutput
from .batch_runner import BatchEmbeddingRunner

__all__ = ["BGEM3Embedder", "EmbeddingOutput", "BatchEmbeddingRunner"]
