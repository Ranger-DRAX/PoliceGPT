"""
Embedding generator leveraging BAAI/bge-m3 for dense vectors and sparse lexical representations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from loguru import logger


class EmbeddingOutput(BaseModel):
    dense_vec: List[float]
    sparse_weights: Dict[int, float] = {}  # token_id -> weight for sparse BM25/lexical matching
    colbert_vecs: Optional[List[List[float]]] = None


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
        max_length: int = 8192,
    ):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading BGEM3FlagModel: {self.model_name} on device={self.device}")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=None if self.device == "auto" else self.device,
            )
        except ImportError:
            logger.warning(
                "FlagEmbedding is not installed. Falling back to sentence-transformers / dummy vector generator."
            )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer: {e}. Stub mode active.")
                self._model = None

    def encode_queries(self, queries: List[str]) -> List[EmbeddingOutput]:
        """
        Encode search queries into dense and sparse embeddings.
        """
        return self._encode_texts(queries, return_dense=True, return_sparse=True)

    def encode_documents(self, documents: List[str]) -> List[EmbeddingOutput]:
        """
        Encode legal document passages into dense and sparse embeddings.
        """
        return self._encode_texts(documents, return_dense=True, return_sparse=True)

    def _encode_texts(
        self, texts: List[str], return_dense: bool = True, return_sparse: bool = True
    ) -> List[EmbeddingOutput]:
        self._load_model()

        if self._model is None:
            # Fallback stub for unit testing & environment bootstrap
            dim = 1024
            return [
                EmbeddingOutput(
                    dense_vec=np.random.randn(dim).tolist(),
                    sparse_weights={101: 0.8, 202: 0.5},
                )
                for _ in texts
            ]

        # If native BGEM3FlagModel is available
        if hasattr(self._model, "encode"):
            output = self._model.encode(
                texts,
                return_dense=return_dense,
                return_sparse=return_sparse,
                max_length=self.max_length,
            )

            results = []
            dense_vecs = output.get("dense_vecs", [])
            sparse_vecs = output.get("lexical_weights", [])

            for i in range(len(texts)):
                d_vec = dense_vecs[i].tolist() if len(dense_vecs) > 0 else []
                s_vec = sparse_vecs[i] if len(sparse_vecs) > 0 else {}
                results.append(EmbeddingOutput(dense_vec=d_vec, sparse_weights=s_vec))
            return results

        # Fallback to SentenceTransformer
        dense_embs = self._model.encode(texts, normalize_embeddings=True)
        return [EmbeddingOutput(dense_vec=emb.tolist(), sparse_weights={}) for emb in dense_embs]
