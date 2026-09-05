"""
Cross-Encoder Reranker using BAAI/bge-reranker models.
Optional post-retrieval stage for precision ranking of legal candidate chunks.
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class BGEReranker:
    """
    Cross-Encoder Reranker using FlagReranker (BAAI/bge-reranker-base or large).
    Lazy-loaded to prevent consuming VRAM/RAM until explicitly needed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.requested_device = device.lower()
        self.use_fp16 = use_fp16
        self._model = None

        # Resolve device
        if self.requested_device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    self.use_fp16 = False
            except ImportError:
                self.device = "cpu"
                self.use_fp16 = False
        else:
            self.device = "cpu"
            self.use_fp16 = False

    def load(self):
        """Lazy load the FlagReranker."""
        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagReranker requires FlagEmbedding. Install via `pip install FlagEmbedding`."
            )

        logger.info(f"Loading FlagReranker ('{self.model_name}') on {self.device} (fp16={self.use_fp16})...")
        self._model = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score (query, document) pairs using the cross-encoder.
        Args:
            query: User's legal search prompt.
            candidates: List of retrieved chunks from hybrid retrieval.
            top_k: Number of highest-scoring chunks to return.
        """
        if not candidates:
            return []

        model = self.load()
        pairs = []
        for cand in candidates:
            meta = cand.get("metadata", {})
            content = meta.get("content") or cand.get("content", "")
            pairs.append([query, content])

        scores = model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        reranked = []
        for cand, score in zip(candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            reranked.append(cand_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
