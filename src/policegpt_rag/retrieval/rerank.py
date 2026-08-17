"""
Cross-encoder reranker leveraging BAAI/bge-reranker-v2-m3 for precise passage scoring.
"""

from typing import List, Dict, Any, Tuple, Optional
from loguru import logger


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        score_threshold: float = 0.25,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self._reranker = None

    def _load_reranker(self):
        if self._reranker is not None:
            return

        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading FlagReranker: {self.model_name}")
            self._reranker = FlagReranker(self.model_name, use_fp16=True)
        except ImportError:
            logger.warning(
                "FlagEmbedding is not installed for reranker. Falling back to sentence-transformers / pass-through."
            )
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Reranker unavailable: {e}. Passing documents through.")
                self._reranker = None

    def rerank(
        self, query: str, candidate_chunks: List[Dict[str, Any]], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Score (query, chunk_content) pairs and return the top_n most relevant chunks.
        """
        if not candidate_chunks:
            return []

        limit = top_n or self.top_n
        self._load_reranker()

        if self._reranker is None:
            # Fallback: sort by initial score
            sorted_chunks = sorted(candidate_chunks, key=lambda x: x.get("score", 0.0), reverse=True)
            return sorted_chunks[:limit]

        pairs = [[query, chunk["content"]] for chunk in candidate_chunks]

        if hasattr(self._reranker, "compute_score"):
            scores = self._reranker.compute_score(pairs, batch_size=self.batch_size)
            if isinstance(scores, float):
                scores = [scores]
        else:
            scores = self._reranker.predict(pairs, batch_size=self.batch_size)

        scored_chunks = []
        for chunk, score in zip(candidate_chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            if float(score) >= self.score_threshold or len(candidate_chunks) <= limit:
                scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks[:limit]
