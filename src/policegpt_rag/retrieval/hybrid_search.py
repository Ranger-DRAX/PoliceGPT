"""
Hybrid search retriever combining dense vector similarity with sparse lexical matching using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..embedding.embed import BGEM3Embedder
from ..indexing.qdrant_index import QdrantHybridIndex
from ..indexing.faiss_index import LocalFaissIndex


class HybridSearchRetriever:
    def __init__(
        self,
        embedder: BGEM3Embedder,
        qdrant_index: Optional[QdrantHybridIndex] = None,
        faiss_index: Optional[LocalFaissIndex] = None,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.qdrant_index = qdrant_index
        self.faiss_index = faiss_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_qdrant: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract query embeddings and retrieve top-k legal chunks.
        """
        query_embs = self.embedder.encode_queries([query])
        q_emb = query_embs[0]

        if use_qdrant and self.qdrant_index is not None:
            results = self.qdrant_index.hybrid_search(
                query_dense=q_emb.dense_vec,
                query_sparse=q_emb.sparse_weights,
                top_k=top_k,
                filter_dict=filters,
            )
            if results:
                return results

        # Fallback to local FAISS index
        if self.faiss_index is not None:
            return self.faiss_index.search(query_vector=q_emb.dense_vec, top_k=top_k)

        return []

    def reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge ranked lists from dense and sparse retrieval via RRF.
        RRF Score(d) = sum(1 / (k + rank(d)))
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["chunk_id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["chunk_id"]
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_results = []
        for doc_id, score in sorted_docs:
            item = dict(doc_map[doc_id])
            item["rrf_score"] = score
            final_results.append(item)

        return final_results
