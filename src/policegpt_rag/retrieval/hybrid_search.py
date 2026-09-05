"""
Hybrid Legal Search Engine combining Dense FAISS and Sparse Lexical Inverted Index.
Fuses candidate rankings via Reciprocal Rank Fusion (RRF, k=60).
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger
import yaml

try:
    from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from policegpt_rag.embedding.embed import BGEM3Embedder
    from policegpt_rag.retrieval.rerank import BGEReranker
except ImportError:
    from src.policegpt_rag.indexing.faiss_index import FAISSVectorIndex
    from src.policegpt_rag.indexing.sparse_index import SparseLexicalIndex
    from src.policegpt_rag.embedding.embed import BGEM3Embedder
    from src.policegpt_rag.retrieval.rerank import BGEReranker


class RetrievedChunk(BaseModel):
    """Rich legal chunk representation returned by the Hybrid Retriever."""
    chunk_id: str
    doc_id: str
    act_name_en: Optional[str] = None
    act_name_bn: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    content: str
    page_numbers: List[int] = Field(default_factory=list)
    rrf_score: float
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class HybridRetriever:
    """
    Two-Stage Hybrid Search Engine:
      1. Parallel Dense (FAISS Cosine) + Sparse (Inverted Lexical) search.
      2. Reciprocal Rank Fusion (RRF) with optional cross-encoder reranking.
    """

    def __init__(
        self,
        dense_index: Optional[FAISSVectorIndex] = None,
        sparse_index: Optional[SparseLexicalIndex] = None,
        embedder: Optional[BGEM3Embedder] = None,
        reranker: Optional[BGEReranker] = None,
        rrf_k: int = 60,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        top_k: int = 5,
        use_reranker: bool = False,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.dense_index = dense_index or FAISSVectorIndex()
        self.sparse_index = sparse_index or SparseLexicalIndex()
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.top_k = top_k
        self.use_reranker = use_reranker

        if config_path:
            self._load_config(config_path)

        if self.embedder is None:
            self.embedder = BGEM3Embedder(
                device="cuda",
                use_fp16=True,
                return_dense=True,
                return_sparse=True,
            )

        if self.use_reranker and self.reranker is None:
            self.reranker = BGEReranker()

    def _load_config(self, config_path: Union[str, Path]):
        path = Path(config_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ret_cfg = cfg.get("retrieval", {})
            self.top_k = ret_cfg.get("top_k", self.top_k)
            self.dense_top_k = ret_cfg.get("dense_top_k", self.dense_top_k)
            self.sparse_top_k = ret_cfg.get("sparse_top_k", self.sparse_top_k)
            self.rrf_k = ret_cfg.get("rrf_k", self.rrf_k)
            self.use_reranker = ret_cfg.get("use_reranker", self.use_reranker)

    def load_indexes(self, index_dir: Union[str, Path]):
        """Load both FAISS dense index and Sparse lexical index from directory."""
        dir_path = Path(index_dir)
        self.dense_index.load(dir_path)
        self.sparse_index.load(dir_path)
        logger.info(f"Loaded Hybrid Indexes from {dir_path.resolve()}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute Hybrid Retrieval for a given Bengali or English query.
        Returns top_k fused results sorted by relevance.
        """
        final_top_k = top_k or self.top_k

        # 1. Encode query with BGE-M3 (dense vector + sparse lexical weights)
        model = self.embedder.load()
        outputs = model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = outputs.get("dense_vecs", []) if isinstance(outputs, dict) else outputs
        query_dense = dense_vecs[0] if len(dense_vecs) > 0 else []

        sparse_weights = outputs.get("lexical_weights", None) if isinstance(outputs, dict) else None
        query_sparse = sparse_weights[0] if (sparse_weights and len(sparse_weights) > 0) else {}

        # 2. Dense Vector Retrieval (FAISS)
        dense_hits = self.dense_index.search(query_dense, top_k=self.dense_top_k)

        # 3. Sparse Lexical Retrieval (Inverted Index)
        sparse_hits = self.sparse_index.search(query_sparse, top_k=self.sparse_top_k)

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (rrf_k + rank_m)
        fused_scores: Dict[str, float] = {}
        metadata_by_chunk: Dict[str, Dict[str, Any]] = {}
        dense_details: Dict[str, Dict[str, Any]] = {}
        sparse_details: Dict[str, Dict[str, Any]] = {}

        for hit in dense_hits:
            cid = hit["chunk_id"]
            rank = hit["dense_rank"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            metadata_by_chunk[cid] = hit["metadata"]
            dense_details[cid] = hit

        # For sparse hits, look up chunk metadata from dense index if not already present
        for hit in sparse_hits:
            cid = hit["chunk_id"]
            rank = hit["sparse_rank"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            sparse_details[cid] = hit

            if cid not in metadata_by_chunk:
                idx = self.dense_index._chunk_id_to_idx.get(cid)
                if idx is not None and idx < len(self.dense_index.metadata_registry):
                    metadata_by_chunk[cid] = self.dense_index.metadata_registry[idx]
                else:
                    metadata_by_chunk[cid] = {"chunk_id": cid, "doc_id": "", "content": ""}

        # Sort candidates by combined RRF score descending
        sorted_cands = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        candidate_list: List[Dict[str, Any]] = []
        for cid, rrf_score in sorted_cands:
            meta = metadata_by_chunk.get(cid, {})
            d_hit = dense_details.get(cid, {})
            s_hit = sparse_details.get(cid, {})

            candidate_list.append({
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", ""),
                "act_name_en": meta.get("act_name_en"),
                "act_name_bn": meta.get("act_name_bn"),
                "section_number": meta.get("section_number"),
                "section_title": meta.get("section_title"),
                "content": meta.get("content", ""),
                "page_numbers": meta.get("page_numbers", []),
                "rrf_score": rrf_score,
                "dense_score": d_hit.get("dense_score"),
                "dense_rank": d_hit.get("dense_rank"),
                "sparse_score": s_hit.get("sparse_score"),
                "sparse_rank": s_hit.get("sparse_rank"),
                "metadata": meta,
            })

        # 5. Optional Cross-Encoder Reranking
        if self.use_reranker and self.reranker:
            top_candidates = candidate_list[: max(final_top_k * 2, 10)]
            reranked = self.reranker.rerank(query, top_candidates, top_k=final_top_k)
            results = [RetrievedChunk(**c) for c in reranked]
        else:
            top_candidates = candidate_list[:final_top_k]
            results = [RetrievedChunk(**c) for c in top_candidates]

        return results
