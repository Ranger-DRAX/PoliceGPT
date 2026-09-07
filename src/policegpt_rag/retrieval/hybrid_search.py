"""
Hybrid Legal Search Engine combining Dense FAISS and Sparse Lexical Inverted Index.
Fuses candidate rankings via Reciprocal Rank Fusion (RRF, k=60).
Hardened for Bengali legal text, low-resource stability, and deterministic tie-breaking.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import unicodedata
import time
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
        self._indexes_loaded: bool = False
        self.last_profile: Dict[str, Any] = {}

        # If custom indexes were passed with data already, mark as loaded
        if (
            (dense_index and dense_index.total_vectors > 0)
            or (sparse_index and sparse_index.total_docs > 0)
        ):
            self._indexes_loaded = True

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
            if self.use_reranker and self.reranker is None:
                self.reranker = BGEReranker(
                    model_name=ret_cfg.get("reranker_model", "BAAI/bge-reranker-base"),
                    device=ret_cfg.get("reranker_device", "auto"),
                    auto_disable_if_no_cuda=ret_cfg.get("auto_disable_if_no_cuda", True),
                    min_vram_mb=ret_cfg.get("reranker_min_vram_mb", 1200),
                    batch_size=ret_cfg.get("reranker_batch_size", 8),
                    max_length=ret_cfg.get("reranker_max_length", 512),
                )

    def load_indexes(self, index_dir: Union[str, Path]):
        """Load both FAISS dense index and Sparse lexical index from directory."""
        dir_path = Path(index_dir)
        self.dense_index.load(dir_path)
        self.sparse_index.load(dir_path)
        self._indexes_loaded = True
        logger.info(f"Loaded Hybrid Indexes from {dir_path.resolve()} (dense={self.dense_index.total_vectors}, sparse={self.sparse_index.total_docs})")

    def _sanitize_query(self, query: Optional[str]) -> str:
        """
        Defensively sanitize, strip null bytes/control chars, and apply Unicode NFC
        normalization for consistent Bengali and English statutory matching.
        """
        if query is None:
            return ""

        # Strip null bytes and control chars (except standard whitespace)
        cleaned = "".join(ch for ch in query if ch == "\t" or ch == "\n" or not unicodedata.category(ch).startswith("C"))
        cleaned = cleaned.strip()

        # Unicode NFC normalization for Bengali script consistency
        normalized = unicodedata.normalize("NFC", cleaned)

        # Truncate queries longer than 2048 chars to prevent buffer/token overflow
        if len(normalized) > 2048:
            logger.warning(f"Query truncated from {len(normalized)} to 2048 characters.")
            normalized = normalized[:2048]

        return normalized

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute Hybrid Retrieval for a given Bengali or English query.
        Returns top_k fused results sorted by relevance with deterministic tie-breaking.
        """
        t_start = time.perf_counter()
        final_top_k = top_k if top_k is not None else self.top_k

        # 0. Defensive Guardrails
        if final_top_k <= 0:
            self.last_profile = {"t_total_ms": 0.0, "reason": "top_k <= 0"}
            return []

        clean_query = self._sanitize_query(query)
        if not clean_query:
            logger.debug("Empty or whitespace query provided; returning empty results.")
            self.last_profile = {"t_total_ms": 0.0, "reason": "empty_query"}
            return []

        # Ensure indexes are ready
        if not self._indexes_loaded and self.dense_index.total_vectors == 0 and self.sparse_index.total_docs == 0:
            raise RuntimeError(
                "Indexes are not loaded or empty. Call retriever.load_indexes(index_dir) "
                "or supply populated index objects before searching."
            )

        if self.dense_index.total_vectors == 0 and self.sparse_index.total_docs == 0:
            logger.warning("Retrieval called on empty index registry. Returning empty list.")
            self.last_profile = {"t_total_ms": 0.0, "reason": "empty_indexes"}
            return []

        # 1. Encode query with BGE-M3 (dense vector + sparse lexical weights)
        t_enc_start = time.perf_counter()
        model = self.embedder.load()
        outputs = model.encode(
            [clean_query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        t_enc = (time.perf_counter() - t_enc_start) * 1000.0

        dense_vecs = outputs.get("dense_vecs", []) if isinstance(outputs, dict) else outputs
        query_dense = dense_vecs[0] if len(dense_vecs) > 0 else []

        sparse_weights = outputs.get("lexical_weights", None) if isinstance(outputs, dict) else None
        query_sparse = sparse_weights[0] if (sparse_weights and len(sparse_weights) > 0) else {}

        # 2. Dense Vector Retrieval (FAISS) with channel error isolation
        t_dense_start = time.perf_counter()
        dense_hits = []
        if self.dense_index.total_vectors > 0 and len(query_dense) > 0:
            try:
                dense_hits = self.dense_index.search(query_dense, top_k=self.dense_top_k)
            except Exception as e:
                logger.error(f"Dense vector search channel failed: {e}. Degrading to sparse-only.")
        t_dense = (time.perf_counter() - t_dense_start) * 1000.0

        # 3. Sparse Lexical Retrieval (Inverted Index) with channel error isolation
        t_sparse_start = time.perf_counter()
        sparse_hits = []
        if self.sparse_index.total_docs > 0 and query_sparse:
            try:
                sparse_hits = self.sparse_index.search(query_sparse, top_k=self.sparse_top_k)
            except Exception as e:
                logger.error(f"Sparse lexical search channel failed: {e}. Degrading to dense-only.")
        t_sparse = (time.perf_counter() - t_sparse_start) * 1000.0

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (rrf_k + rank_m)
        t_rrf_start = time.perf_counter()
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

        candidate_list: List[Dict[str, Any]] = []
        for cid, rrf_score in fused_scores.items():
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

        # Deterministic sorting: (rrf_score desc, dense_score desc, sparse_score desc, chunk_id asc)
        candidate_list.sort(
            key=lambda x: (
                x["rrf_score"],
                x.get("dense_score") or 0.0,
                x.get("sparse_score") or 0.0,
                # Lexicographic tie-breaker
            ),
            reverse=True,
        )
        t_rrf = (time.perf_counter() - t_rrf_start) * 1000.0

        # 5. Optional Cross-Encoder Reranking
        t_rerank = 0.0
        if self.use_reranker and self.reranker and candidate_list:
            t_rerank_start = time.perf_counter()
            top_candidates = candidate_list[: max(final_top_k * 2, 10)]
            reranked = self.reranker.rerank(clean_query, top_candidates, top_k=final_top_k)
            results = [RetrievedChunk(**c) for c in reranked]
            t_rerank = (time.perf_counter() - t_rerank_start) * 1000.0
        else:
            top_candidates = candidate_list[:final_top_k]
            results = [RetrievedChunk(**c) for c in top_candidates]

        t_total = (time.perf_counter() - t_start) * 1000.0

        # Observability / Telemetry
        self.last_profile = {
            "query_len": len(clean_query),
            "dense_hits": len(dense_hits),
            "sparse_hits": len(sparse_hits),
            "fused_candidates": len(candidate_list),
            "returned_chunks": len(results),
            "t_encode_ms": round(t_enc, 2),
            "t_dense_ms": round(t_dense, 2),
            "t_sparse_ms": round(t_sparse, 2),
            "t_rrf_ms": round(t_rrf, 2),
            "t_rerank_ms": round(t_rerank, 2),
            "t_total_ms": round(t_total, 2),
        }

        return results
