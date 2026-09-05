"""
FAISS Vector Index for BGE-M3 1024-dimensional dense legal embeddings.
Provides sub-millisecond exact cosine similarity search (IndexFlatIP) with 100% recall.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pyarrow.parquet as pq
from loguru import logger

try:
    import faiss
except ImportError:
    faiss = None


class FAISSVectorIndex:
    """
    Dense Vector Index wrapping FAISS IndexFlatIP.
    Stores and searches 1024-dim BGE-M3 embeddings, maintaining an aligned metadata registry.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.metadata_registry: List[Dict[str, Any]] = []
        self._chunk_id_to_idx: Dict[str, int] = {}

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def _ensure_faiss(self):
        if faiss is None:
            raise ImportError(
                "faiss is not installed. Install via `pip install faiss-cpu`."
            )

    def init_index(self):
        """Create an empty FAISS IndexFlatIP (exact inner product / cosine similarity)."""
        self._ensure_faiss()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_registry = []
        self._chunk_id_to_idx = {}

    def build_from_records(
        self,
        dense_vectors: Union[np.ndarray, List[List[float]]],
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """
        Build the FAISS index from in-memory vectors and associated metadata.
        """
        self.init_index()

        if isinstance(dense_vectors, list):
            vectors_np = np.array(dense_vectors, dtype=np.float32)
        else:
            vectors_np = dense_vectors.astype(np.float32)

        if len(vectors_np) == 0:
            logger.warning("Empty vector set passed to FAISSVectorIndex.build_from_records().")
            return

        if vectors_np.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors_np.shape[1]}"
            )

        # Normalize to unit length for exact cosine similarity via inner product
        faiss.normalize_L2(vectors_np)
        self.index.add(vectors_np)

        self.metadata_registry = metadata_list
        self._chunk_id_to_idx = {
            m.get("chunk_id", str(i)): i for i, m in enumerate(metadata_list)
        }
        logger.info(f"Built FAISS IndexFlatIP with {self.index.ntotal} vectors (dim={self.dimension}).")

    def build_from_parquet_files(
        self,
        parquet_files: List[Union[str, Path]],
        chunks_json_dir: Optional[Union[str, Path]] = "data/processed",
    ) -> None:
        """
        Extract dense vectors from Parquet files and enrich metadata using raw chunk JSON files.
        """
        dense_vectors: List[List[float]] = []
        metadata_list: List[Dict[str, Any]] = []

        # Build chunk_id -> content/metadata lookup from raw chunk JSON files if available
        raw_chunks_lookup: Dict[str, Dict[str, Any]] = {}
        if chunks_json_dir:
            json_dir = Path(chunks_json_dir)
            if json_dir.is_dir():
                for json_file in json_dir.glob("*_chunks.json"):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            chunks_data = json.load(f)
                            for c in chunks_data:
                                raw_chunks_lookup[c["chunk_id"]] = c
                    except Exception as e:
                        logger.warning(f"Error reading {json_file.name}: {e}")

        for p_file in parquet_files:
            path = Path(p_file)
            if not path.is_file():
                continue

            table = pq.read_table(str(path))
            chunk_ids = table["chunk_id"].to_pylist()
            doc_ids = table["doc_id"].to_pylist()
            vectors = table["dense_vector"].to_pylist()
            section_numbers = table["section_number"].to_pylist()
            section_titles = table["section_title"].to_pylist()
            models = table["embedding_model"].to_pylist()

            for i in range(len(chunk_ids)):
                cid = chunk_ids[i]
                dense_vectors.append(vectors[i])

                raw_meta = raw_chunks_lookup.get(cid, {})
                metadata_list.append({
                    "chunk_id": cid,
                    "doc_id": doc_ids[i],
                    "act_name_bn": raw_meta.get("act_name_bn"),
                    "act_name_en": raw_meta.get("act_name_en"),
                    "act_year": raw_meta.get("act_year"),
                    "chapter": raw_meta.get("chapter"),
                    "section_number": section_numbers[i] or raw_meta.get("section_number"),
                    "section_title": section_titles[i] or raw_meta.get("section_title"),
                    "content": raw_meta.get("content", ""),
                    "page_numbers": raw_meta.get("page_numbers", []),
                    "embedding_model": models[i],
                })

        self.build_from_records(dense_vectors, metadata_list)

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform exact inner-product / cosine similarity search.
        Returns:
            List of dicts: {"chunk_id": str, "dense_score": float, "metadata": dict}
        """
        self._ensure_faiss()
        if self.index is None or self.index.ntotal == 0:
            return []

        if isinstance(query_vector, list):
            q_np = np.array([query_vector], dtype=np.float32)
        elif len(query_vector.shape) == 1:
            q_np = query_vector.reshape(1, -1).astype(np.float32)
        else:
            q_np = query_vector.astype(np.float32)

        # Normalize query vector for cosine similarity
        faiss.normalize_L2(q_np)

        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_np, actual_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1 or idx >= len(self.metadata_registry):
                continue
            meta = self.metadata_registry[idx]
            results.append({
                "chunk_id": meta["chunk_id"],
                "dense_score": float(score),
                "dense_rank": rank + 1,
                "metadata": meta,
            })

        return results

    def save(self, index_dir: Union[str, Path]):
        """Persist FAISS binary index and metadata JSON to disk."""
        self._ensure_faiss()
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = out_dir / "dense_index.faiss"
        meta_path = out_dir / "chunks_metadata.json"

        if self.index is not None:
            faiss.write_index(self.index, str(faiss_path.resolve()))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_registry, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index ({self.total_vectors} vectors) -> {faiss_path}")
        logger.info(f"Saved chunks metadata registry -> {meta_path}")

    def load(self, index_dir: Union[str, Path]):
        """Load FAISS binary index and metadata JSON from disk."""
        self._ensure_faiss()
        in_dir = Path(index_dir)
        faiss_path = in_dir / "dense_index.faiss"
        meta_path = in_dir / "chunks_metadata.json"

        if not faiss_path.is_file():
            raise FileNotFoundError(f"FAISS index file not found at: {faiss_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Chunks metadata file not found at: {meta_path}")

        self.index = faiss.read_index(str(faiss_path.resolve()))
        self.dimension = self.index.d

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_registry = json.load(f)

        self._chunk_id_to_idx = {
            m.get("chunk_id", str(i)): i for i, m in enumerate(self.metadata_registry)
        }
        logger.info(
            f"Loaded FAISS index with {self.total_vectors} vectors (dim={self.dimension}) from {faiss_path}"
        )
