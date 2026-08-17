"""
FAISS local vector index for offline prototyping, development, and fast standalone inference.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
from loguru import logger


class LocalFaissIndex:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.doc_store: List[Dict[str, Any]] = []

    def _init_index(self):
        if self.index is not None:
            return
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized cosine similarity
        except ImportError:
            logger.warning("FAISS is not installed. Using in-memory cosine numpy search.")
            self.index = "numpy_fallback"

    def build_index(self, records: List[Dict[str, Any]]):
        self._init_index()
        self.doc_store = [
            {k: v for k, v in r.items() if k not in ("dense_vector", "sparse_weights")}
            for r in records
        ]

        vectors = np.array([r["dense_vector"] for r in records], dtype=np.float32)
        # Normalize vectors for Cosine Similarity
        faiss_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        faiss_norm[faiss_norm == 0] = 1e-10
        vectors = vectors / faiss_norm

        if self.index != "numpy_fallback":
            self.index.reset()
            self.index.add(vectors)
        else:
            self._vectors = vectors

        logger.info(f"Built Local FAISS Index with {len(records)} vectors (dim={self.dimension}).")

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.doc_store:
            return []

        q_vec = np.array([query_vector], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        if self.index != "numpy_fallback":
            scores, indices = self.index.search(q_vec, min(top_k, len(self.doc_store)))
            scores = scores[0]
            indices = indices[0]
        else:
            sims = np.dot(self._vectors, q_vec.T).squeeze()
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.doc_store):
                item = dict(self.doc_store[idx])
                item["score"] = float(score)
                results.append(item)

        return results

    def save(self, directory: str | Path):
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.index != "numpy_fallback":
            import faiss
            faiss.write_index(self.index, str(dir_path / "index.faiss"))

        with open(dir_path / "docstore.pkl", "wb") as f:
            pickle.dump(self.doc_store, f)
        logger.info(f"Saved FAISS index and docstore to {dir_path}")

    def load(self, directory: str | Path):
        dir_path = Path(directory)
        with open(dir_path / "docstore.pkl", "rb") as f:
            self.doc_store = pickle.load(f)

        faiss_path = dir_path / "index.faiss"
        if faiss_path.exists():
            import faiss
            self.index = faiss.read_index(str(faiss_path))
        else:
            self.index = "numpy_fallback"
        logger.info(f"Loaded FAISS index with {len(self.doc_store)} items.")
