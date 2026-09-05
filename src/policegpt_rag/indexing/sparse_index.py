"""
Inverted Lexical Index for BGE-M3 Sparse (Learned Lexical) Weights.
Enables fast exact keyword and statutory terminology search (e.g. section numbers, legal terms).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
import pyarrow.parquet as pq
from loguru import logger


class SparseLexicalIndex:
    """
    In-memory Inverted Index for sparse lexical weights produced by BGE-M3.
    Computes dot-product similarity between query lexical weights and document token weights.
    """

    def __init__(self):
        # term -> [(chunk_id, weight), ...]
        self.inverted_index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.total_docs: int = 0
        self.indexed_chunk_ids: List[str] = []

    def build_from_records(self, chunk_sparse_records: List[Tuple[str, Dict[str, float]]]):
        """Build index from a list of (chunk_id, sparse_dict) tuples."""
        self.inverted_index = defaultdict(list)
        chunk_ids_set = set()

        for chunk_id, weights in chunk_sparse_records:
            if not weights:
                continue
            chunk_ids_set.add(chunk_id)
            for token, weight in weights.items():
                self.inverted_index[str(token)].append((chunk_id, float(weight)))

        self.indexed_chunk_ids = sorted(list(chunk_ids_set))
        self.total_docs = len(self.indexed_chunk_ids)
        logger.info(
            f"Built SparseLexicalIndex with {len(self.inverted_index)} distinct terms across {self.total_docs} chunks."
        )

    def build_from_parquet_files(self, parquet_files: List[Union[str, Path]]):
        """Extract sparse_weights_json from Parquet files and populate the inverted index."""
        records: List[Tuple[str, Dict[str, float]]] = []

        for p_file in parquet_files:
            path = Path(p_file)
            if not path.is_file():
                continue

            table = pq.read_table(str(path), columns=["chunk_id", "sparse_weights_json"])
            chunk_ids = table["chunk_id"].to_pylist()
            sparse_jsons = table["sparse_weights_json"].to_pylist()

            for cid, raw_json in zip(chunk_ids, sparse_jsons):
                if raw_json:
                    try:
                        weights = json.loads(raw_json)
                        if isinstance(weights, dict):
                            records.append((cid, weights))
                    except Exception:
                        pass

        self.build_from_records(records)

    def search(
        self,
        query_sparse_weights: Optional[Dict[str, float]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Compute dot product: sum_{term in query} (query_weight * doc_weight).
        Returns:
            List of dicts: {"chunk_id": str, "sparse_score": float, "sparse_rank": int}
        """
        if not query_sparse_weights or not self.inverted_index:
            return []

        doc_scores: Dict[str, float] = defaultdict(float)

        for token, q_weight in query_sparse_weights.items():
            token_str = str(token)
            postings = self.inverted_index.get(token_str)
            if not postings:
                continue

            for chunk_id, d_weight in postings:
                doc_scores[chunk_id] += float(q_weight) * float(d_weight)

        if not doc_scores:
            return []

        # Sort by accumulated score descending
        sorted_candidates = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = sorted_candidates[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(top_candidates, 1):
            results.append({
                "chunk_id": chunk_id,
                "sparse_score": float(score),
                "sparse_rank": rank,
            })

        return results

    def save(self, index_dir: Union[str, Path]):
        """Persist inverted index to JSON file."""
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sparse_file = out_dir / "sparse_index.json"

        data = {
            "total_docs": self.total_docs,
            "indexed_chunk_ids": self.indexed_chunk_ids,
            "inverted_index": dict(self.inverted_index),
        }

        with open(sparse_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info(f"Saved SparseLexicalIndex ({len(self.inverted_index)} terms) -> {sparse_file}")

    def load(self, index_dir: Union[str, Path]):
        """Load inverted index from JSON file."""
        in_dir = Path(index_dir)
        sparse_file = in_dir / "sparse_index.json"

        if not sparse_file.is_file():
            raise FileNotFoundError(f"Sparse index file not found at: {sparse_file}")

        with open(sparse_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.total_docs = data.get("total_docs", 0)
        self.indexed_chunk_ids = data.get("indexed_chunk_ids", [])
        raw_inv = data.get("inverted_index", {})
        self.inverted_index = defaultdict(list, {k: [(item[0], float(item[1])) for item in v] for k, v in raw_inv.items()})

        logger.info(
            f"Loaded SparseLexicalIndex with {len(self.inverted_index)} terms across {self.total_docs} chunks from {sparse_file}"
        )
