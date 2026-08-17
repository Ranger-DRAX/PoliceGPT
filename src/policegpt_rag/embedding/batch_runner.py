"""
Batch runner to process and embed large collections of legal chunks efficiently.
"""

from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger
import pandas as pd
from pathlib import Path

from .embed import BGEM3Embedder, EmbeddingOutput
from ..preprocessing.chunker import LegalChunk


class BatchEmbeddingRunner:
    def __init__(self, embedder: BGEM3Embedder, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = batch_size

    def process_chunks(self, chunks: List[LegalChunk]) -> List[Dict[str, Any]]:
        """
        Embed a list of LegalChunks and return enriched records with dense/sparse embeddings.
        """
        logger.info(f"Generating embeddings for {len(chunks)} legal chunks (batch_size={self.batch_size})...")
        records: List[Dict[str, Any]] = []

        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding Batches"):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = [c.content for c in batch_chunks]

            embeddings: List[EmbeddingOutput] = self.embedder.encode_documents(batch_texts)

            for chunk, emb in zip(batch_chunks, embeddings):
                record = chunk.model_dump()
                record["dense_vector"] = emb.dense_vec
                record["sparse_weights"] = emb.sparse_weights
                records.append(record)

        return records

    def save_processed_parquet(self, records: List[Dict[str, Any]], output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_parquet(str(path), index=False)
        logger.info(f"Saved {len(records)} embedded chunks to: {path}")
