"""
Batch embedding runner with PyArrow Parquet persistence and resume capability.
Processes LegalChunk JSON files and outputs FAISS-ready Parquet tables.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Callable
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
import yaml

try:
    from policegpt_rag.preprocessing.chunker import LegalChunk
    from policegpt_rag.embedding.embed import BGEM3Embedder, ChunkEmbedding
except ImportError:
    from src.policegpt_rag.preprocessing.chunker import LegalChunk
    from src.policegpt_rag.embedding.embed import BGEM3Embedder, ChunkEmbedding


# PyArrow schema matching downstream FAISS and Hybrid retrieval requirements
EMBEDDING_PARQUET_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("dense_vector", pa.list_(pa.float32())),
    ("sparse_weights_json", pa.string()),
    ("section_number", pa.string()),
    ("section_title", pa.string()),
    ("embedding_model", pa.string()),
])


class EmbeddingBatchRunner:
    """
    Streams LegalChunk files through BGEM3Embedder, persisting output to Parquet.
    Supports idempotent runs by checking existing chunk_ids in destination Parquet.
    """

    def __init__(
        self,
        embedder: Optional[BGEM3Embedder] = None,
        output_dir: Union[str, Path] = "data/processed/embeddings",
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = self._create_embedder_from_config(config_path)

    def _create_embedder_from_config(self, config_path: Optional[Union[str, Path]]) -> BGEM3Embedder:
        cfg = {}
        target_cfg = Path(config_path) if config_path else Path("configs/embedding.yaml")
        if target_cfg.is_file():
            try:
                with open(target_cfg, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not read config from {target_cfg}: {e}. Using defaults.")

        paths_cfg = cfg.get("paths", {})
        if "output_dir" in paths_cfg:
            self.output_dir = Path(paths_cfg["output_dir"])
            self.output_dir.mkdir(parents=True, exist_ok=True)

        return BGEM3Embedder(
            model_name=cfg.get("model_name", "BAAI/bge-m3"),
            device=cfg.get("device", "cuda"),
            use_fp16=cfg.get("use_fp16", True),
            batch_size=cfg.get("batch_size", 4),
            max_length=cfg.get("max_length", 512),
            return_dense=cfg.get("return_dense", True),
            return_sparse=cfg.get("return_sparse", True),
            return_colbert_vecs=cfg.get("return_colbert_vecs", False),
            normalize_embeddings=cfg.get("normalize_embeddings", True),
        )

    def get_existing_chunk_ids(self, parquet_path: Path) -> Set[str]:
        """Read set of chunk_ids already embedded in an existing Parquet file."""
        if not parquet_path.is_file():
            return set()

        try:
            table = pq.read_table(str(parquet_path), columns=["chunk_id"])
            chunk_ids = table["chunk_id"].to_pylist()
            return set(chunk_ids)
        except Exception as e:
            logger.warning(f"Failed to read existing Parquet at {parquet_path}: {e}. Re-embedding.")
            return set()

    def process_document(
        self,
        chunks_path_or_doc_id: Union[str, Path],
        stream_batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document's chunks into embeddings.
        Args:
            chunks_path_or_doc_id: Path to <doc_id>_chunks.json or doc_id string.
            stream_batch_size: Number of chunks to process before appending to memory/disk.
            progress_callback: Optional callback fn(processed_count, total_count).
        Returns:
            Dict summary of processing run.
        """
        path = Path(chunks_path_or_doc_id)
        if not path.is_file():
            # Try finding it in data/processed/<doc_id>_chunks.json
            cand = Path("data/processed") / f"{chunks_path_or_doc_id}_chunks.json"
            if cand.is_file():
                path = cand
            else:
                raise FileNotFoundError(f"Chunks JSON not found at {path} or {cand}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list of chunks in {path}, got {type(data)}")

        chunks = [LegalChunk(**item) for item in data]
        if not chunks:
            logger.warning(f"No chunks found in {path.name}.")
            return {
                "doc_id": path.stem.replace("_chunks", ""),
                "source_file": path.name,
                "total_chunks": 0,
                "newly_embedded": 0,
                "skipped_chunks": 0,
                "embedding_dim": 0,
                "output_path": "",
            }

        doc_id = chunks[0].doc_id
        parquet_path = self.output_dir / f"{doc_id}_embeddings.parquet"
        existing_ids = self.get_existing_chunk_ids(parquet_path)

        chunks_to_embed = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped_count = len(chunks) - len(chunks_to_embed)

        logger.info(
            f"Document '{doc_id}': {len(chunks)} total chunks, "
            f"{skipped_count} already embedded, {len(chunks_to_embed)} to embed."
        )

        embedding_dim = 1024
        new_embeddings: List[ChunkEmbedding] = []

        if chunks_to_embed:
            total_to_embed = len(chunks_to_embed)
            for i in range(0, total_to_embed, stream_batch_size):
                sub_chunks = chunks_to_embed[i : i + stream_batch_size]
                batch_embeddings = self.embedder.embed_chunks(sub_chunks)
                new_embeddings.extend(batch_embeddings)

                if batch_embeddings:
                    embedding_dim = batch_embeddings[0].embedding_dim

                if progress_callback:
                    progress_callback(len(new_embeddings), total_to_embed)

            # Convert new embeddings to PyArrow Table
            new_table = self._embeddings_to_table(new_embeddings)

            # If existing parquet exists, merge tables
            if parquet_path.is_file():
                try:
                    existing_table = pq.read_table(str(parquet_path))
                    final_table = pa.concat_tables([existing_table, new_table])
                except Exception as e:
                    logger.warning(f"Error merging with existing parquet: {e}. Overwriting with new.")
                    final_table = new_table
            else:
                final_table = new_table

            pq.write_table(final_table, str(parquet_path), compression="snappy")
            logger.info(f"Saved {final_table.num_rows} total embeddings to {parquet_path}")

        elif parquet_path.is_file():
            # Get dimension from existing parquet
            try:
                sample_tbl = pq.read_table(str(parquet_path))
                if sample_tbl.num_rows > 0:
                    sample_vec = sample_tbl["dense_vector"][0].as_py()
                    embedding_dim = len(sample_vec)
            except Exception:
                pass

        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "total_chunks": len(chunks),
            "newly_embedded": len(chunks_to_embed),
            "skipped_chunks": skipped_count,
            "embedding_dim": embedding_dim,
            "output_path": str(parquet_path.resolve()),
            "oom_count": self.embedder.oom_count,
            "final_batch_size": self.embedder.batch_size,
        }

    def _embeddings_to_table(self, embeddings: List[ChunkEmbedding]) -> pa.Table:
        """Convert a list of ChunkEmbedding objects into a typed PyArrow Table."""
        chunk_ids = []
        doc_ids = []
        dense_vectors = []
        sparse_weights_jsons = []
        section_numbers = []
        section_titles = []
        models = []

        for e in embeddings:
            chunk_ids.append(e.chunk_id)
            doc_ids.append(e.doc_id)
            dense_vectors.append(e.dense_vector)
            sparse_weights_jsons.append(
                json.dumps(e.sparse_weights, ensure_ascii=False) if e.sparse_weights else None
            )
            section_numbers.append(e.section_number or "")
            section_titles.append(e.section_title or "")
            models.append(e.embedding_model)

        return pa.Table.from_arrays(
            [
                pa.array(chunk_ids, type=pa.string()),
                pa.array(doc_ids, type=pa.string()),
                pa.array(dense_vectors, type=pa.list_(pa.float32())),
                pa.array(sparse_weights_jsons, type=pa.string()),
                pa.array(section_numbers, type=pa.string()),
                pa.array(section_titles, type=pa.string()),
                pa.array(models, type=pa.string()),
            ],
            schema=EMBEDDING_PARQUET_SCHEMA,
        )
