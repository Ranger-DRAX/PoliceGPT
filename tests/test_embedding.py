"""
Unit tests for BGE-M3 Embedder, Batching, Hardware Scoping, and Resume Logic.
All FlagEmbedding / torch GPU calls are mocked — no GPU, download, or network required.
"""

import sys
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure tests can import policegpt_rag or src.policegpt_rag
try:
    from policegpt_rag.preprocessing.chunker import LegalChunk
    from policegpt_rag.embedding.embed import BGEM3Embedder, ChunkEmbedding
    from policegpt_rag.embedding.batch_runner import EmbeddingBatchRunner, EMBEDDING_PARQUET_SCHEMA
except ImportError:
    from src.policegpt_rag.preprocessing.chunker import LegalChunk
    from src.policegpt_rag.embedding.embed import BGEM3Embedder, ChunkEmbedding
    from src.policegpt_rag.embedding.batch_runner import EmbeddingBatchRunner, EMBEDDING_PARQUET_SCHEMA


def _create_dummy_chunks(num_chunks: int, doc_id: str = "test_doc") -> list:
    """Helper to create dummy LegalChunk objects."""
    chunks = []
    for i in range(num_chunks):
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                act_name_en="Test Act",
                section_number=str(i + 1),
                section_title=f"Title {i + 1}",
                content=f"Legal content for section {i + 1} of test document.",
                page_numbers=[i + 1],
            )
        )
    return chunks


def test_cpu_fallback_forces_fp16_false():
    """Verify CPU mode unconditionally disables fp16."""
    embedder = BGEM3Embedder(device="cpu", use_fp16=True)
    assert embedder.device == "cpu"
    assert embedder.use_fp16 is False


def test_cuda_fallback_to_cpu_when_unavailable():
    """Verify requested CUDA falls back to CPU when torch.cuda.is_available() is False."""
    with patch("torch.cuda.is_available", return_value=False):
        embedder = BGEM3Embedder(device="cuda", use_fp16=True)
        assert embedder.device == "cpu"
        assert embedder.use_fp16 is False


def test_cuda_enabled_when_available():
    """Verify requested CUDA is active with FP16 when torch.cuda.is_available() is True."""
    mock_prop = MagicMock()
    mock_prop.total_memory = 4 * 1024**3
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1050 Ti"), \
         patch("torch.cuda.get_device_properties", return_value=mock_prop):
        embedder = BGEM3Embedder(device="cuda", use_fp16=True)
        assert embedder.device == "cuda"
        assert embedder.use_fp16 is True


def test_batching_splits_into_correct_sizes():
    """Verify embed_chunks splits 10 chunks into batches of [3, 3, 3, 1] for batch_size=3."""
    chunks = _create_dummy_chunks(10)
    embedder = BGEM3Embedder(device="cpu", batch_size=3)

    recorded_batch_sizes = []

    def mock_encode(texts, **kwargs):
        recorded_batch_sizes.append(len(texts))
        return {
            "dense_vecs": [[0.1] * 1024 for _ in texts],
            "lexical_weights": [{"101": 0.5} for _ in texts],
        }

    mock_model = MagicMock()
    mock_model.encode = MagicMock(side_effect=mock_encode)
    embedder._model = mock_model

    results = embedder.embed_chunks(chunks)

    assert recorded_batch_sizes == [3, 3, 3, 1]
    assert len(results) == 10
    assert results[0].embedding_dim == 1024
    assert results[0].chunk_id == "test_doc_chunk_0"
    assert results[0].sparse_weights == {"101": 0.5}


def test_resume_logic_skips_existing_chunk_ids(tmp_path):
    """Verify that chunks already in Parquet are skipped on subsequent runs."""
    doc_id = "statute_1861"
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{doc_id}_embeddings.parquet"

    # 1. Pre-create a parquet with chunk_0 and chunk_1
    pre_existing_table = pa.Table.from_arrays(
        [
            pa.array(["statute_1861_chunk_0", "statute_1861_chunk_1"], type=pa.string()),
            pa.array([doc_id, doc_id], type=pa.string()),
            pa.array([[0.1] * 1024, [0.2] * 1024], type=pa.list_(pa.float32())),
            pa.array([None, None], type=pa.string()),
            pa.array(["1", "2"], type=pa.string()),
            pa.array(["T1", "T2"], type=pa.string()),
            pa.array(["BAAI/bge-m3", "BAAI/bge-m3"], type=pa.string()),
        ],
        schema=EMBEDDING_PARQUET_SCHEMA,
    )
    pq.write_table(pre_existing_table, str(parquet_path))

    # 2. Prepare chunks JSON file with 4 chunks (0, 1, 2, 3)
    chunks = _create_dummy_chunks(4, doc_id=doc_id)
    json_path = tmp_path / f"{doc_id}_chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f)

    # 3. Mock embedder
    embedder = BGEM3Embedder(device="cpu", batch_size=2)
    embedded_chunk_ids = []

    def mock_encode(texts, **kwargs):
        return {
            "dense_vecs": [[0.5] * 1024 for _ in texts],
            "lexical_weights": None,
        }

    mock_model = MagicMock()
    mock_model.encode = MagicMock(side_effect=mock_encode)
    embedder._model = mock_model

    # 4. Run batch runner
    runner = EmbeddingBatchRunner(embedder=embedder, output_dir=output_dir)
    report = runner.process_document(json_path)

    # Chunks 0 and 1 should be skipped; only chunks 2 and 3 should be newly embedded
    assert report["total_chunks"] == 4
    assert report["skipped_chunks"] == 2
    assert report["newly_embedded"] == 2

    # Verify final parquet has all 4 rows
    final_table = pq.read_table(str(parquet_path))
    assert final_table.num_rows == 4
    final_ids = final_table["chunk_id"].to_pylist()
    assert final_ids == [
        "statute_1861_chunk_0",
        "statute_1861_chunk_1",
        "statute_1861_chunk_2",
        "statute_1861_chunk_3",
    ]


def test_cuda_oom_recovery_halves_batch_size():
    """Verify CUDA OOM catches OutOfMemoryError, halves batch size, and retries."""
    chunks = _create_dummy_chunks(4)
    embedder = BGEM3Embedder(device="cuda", batch_size=4)
    # Ensure embedder.device is set to cuda for testing OOM logic
    embedder.device = "cuda"

    class FakeCUDAOutOfMemory(Exception):
        pass

    first_call = True

    def mock_encode(texts, **kwargs):
        nonlocal first_call
        if first_call and len(texts) == 4:
            first_call = False
            raise FakeCUDAOutOfMemory("CUDA out of memory. Tried to allocate 512MiB")
        return {
            "dense_vecs": [[0.1] * 1024 for _ in texts],
            "lexical_weights": None,
        }

    mock_model = MagicMock()
    mock_model.encode = MagicMock(side_effect=mock_encode)
    embedder._model = mock_model

    with patch("torch.cuda.empty_cache") as mock_empty_cache:
        results = embedder.embed_chunks(chunks)
        assert embedder.oom_count == 1
        assert embedder.batch_size == 2
        assert len(results) == 4
        assert mock_empty_cache.called
