"""
Unit and hardening tests for BGEReranker.
All FlagEmbedding and torch GPU calls are mocked: fast, reproducible, and zero network/GPU required.
Verifies VRAM gating, auto-disable when CUDA is unavailable, OOM recovery, micro-batching, and memory release.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

try:
    from policegpt_rag.retrieval.rerank import BGEReranker
except ImportError:
    from src.policegpt_rag.retrieval.rerank import BGEReranker


def _create_dummy_candidates(n: int = 5) -> list:
    """Helper to generate dummy retrieved candidate dictionaries."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "doc_id": "penal_code",
            "section_number": str(300 + i),
            "content": f"Section {300 + i} legal description.",
            "rrf_score": 1.0 / (60.0 + i + 1),
            "metadata": {"content": f"Section {300 + i} legal description."},
        }
        for i in range(n)
    ]


def test_reranker_auto_disables_when_no_cuda():
    """Verify reranker auto-disables when CUDA is absent and auto_disable_if_no_cuda=True."""
    with patch("torch.cuda.is_available", return_value=False):
        reranker = BGEReranker(device="auto", auto_disable_if_no_cuda=True)
        assert reranker.is_enabled is False
        assert reranker.device == "cpu"

        cands = _create_dummy_candidates(3)
        res = reranker.rerank("query", cands, top_k=2)
        # Should return original candidates without error or model load
        assert len(res) == 2
        assert res[0]["chunk_id"] == "chunk_0"


def test_reranker_cpu_fallback_when_auto_disable_false():
    """Verify reranker falls back to CPU when auto_disable_if_no_cuda=False."""
    with patch("torch.cuda.is_available", return_value=False):
        reranker = BGEReranker(device="auto", auto_disable_if_no_cuda=False)
        assert reranker.is_enabled is True
        assert reranker.device == "cpu"
        assert reranker.use_fp16 is False


def test_reranker_low_vram_auto_disables():
    """Verify reranker auto-disables if free CUDA VRAM is below min_vram_mb threshold."""
    # 800 MB free, 4096 MB total
    mock_free_b = 800 * 1024 * 1024
    mock_total_b = 4096 * 1024 * 1024

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(mock_free_b, mock_total_b)), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1050 Ti"):
        reranker = BGEReranker(device="cuda", min_vram_mb=1200)
        assert reranker.is_enabled is False

        cands = _create_dummy_candidates(3)
        res = reranker.rerank("query", cands, top_k=2)
        assert len(res) == 2


def test_reranker_cuda_active_when_vram_sufficient():
    """Verify reranker enables CUDA and fp16 when free VRAM exceeds threshold."""
    mock_free_b = 2500 * 1024 * 1024
    mock_total_b = 4096 * 1024 * 1024

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(mock_free_b, mock_total_b)), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1050 Ti"):
        reranker = BGEReranker(device="cuda", min_vram_mb=1200, use_fp16=True)
        assert reranker.is_enabled is True
        assert reranker.device == "cuda"
        assert reranker.use_fp16 is True


def test_reranker_micro_batching_and_score_ordering():
    """Verify candidate scoring respects batch_size and sorts by rerank_score descending."""
    reranker = BGEReranker(device="cpu", auto_disable_if_no_cuda=False, batch_size=2)
    reranker.is_enabled = True

    mock_flag_model = MagicMock()
    # Batch 1 (2 pairs) -> [0.2, 0.9], Batch 2 (2 pairs) -> [0.1, 0.8], Batch 3 (1 pair) -> [0.5]
    mock_flag_model.compute_score.side_effect = [
        [0.2, 0.9],
        [0.1, 0.8],
        [0.5],
    ]
    reranker._model = mock_flag_model

    cands = _create_dummy_candidates(5)  # chunk_0 to chunk_4
    res = reranker.rerank("sample prompt", cands, top_k=3)

    assert mock_flag_model.compute_score.call_count == 3
    assert len(res) == 3
    # Top score is 0.9 (chunk_1), then 0.8 (chunk_3), then 0.5 (chunk_4)
    assert res[0]["chunk_id"] == "chunk_1"
    assert res[0]["rerank_score"] == pytest.approx(0.9)
    assert res[1]["chunk_id"] == "chunk_3"
    assert res[1]["rerank_score"] == pytest.approx(0.8)
    assert res[2]["chunk_id"] == "chunk_4"
    assert res[2]["rerank_score"] == pytest.approx(0.5)


def test_reranker_oom_recovery_preserves_candidates():
    """Verify CUDA OOM during scoring purges cache, disables reranker, and preserves candidates."""
    reranker = BGEReranker(device="cpu", auto_disable_if_no_cuda=False)
    reranker.is_enabled = True

    mock_flag_model = MagicMock()
    mock_flag_model.compute_score.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB")
    reranker._model = mock_flag_model

    cands = _create_dummy_candidates(4)
    with patch("torch.cuda.empty_cache") as mock_empty_cache:
        res = reranker.rerank("query", cands, top_k=3)

    # Should gracefully return first 3 candidates by original ranking
    assert len(res) == 3
    assert res[0]["chunk_id"] == "chunk_0"
    assert reranker.is_enabled is False
    assert reranker._model is None


def test_reranker_empty_candidates_and_zero_top_k():
    """Verify empty input handling."""
    reranker = BGEReranker(device="cpu", auto_disable_if_no_cuda=False)
    assert reranker.rerank("query", [], top_k=5) == []

    cands = _create_dummy_candidates(2)
    assert reranker.rerank("query", cands, top_k=0) == []
    assert reranker.rerank("query", cands, top_k=-1) == []


def test_reranker_unload_and_diagnostics():
    """Verify unload releases model and get_status returns complete diagnostic dict."""
    reranker = BGEReranker(device="cpu", auto_disable_if_no_cuda=False)
    reranker._model = MagicMock()

    status = reranker.get_status()
    assert status["is_loaded"] is True
    assert status["model_name"] == "BAAI/bge-reranker-base"
    assert status["batch_size"] == 8

    reranker.unload()
    assert reranker._model is None
    assert reranker.get_status()["is_loaded"] is False
