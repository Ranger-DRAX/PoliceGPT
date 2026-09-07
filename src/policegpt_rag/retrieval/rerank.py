"""
Cross-Encoder Reranker using BAAI/bge-reranker models.
Hardened for low-resource environments (GTX 1050 Ti 4GB / Intel i3 6th Gen / 8GB RAM).
Features:
  - Auto-disable on non-CUDA systems (prevents CPU freezes).
  - VRAM budget threshold gating (prevents CUDA OOM).
  - Runtime CUDA OOM trap with automatic cache purge & RRF fallback.
  - Micro-batching to smooth memory allocation spikes.
  - Text length truncation guards.
  - Explicit memory teardown (`unload()`) and diagnostic status inspection.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import os


class BGEReranker:
    """
    Cross-Encoder Reranker using FlagReranker (BAAI/bge-reranker-base or large).
    Lazy-loaded to prevent consuming VRAM/RAM until explicitly needed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        use_fp16: bool = True,
        auto_disable_if_no_cuda: bool = True,
        min_vram_mb: int = 1200,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.requested_device = device.lower()
        self.use_fp16 = use_fp16
        self.auto_disable_if_no_cuda = auto_disable_if_no_cuda
        self.min_vram_mb = min_vram_mb
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self._model = None
        self.is_enabled = True

        # Resolve device, VRAM budget, and safety toggles
        self.device, self.use_fp16, self.is_enabled = self._resolve_device_and_safety()

    def _get_vram_info(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (free_vram_mb, total_vram_mb) or (None, None) if CUDA is unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                return free_b / (1024 * 1024), total_b / (1024 * 1024)
        except Exception:
            pass
        return None, None

    def _resolve_device_and_safety(self) -> Tuple[str, bool, bool]:
        """
        Determine execution device and determine if reranker can run safely.
        If CUDA is unavailable and auto_disable_if_no_cuda is True, disable reranker.
        If VRAM is below threshold, auto-disable to prevent OOM crash.
        """
        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        # Case 1: Device requested is CUDA or auto
        if self.requested_device in ("cuda", "auto"):
            if not has_cuda:
                if self.auto_disable_if_no_cuda:
                    logger.warning(
                        "CUDA is not available. Cross-Encoder reranker auto-disabled "
                        "(satisfies 'disable if cuda!=gpu' policy to prevent CPU latency freeze). "
                        "Passing through RRF candidates."
                    )
                    return "cpu", False, False
                else:
                    logger.info("CUDA not available. Falling back to CPU execution.")
                    return "cpu", False, True

            # CUDA is available; check VRAM budget
            free_mb, total_mb = self._get_vram_info()
            if free_mb is not None and free_mb < self.min_vram_mb:
                logger.warning(
                    f"Free VRAM ({free_mb:.0f} MB) is below safe threshold ({self.min_vram_mb} MB). "
                    f"Auto-disabling BGEReranker to protect against CUDA OOM. "
                    f"Using pure RRF fusion results."
                )
                return "cuda", self.use_fp16, False

            gpu_name = torch.cuda.get_device_name(0) if has_cuda else "Unknown GPU"
            logger.info(
                f"BGEReranker configured on CUDA: {gpu_name} "
                f"({free_mb:.0f} MB free / {total_mb:.0f} MB total) | fp16={self.use_fp16}"
            )
            return "cuda", self.use_fp16, True

        # Case 2: Explicitly requested CPU
        if self.requested_device == "cpu":
            if self.auto_disable_if_no_cuda:
                logger.warning(
                    "Device explicitly configured to CPU with auto_disable_if_no_cuda=True. "
                    "Disabling reranker to avoid CPU latency penalty on low-spec host."
                )
                return "cpu", False, False
            return "cpu", False, True

        return "cpu", False, True

    def load(self):
        """Lazy load the FlagReranker instance."""
        if not self.is_enabled:
            return None

        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagReranker requires FlagEmbedding. Install via `pip install FlagEmbedding`."
            )

        logger.info(f"Loading FlagReranker ('{self.model_name}') on {self.device} (fp16={self.use_fp16})...")
        self._model = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        return self._model

    def unload(self):
        """Explicitly unload model and release CUDA memory."""
        if self._model is not None:
            logger.info("Unloading FlagReranker from memory.")
            del self._model
            self._model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and status metrics for observability."""
        free_mb, total_mb = self._get_vram_info()
        return {
            "model_name": self.model_name,
            "is_enabled": self.is_enabled,
            "is_loaded": self._model is not None,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "min_vram_mb": self.min_vram_mb,
            "auto_disable_if_no_cuda": self.auto_disable_if_no_cuda,
            "vram_free_mb": round(free_mb, 1) if free_mb is not None else None,
            "vram_total_mb": round(total_mb, 1) if total_mb is not None else None,
        }

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score (query, document) pairs using the cross-encoder with micro-batching
        and defensive OOM recovery.
        """
        if not candidates or top_k <= 0:
            return []

        # If disabled by policy (e.g. no CUDA or low VRAM), pass through candidates directly
        if not self.is_enabled:
            return candidates[:top_k]

        try:
            model = self.load()
            if model is None:
                return candidates[:top_k]

            # Build text pairs with length truncation guard
            pairs = []
            for cand in candidates:
                meta = cand.get("metadata", {})
                content = meta.get("content") or cand.get("content", "")
                if len(content) > self.max_length * 4:  # Rough char cap for max_length tokens
                    content = content[: self.max_length * 4]
                pairs.append([query, content])

            # Micro-batched cross-encoder inference
            scores: List[float] = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                batch_scores = model.compute_score(batch_pairs, normalize=True)
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

            reranked = []
            for cand, score in zip(candidates, scores):
                cand_copy = dict(cand)
                cand_copy["rerank_score"] = float(score)
                reranked.append(cand_copy)

            # Sort descending by rerank score, breaking ties by original RRF score
            reranked.sort(
                key=lambda x: (
                    x.get("rerank_score", -999.0),
                    x.get("rrf_score", 0.0),
                ),
                reverse=True,
            )
            return reranked[:top_k]

        except Exception as e:
            # Trap CUDA OOM or runtime model error without failing user query
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda" in err_msg:
                logger.error(
                    f"CUDA error during reranking: {e}. "
                    f"Purging cache, disabling reranker, and returning original RRF candidates."
                )
                self.unload()
                self.is_enabled = False
            else:
                logger.error(f"Unexpected error in reranking: {e}. Falling back to RRF.")

            return candidates[:top_k]
