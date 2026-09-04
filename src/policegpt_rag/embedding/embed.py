"""
BGE-M3 Multilingual Embedding Engine for Legal Statutes and Regulations.
Optimized for low-VRAM constraints (GTX 1050 Ti 4GB / Intel i3 6th Gen / 8GB RAM).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

try:
    from policegpt_rag.preprocessing.chunker import LegalChunk
except ImportError:
    from src.policegpt_rag.preprocessing.chunker import LegalChunk


class ChunkEmbedding(BaseModel):
    """Embeddings and representation metadata for a single LegalChunk."""
    chunk_id: str
    doc_id: str
    dense_vector: List[float]
    sparse_weights: Optional[Dict[str, float]] = None
    embedding_model: str
    embedding_dim: int
    section_number: Optional[str] = None
    section_title: Optional[str] = None


class BGEM3Embedder:
    """
    Wrapper for BAAI/bge-m3 embedding model using FlagEmbedding.
    - Lazy model loading (zero overhead at import time).
    - Automatic CPU fallback with use_fp16=False.
    - Automatic CUDA OOM recovery via dynamic batch halving.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 4,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.requested_device = device.lower()
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.normalize_embeddings = normalize_embeddings
        self.oom_count = 0
        self._model = None

        # Resolve device and fp16 settings based on hardware availability
        self.device, self.use_fp16 = self._resolve_device_and_precision(
            self.requested_device, use_fp16
        )

    def _resolve_device_and_precision(self, requested_device: str, requested_fp16: bool):
        """Determine device and precision safely, applying GTX 1050 Ti memory optimizations."""
        if requested_device == "cuda":
            try:
                import os
                import torch
                if torch.cuda.is_available():
                    # Optimize CUDA memory allocation for 4GB VRAM cards
                    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    torch.backends.cudnn.benchmark = True
                    gpu_name = torch.cuda.get_device_name(0)
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(
                        f"CUDA active: {gpu_name} ({total_vram_gb:.1f} GB VRAM) | fp16={requested_fp16}"
                    )
                    return "cuda", requested_fp16
                else:
                    logger.warning(
                        "CUDA requested, but torch.cuda.is_available() is False. "
                        "To activate CUDA for your GTX 1050 Ti, install CUDA PyTorch via:\n"
                        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                        "Falling back to CPU with use_fp16=False."
                    )
                    return "cpu", False
            except ImportError:
                logger.warning("torch is not installed. Falling back to CPU.")
                return "cpu", False
        return "cpu", False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Explicitly or lazily load the BGEM3FlagModel."""
        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is not installed. Install via `pip install FlagEmbedding`."
            )

        logger.info(
            f"Loading BGE-M3 model ('{self.model_name}') on {self.device} (fp16={self.use_fp16})..."
        )
        self._model = BGEM3FlagModel(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        return self._model

    def embed_chunks(self, chunks: List[LegalChunk]) -> List[ChunkEmbedding]:
        """
        Embed a list of LegalChunk instances in batches, returning ChunkEmbedding records.
        Recovers dynamically from CUDA OOM errors by halving batch size.
        """
        if not chunks:
            return []

        model = self.load()
        results: List[ChunkEmbedding] = []
        i = 0

        while i < len(chunks):
            curr_batch = chunks[i : i + self.batch_size]
            texts = [c.content for c in curr_batch]

            try:
                import torch
                inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            except Exception:
                from contextlib import nullcontext
                inference_ctx = nullcontext()

            try:
                with inference_ctx:
                    outputs = model.encode(
                        texts,
                        batch_size=len(texts),
                        max_length=self.max_length,
                        return_dense=self.return_dense,
                        return_sparse=self.return_sparse,
                        return_colbert_vecs=self.return_colbert_vecs,
                    )
            except Exception as e:
                is_cuda_oom = False
                try:
                    import torch
                    if (
                        isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ()))
                        or "out of memory" in str(e).lower()
                    ):
                        is_cuda_oom = True
                except Exception:
                    pass

                if is_cuda_oom and self.device == "cuda":
                    self.oom_count += 1
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    new_batch_size = max(1, self.batch_size // 2)
                    if new_batch_size < self.batch_size:
                        logger.warning(
                            f"CUDA OutOfMemory encountered! Halving batch size from {self.batch_size} to {new_batch_size} and retrying batch."
                        )
                        self.batch_size = new_batch_size
                        continue
                    else:
                        logger.warning(
                            "CUDA OutOfMemory encountered at batch size 1! Falling back to CPU for remainder of document."
                        )
                        self.device = "cpu"
                        self.use_fp16 = False
                        self._model = None
                        model = self.load()
                        continue
                raise e

            # Parse dense and sparse outputs
            dense_vectors = outputs.get("dense_vecs", []) if isinstance(outputs, dict) else outputs
            sparse_weights = outputs.get("lexical_weights", None) if isinstance(outputs, dict) else None

            for idx, chunk in enumerate(curr_batch):
                dense = dense_vectors[idx]
                if hasattr(dense, "tolist"):
                    dense_list = dense.tolist()
                else:
                    dense_list = list(dense)

                sparse_dict = None
                if self.return_sparse and sparse_weights is not None and idx < len(sparse_weights):
                    raw_sparse = sparse_weights[idx]
                    if isinstance(raw_sparse, dict):
                        # Ensure keys are strings and values are floats
                        sparse_dict = {str(k): float(v) for k, v in raw_sparse.items()}

                embedding = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    dense_vector=dense_list,
                    sparse_weights=sparse_dict,
                    embedding_model=self.model_name,
                    embedding_dim=len(dense_list),
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                )
                results.append(embedding)

            i += len(curr_batch)

        return results
