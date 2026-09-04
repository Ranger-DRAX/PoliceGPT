"""
PoliceGPT: Specialized Legal Document Ingestion & Structure-Aware Chunking Engine for Bangladesh Police.
"""

from .pipeline import PoliceGPTChunkingPipeline, DocumentProcessingResult

__version__ = "0.2.0"
__all__ = ["PoliceGPTChunkingPipeline", "DocumentProcessingResult"]
