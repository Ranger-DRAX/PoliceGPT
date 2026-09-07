"""
Evaluation package for PoliceGPT RAG.
Provides standard retrieval and generation evaluation metrics.
"""

from .retrieval_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rank_correlation_spearman,
    rank_correlation_kendall,
)

__all__ = [
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "rank_correlation_spearman",
    "rank_correlation_kendall",
]
