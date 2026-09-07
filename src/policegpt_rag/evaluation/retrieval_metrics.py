"""
Information Retrieval (IR) and Reranking Evaluation Metrics.
Production-grade, zero-dependency implementations for statutory search evaluation.
Includes HitRate@K, MRR@K, NDCG@K, Precision@K, Recall@K, and Rank Correlations.
"""

from typing import List, Dict, Set, Union, Optional
import math


def hit_rate_at_k(
    relevant_ids: Union[Set[str], List[str]],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Hit Rate @ K: 1.0 if at least one relevant document appears in top-K, else 0.0.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    top_k_retrieved = retrieved_ids[:k]
    return 1.0 if any(cid in target_set for cid in top_k_retrieved) else 0.0


def mrr_at_k(
    relevant_ids: Union[Set[str], List[str]],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Mean Reciprocal Rank @ K: 1 / rank of the highest-ranked relevant document in top-K.
    Returns 0.0 if no relevant document appears in top-K.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in target_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    relevance_dict: Dict[str, float],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K with graded relevance.
    DCG = sum_{i=1}^K (2^rel - 1) / log2(i + 1)
    IDCG is computed by sorting relevance scores in descending order.
    """
    if not relevance_dict or not retrieved_ids or k <= 0:
        return 0.0

    # Calculate DCG@K
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_dict.get(cid, 0.0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1.0)

    # Calculate Ideal DCG@K
    ideal_scores = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_scores, start=1):
        if rel > 0:
            idcg += (2.0**rel - 1.0) / math.log2(rank + 1.0)

    if idcg <= 0.0:
        return 0.0

    return min(1.0, max(0.0, dcg / idcg))


def precision_at_k(
    relevant_ids: Union[Set[str], List[str]],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Precision @ K: Proportion of retrieved documents in top-K that are relevant.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    top_k_retrieved = retrieved_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in target_set)
    return hits / k


def recall_at_k(
    relevant_ids: Union[Set[str], List[str]],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Recall @ K: Proportion of all relevant documents that were retrieved in top-K.
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    if not target_set:
        return 0.0

    top_k_retrieved = retrieved_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in target_set)
    return hits / len(target_set)


def rank_correlation_spearman(
    ranks_a: List[str],
    ranks_b: List[str],
) -> float:
    """
    Spearman rank correlation between two candidate rankings on intersecting items.
    Returns 1.0 for identical ordering, -1.0 for exact inverse, ~0.0 for uncorrelated.
    Returns 1.0 if fewer than 2 common elements exist.
    """
    common_items = [item for item in ranks_a if item in ranks_b]
    n = len(common_items)
    if n < 2:
        return 1.0

    pos_a = {item: i for i, item in enumerate(ranks_a)}
    pos_b = {item: i for i, item in enumerate(ranks_b)}

    d_squared_sum = sum((pos_a[item] - pos_b[item]) ** 2 for item in common_items)
    denom = n * (n**2 - 1)
    if denom == 0:
        return 1.0

    rho = 1.0 - (6.0 * d_squared_sum) / denom
    return max(-1.0, min(1.0, rho))


def rank_correlation_kendall(
    ranks_a: List[str],
    ranks_b: List[str],
) -> float:
    """
    Kendall's Tau rank correlation coefficient between two rankings on intersecting items.
    Measures ratio of concordant pairs to discordant pairs.
    """
    common = [item for item in ranks_a if item in ranks_b]
    n = len(common)
    if n < 2:
        return 1.0

    pos_b = {item: i for i, item in enumerate(ranks_b)}
    # common is already ordered by ranks_a, so we check inversion count in ranks_b
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            order_in_b = pos_b[common[i]] - pos_b[common[j]]
            if order_in_b < 0:
                concordant += 1
            elif order_in_b > 0:
                discordant += 1

    total_pairs = n * (n - 1) / 2.0
    if total_pairs == 0:
        return 1.0

    tau = (concordant - discordant) / total_pairs
    return max(-1.0, min(1.0, tau))
