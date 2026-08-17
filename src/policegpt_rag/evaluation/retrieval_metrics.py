"""
Retrieval evaluation metrics: HitRate@k, Recall@k, and Mean Reciprocal Rank (MRR@k).
"""

from typing import List, Dict, Any, Set
from pydantic import BaseModel


class RetrievalMetricScores(BaseModel):
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    mrr_at_10: float
    total_queries: int


class RetrievalEvaluator:
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = sorted(k_values)

    def evaluate_retrievals(
        self, gold_pairs: List[Dict[str, Any]], retrieved_results: List[List[str]]
    ) -> RetrievalMetricScores:
        """
        gold_pairs: List of dicts with 'relevant_doc_ids' (List[str])
        retrieved_results: List of retrieved doc_ids lists per query.
        """
        assert len(gold_pairs) == len(retrieved_results), "Mismatched queries and retrieval results."

        hits = {k: 0 for k in self.k_values}
        reciprocal_ranks = []
        total = len(gold_pairs)

        if total == 0:
            return RetrievalMetricScores(
                hit_rate_at_1=0.0,
                hit_rate_at_3=0.0,
                hit_rate_at_5=0.0,
                hit_rate_at_10=0.0,
                mrr_at_10=0.0,
                total_queries=0,
            )

        for gold, retrieved in zip(gold_pairs, retrieved_results):
            gold_set: Set[str] = set(gold.get("relevant_doc_ids", []))
            
            # Hit Rate at K
            for k in self.k_values:
                top_k_docs = set(retrieved[:k])
                if len(gold_set.intersection(top_k_docs)) > 0:
                    hits[k] += 1

            # Reciprocal Rank (MRR)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved[:10]):
                if doc_id in gold_set:
                    rr = 1.0 / (rank + 1)
                    break
            reciprocal_ranks.append(rr)

        return RetrievalMetricScores(
            hit_rate_at_1=hits.get(1, 0) / total,
            hit_rate_at_3=hits.get(3, 0) / total,
            hit_rate_at_5=hits.get(5, 0) / total,
            hit_rate_at_10=hits.get(10, 0) / total,
            mrr_at_10=sum(reciprocal_ranks) / total,
            total_queries=total,
        )
