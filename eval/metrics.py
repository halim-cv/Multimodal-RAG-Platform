"""
eval/metrics.py

Standard information-retrieval metrics for evaluating RAG retrieval quality.
All functions follow the same signature pattern for consistency.
"""

from typing import Any


def mrr_at_k(retrieved: list[Any], relevant: list[Any], k: int) -> float:
    """
    Mean Reciprocal Rank @ K.
    Returns 1/rank of the first relevant item in the top-K retrieved list.
    Higher is better (max = 1.0, min = 0.0).
    """
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: list[Any], relevant: list[Any], k: int) -> float:
    """
    Hit Rate @ K.
    1.0 if any relevant item appears in top-K, else 0.0.
    """
    return float(any(item in relevant for item in retrieved[:k]))


def precision_at_k(retrieved: list[Any], relevant: list[Any], k: int) -> float:
    """
    Precision @ K.
    Fraction of the top-K retrieved items that are relevant.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def recall_at_k(retrieved: list[Any], relevant: list[Any], k: int) -> float:
    """
    Recall @ K.
    Fraction of all relevant items that appear in the top-K results.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[Any], relevant: list[Any]) -> float:
    """
    Average Precision (AP).
    Area under the precision-recall curve for a single query.
    Used to compute Mean Average Precision (MAP) across queries.
    """
    hits    = 0
    ap_sum  = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits   += 1
            ap_sum += hits / rank
    return ap_sum / max(len(relevant), 1)


def compute_all(
    retrieved: list[Any],
    relevant:  list[Any],
    k:         int = 5,
) -> dict[str, float]:
    """
    Compute all metrics at once. Convenience wrapper.

    Returns:
        {"mrr": ..., "hit": ..., "precision": ..., "recall": ..., "ap": ...}
    """
    return {
        "mrr":       mrr_at_k(retrieved, relevant, k),
        "hit":       hit_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "recall":    recall_at_k(retrieved, relevant, k),
        "ap":        average_precision(retrieved, relevant),
    }
