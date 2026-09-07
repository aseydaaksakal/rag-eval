"""Metrics, and the registry that maps their names."""

from .generation import (
    answer_correctness,
    answer_relevance,
    groundedness,
    hallucination,
    refusal,
)
from .registry import DEFAULT_METRICS, Metric, available, build, build_all
from .retrieval import (
    context_waste,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "Metric",
    "build",
    "build_all",
    "available",
    "DEFAULT_METRICS",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "context_waste",
    "groundedness",
    "hallucination",
    "answer_relevance",
    "answer_correctness",
    "refusal",
]
