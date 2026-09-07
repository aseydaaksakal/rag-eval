"""Retrieval metrics.

Every function takes ranked chunk ids and the ids that should have been found,
and returns a score in ``[0, 1]``. Rank order matters: pass the retriever's
output in the order it produced, not sorted or deduplicated.

Which one to watch depends on what the retriever feeds:

* **recall@k** — the ceiling on your answer quality. A passage that was never
  retrieved cannot be cited. This is the one to protect with a threshold.
* **mrr** — how far down the list the first useful passage sits. Falls when
  reranking degrades even though recall holds.
* **ndcg@k** — position-weighted, and the most sensitive to reordering.
* **precision@k** — how much of the context window is wasted. Matters when you
  pay per token or when the model gets distracted by near-misses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _top_k(retrieved: Sequence[str], k: int | None) -> list[str]:
    return list(retrieved) if k is None else list(retrieved)[:k]


def hit_rate(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """1.0 if any relevant chunk is in the top *k*, else 0.0.

    The bluntest useful signal: did retrieval find anything at all? Averaged
    over a dataset it reads as "the share of questions we had a chance at".
    """
    if not relevant:
        return 0.0
    return 1.0 if set(_top_k(retrieved, k)) & set(relevant) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Share of relevant chunks that appear in the top *k*.

    >>> recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3)
    0.666...
    """
    if not relevant:
        return 0.0
    found = set(_top_k(retrieved, k)) & set(relevant)
    return len(found) / len(set(relevant))


def precision_at_k(
    retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None
) -> float:
    """Share of the top *k* that is relevant.

    Divides by *k* rather than by the number retrieved, so a retriever that
    returns two chunks when asked for five is not rewarded for it.
    """
    top = _top_k(retrieved, k)
    denominator = k if k is not None else len(top)
    if not denominator:
        return 0.0
    return len(set(top) & set(relevant)) / denominator


def mrr(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant chunk.

    1.0 when the first result is relevant, 0.5 when the second is, and so on.
    0.0 when none of the top *k* is relevant.
    """
    wanted = set(relevant)
    for index, chunk_id in enumerate(_top_k(retrieved, k), start=1):
        if chunk_id in wanted:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Each relevant chunk contributes ``1 / log2(rank + 1)``, normalised against
    the best possible ordering. Unlike recall it notices reordering, which is
    what regresses when you change a reranker.
    """
    if not relevant:
        return 0.0
    wanted = set(relevant)
    top = _top_k(retrieved, k)

    gain = sum(
        1.0 / math.log2(index + 1)
        for index, chunk_id in enumerate(top, start=1)
        if chunk_id in wanted
    )
    ideal_depth = min(len(wanted), len(top)) if top else 0
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_depth + 1))
    return gain / ideal if ideal else 0.0


def context_waste(
    retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None
) -> float:
    """Share of retrieved chunks that are not relevant.

    ``1 - precision``, reported directly because it is the number that
    translates into tokens paid for and attention spent. Lower is better, so
    exclude it from "higher is better" threshold checks or invert it first.
    """
    top = _top_k(retrieved, k)
    if not top or not relevant:
        # Undefined without a ground truth. Zero rather than one, so an example
        # with no relevant_ids cannot masquerade as total waste.
        return 0.0
    return 1.0 - (len(set(top) & set(relevant)) / len(top))
