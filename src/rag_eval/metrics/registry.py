"""Name -> metric, so datasets, configs and the CLI can all say `"recall@5"`.

A metric is a small object that knows how to score one example and whether a
higher number is better. The registry keeps the string form stable, which is
what makes a threshold config readable a year later.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..judges.base import Judge
from ..types import Example, Prediction
from . import generation, retrieval

#: Metrics computed when the caller does not name any.
DEFAULT_METRICS = ("recall@5", "mrr@5", "ndcg@5", "groundedness", "hallucination")

_AT_K = re.compile(r"^([a-z_]+)(?:@(\d+))?$")


@dataclass(slots=True)
class Metric:
    """One scoring rule.

    Args:
        name: Canonical name, e.g. ``"recall@5"``.
        score: Callable receiving the example, the prediction and the judge.
        higher_is_better: False for ``hallucination``, ``context_waste`` and
            ``refusal_rate``. The regression checker reads this, so getting it
            wrong silently inverts a gate.
        needs_judge: True for anything that scores text.
        needs_reference: True when the example must carry a reference answer.
        needs_relevant: True when the example must carry ``relevant_ids``.

    Examples missing what a metric needs are **skipped**, not scored zero.
    Scoring them zero silently drags the aggregate down in proportion to how
    incompletely the dataset is annotated, which looks exactly like a
    regression and is not one.
    """

    name: str
    score: Callable[[Example, Prediction, Judge], float]
    higher_is_better: bool = True
    needs_judge: bool = False
    needs_reference: bool = False
    needs_relevant: bool = False

    def applies_to(self, example: Example) -> bool:
        if self.needs_reference and not example.reference_answer:
            return False
        return not (self.needs_relevant and not example.relevant_ids)


def _retrieval_metric(name: str, func: Callable[..., float], k: int | None,
                      higher_is_better: bool = True) -> Metric:
    def score(example: Example, prediction: Prediction, judge: Judge) -> float:
        return func(prediction.chunk_ids, example.relevant_ids, k)

    return Metric(
        name=name,
        score=score,
        higher_is_better=higher_is_better,
        needs_relevant=True,
    )


def _contexts(example: Example, prediction: Prediction) -> list[str]:
    """Prefer the passages the pipeline actually retrieved.

    Falling back to ``reference_contexts`` lets you score a generator in
    isolation, with retrieval held fixed.
    """
    return prediction.contexts or list(example.reference_contexts)


_RETRIEVAL_FUNCS: dict[str, tuple[Callable[..., float], bool]] = {
    "recall": (retrieval.recall_at_k, True),
    "precision": (retrieval.precision_at_k, True),
    "mrr": (retrieval.mrr, True),
    "ndcg": (retrieval.ndcg_at_k, True),
    "hit_rate": (retrieval.hit_rate, True),
    "context_waste": (retrieval.context_waste, False),
}


def build(name: str) -> Metric:
    """Turn a metric name into a :class:`Metric`.

    Retrieval metrics accept an optional ``@k`` suffix; without it the whole
    retrieved list is scored.

    >>> build("recall@5").name
    'recall@5'
    >>> build("hallucination").higher_is_better
    False
    """
    match = _AT_K.match(name.strip().lower())
    if not match:
        raise KeyError(f"unknown metric {name!r}")
    base, raw_k = match.group(1), match.group(2)
    k = int(raw_k) if raw_k else None

    if base in _RETRIEVAL_FUNCS:
        func, higher = _RETRIEVAL_FUNCS[base]
        canonical = f"{base}@{k}" if k else base
        return _retrieval_metric(canonical, func, k, higher)

    if base == "groundedness":
        return Metric(
            name="groundedness",
            score=lambda e, p, j: generation.groundedness(j, p.answer, _contexts(e, p)),
            needs_judge=True,
        )
    if base == "hallucination":
        return Metric(
            name="hallucination",
            score=lambda e, p, j: generation.hallucination(j, p.answer, _contexts(e, p)),
            higher_is_better=False,
            needs_judge=True,
        )
    if base == "answer_relevance":
        return Metric(
            name="answer_relevance",
            score=lambda e, p, j: generation.answer_relevance(j, p.answer, e.question),
            needs_judge=True,
        )
    if base == "answer_correctness":
        return Metric(
            name="answer_correctness",
            score=lambda e, p, j: generation.answer_correctness(
                j, p.answer, e.reference_answer or ""
            ),
            needs_judge=True,
            needs_reference=True,
        )
    if base == "refusal_rate":
        return Metric(
            name="refusal_rate",
            score=lambda e, p, j: generation.refusal(p.answer),
            higher_is_better=False,
        )

    raise KeyError(
        f"unknown metric {name!r}; available: {', '.join(available())}"
    )


def available() -> list[str]:
    """Every metric name the registry understands, ``@k`` variants elided."""
    return sorted(
        [f"{n}@k" for n in _RETRIEVAL_FUNCS]
        + [
            "groundedness",
            "hallucination",
            "answer_relevance",
            "answer_correctness",
            "refusal_rate",
        ]
    )


def build_all(names: Any = None) -> list[Metric]:
    """Build a list of metrics, defaulting to :data:`DEFAULT_METRICS`."""
    return [build(name) for name in (names or DEFAULT_METRICS)]
