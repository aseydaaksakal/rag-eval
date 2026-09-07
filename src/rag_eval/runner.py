"""Run a pipeline over a dataset and score every answer.

    from rag_eval import evaluate, load_dataset

    result = evaluate(
        load_dataset("data/questions.jsonl"),
        my_pipeline,
        metrics=["recall@5", "mrr@5", "groundedness", "hallucination"],
    )
    print(result.aggregate())

A pipeline is any callable taking a question string. It may return a
``Prediction``, a dict with ``answer`` and ``chunks``, or a bare answer string.
The runner accepts all three so you can point it at what you already have
instead of writing an adapter first.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .judges.base import Judge
from .judges.lexical import LexicalJudge
from .metrics.registry import Metric, build_all
from .types import Example, ExampleResult, Prediction, RunResult

Pipeline = Callable[[str], Any]
ProgressHook = Callable[[int, int, ExampleResult], None]


def evaluate(
    dataset: Sequence[Example],
    pipeline: Pipeline,
    *,
    metrics: Iterable[str] | None = None,
    judge: Judge | None = None,
    on_error: str = "record",
    progress: ProgressHook | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunResult:
    """Score *pipeline* over *dataset*.

    Args:
        dataset: Examples, usually from :func:`rag_eval.dataset.load`.
        pipeline: Callable taking a question, returning a prediction.
        metrics: Metric names. Defaults to
            :data:`~rag_eval.metrics.registry.DEFAULT_METRICS`.
        judge: Scorer for generation metrics. Defaults to
            :class:`~rag_eval.judges.lexical.LexicalJudge`, which is offline
            and deterministic.
        on_error: ``"record"`` marks the example failed and continues;
            ``"raise"`` stops. Record is the default because a run over 500
            questions should not be lost to one bad input, and the failure
            count is itself a signal worth thresholding.
        progress: Called as ``(index, total, result)`` after each example.
        metadata: Copied into the result. Put your commit sha, model name and
            index version here — a score without them is not reproducible.

    Returns:
        A :class:`~rag_eval.types.RunResult`.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")

    active_judge = judge or LexicalJudge()
    built = build_all(metrics)
    total = len(dataset)

    run = RunResult(
        metrics=[m.name for m in built],
        metadata={
            "judge": getattr(active_judge, "name", type(active_judge).__name__),
            "example_count": total,
            **(metadata or {}),
        },
    )

    for index, example in enumerate(dataset, start=1):
        result = _evaluate_one(example, pipeline, built, active_judge, on_error)
        run.results.append(result)
        if progress:
            progress(index, total, result)

    return run


def _evaluate_one(
    example: Example,
    pipeline: Pipeline,
    metrics: list[Metric],
    judge: Judge,
    on_error: str,
) -> ExampleResult:
    started = time.perf_counter()
    try:
        prediction = Prediction.coerce(pipeline(example.question))
    except Exception as exc:
        if on_error == "raise":
            raise
        return ExampleResult(
            example_id=example.id,
            question=example.question,
            scores={},
            relevant_ids=list(example.relevant_ids),
            tags=list(example.tags),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = (time.perf_counter() - started) * 1000.0
    if prediction.latency_ms is None:
        prediction.latency_ms = elapsed

    scores: dict[str, float] = {}
    errors: list[str] = []
    for metric in metrics:
        if not metric.applies_to(example):
            continue
        try:
            scores[metric.name] = float(metric.score(example, prediction, judge))
        except Exception as exc:
            if on_error == "raise":
                raise
            errors.append(f"{metric.name}: {type(exc).__name__}: {exc}")

    return ExampleResult(
        example_id=example.id,
        question=example.question,
        scores=scores,
        answer=prediction.answer,
        retrieved_ids=prediction.chunk_ids,
        relevant_ids=list(example.relevant_ids),
        tags=list(example.tags),
        latency_ms=prediction.latency_ms,
        error="; ".join(errors) if errors else None,
    )


def evaluate_predictions(
    dataset: Sequence[Example],
    predictions: dict[str, Any],
    *,
    metrics: Iterable[str] | None = None,
    judge: Judge | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunResult:
    """Score predictions you already have, keyed by example id.

    For when the pipeline runs somewhere this process cannot reach — a
    different service, a nightly batch, a notebook someone else owns. Dump its
    output to JSON, score it here.

    Examples with no prediction are recorded as failures rather than dropped,
    so the count still matches the dataset.
    """
    return evaluate(
        dataset,
        _lookup_pipeline(dataset, predictions),
        metrics=metrics,
        judge=judge,
        metadata={"source": "precomputed", **(metadata or {})},
    )


def _lookup_pipeline(dataset: Sequence[Example], predictions: dict[str, Any]) -> Pipeline:
    by_question = {e.question: e.id for e in dataset}

    def pipeline(question: str) -> Any:
        example_id = by_question.get(question)
        if example_id is None or example_id not in predictions:
            raise KeyError(f"no prediction for example {example_id or question!r}")
        return predictions[example_id]

    return pipeline
