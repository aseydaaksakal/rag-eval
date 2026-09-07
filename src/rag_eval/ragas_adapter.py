"""Interop with [Ragas](https://github.com/explodinggradients/ragas).

Ragas has a larger metric catalogue and a proper LLM-based judge. This module
does not wrap it — it converts between the two data models, so you can keep one
dataset and one regression harness while using either scorer:

    from rag_eval.ragas_adapter import to_ragas_dataset, from_ragas_scores

    rows = to_ragas_dataset(dataset, predictions)
    scores = ragas.evaluate(Dataset.from_list(rows), metrics=[faithfulness])
    run = from_ragas_scores(dataset, scores.to_pandas().to_dict("records"))

    compare(run, load_baseline("rag-eval-baseline.json"))   # same gate as before

Ragas is not a dependency of this package. Nothing here imports it; the
functions only move dicts around, so they are importable and testable without
it installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import Example, ExampleResult, Prediction, RunResult

#: Ragas metric name -> the name this library uses for the same idea.
RAGAS_TO_RAG_EVAL = {
    "faithfulness": "groundedness",
    "answer_relevancy": "answer_relevance",
    "context_recall": "recall",
    "context_precision": "precision",
    "answer_correctness": "answer_correctness",
    "answer_similarity": "answer_correctness",
}


def to_ragas_dataset(
    dataset: Sequence[Example],
    predictions: dict[str, Prediction | dict[str, Any] | str],
) -> list[dict[str, Any]]:
    """Build the row dicts Ragas expects, keyed by example id.

    Ragas wants ``question``, ``answer``, ``contexts`` and ``ground_truth``.
    Contexts are passage *texts*, not ids, so a prediction whose chunks carry
    no text produces empty contexts and Ragas will score it zero — populate
    ``Chunk.text`` before coming here.
    """
    rows: list[dict[str, Any]] = []
    for example in dataset:
        raw = predictions.get(example.id)
        if raw is None:
            continue
        prediction = Prediction.coerce(raw)
        rows.append(
            {
                "question": example.question,
                "answer": prediction.answer,
                "contexts": prediction.contexts or list(example.reference_contexts),
                "ground_truth": example.reference_answer or "",
                "example_id": example.id,
            }
        )
    return rows


def from_ragas_scores(
    dataset: Sequence[Example],
    rows: Sequence[dict[str, Any]],
    *,
    rename: dict[str, str] | None = None,
) -> RunResult:
    """Turn Ragas per-row scores into a :class:`~rag_eval.types.RunResult`.

    Rows are matched on ``example_id`` when present, otherwise on ``question``.
    Metric names are mapped through :data:`RAGAS_TO_RAG_EVAL` so a baseline
    recorded from a native run still lines up.

    ``faithfulness`` becomes ``groundedness``; a ``hallucination`` metric is
    derived from it, because that is the number people put on dashboards.
    """
    mapping = {**RAGAS_TO_RAG_EVAL, **(rename or {})}
    by_id = {e.id: e for e in dataset}
    by_question = {e.question: e for e in dataset}

    results: list[ExampleResult] = []
    metrics: list[str] = []

    for row in rows:
        example = by_id.get(row.get("example_id")) or by_question.get(row.get("question", ""))
        if example is None:
            continue

        scores: dict[str, float] = {}
        for key, value in row.items():
            if key in ("question", "answer", "contexts", "ground_truth", "example_id"):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            name = mapping.get(key, key)
            scores[name] = float(value)
            if name == "groundedness":
                scores["hallucination"] = 1.0 - float(value)

        for name in scores:
            if name not in metrics:
                metrics.append(name)

        results.append(
            ExampleResult(
                example_id=example.id,
                question=example.question,
                scores=scores,
                answer=row.get("answer", "") or "",
                relevant_ids=list(example.relevant_ids),
                tags=list(example.tags),
            )
        )

    return RunResult(results=results, metrics=metrics, metadata={"judge": "ragas"})
