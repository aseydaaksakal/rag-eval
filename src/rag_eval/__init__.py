"""rag-eval: regression testing for RAG pipelines.

    from rag_eval import evaluate, load_dataset

    result = evaluate(load_dataset("questions.jsonl"), my_pipeline)
    print(result.aggregate())

Retrieval metrics, hallucination tracking, and a baseline you can fail a build
against. The default judge is offline and deterministic, so the whole thing
runs in CI without a network call.
"""

from .baseline import (
    Comparison,
    Finding,
    Threshold,
    compare,
    load_baseline,
    load_thresholds,
    save_baseline,
)
from .dataset import DatasetError, summarize
from .dataset import load as load_dataset
from .dataset import save as save_dataset
from .judges import Judge, LexicalJudge, LLMJudge
from .metrics import DEFAULT_METRICS, Metric, available
from .report import to_html, to_json, to_text
from .runner import evaluate, evaluate_predictions
from .types import Chunk, Example, ExampleResult, Prediction, RunResult

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "evaluate_predictions",
    "load_dataset",
    "save_dataset",
    "summarize",
    "DatasetError",
    "Example",
    "Prediction",
    "Chunk",
    "ExampleResult",
    "RunResult",
    "Judge",
    "LexicalJudge",
    "LLMJudge",
    "Metric",
    "DEFAULT_METRICS",
    "available",
    "compare",
    "save_baseline",
    "load_baseline",
    "load_thresholds",
    "Threshold",
    "Comparison",
    "Finding",
    "to_json",
    "to_text",
    "to_html",
    "__version__",
]
