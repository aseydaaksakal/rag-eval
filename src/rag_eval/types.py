"""The objects that move through an evaluation run.

An evaluation has three stages and each one has a type:

    Example      what you asked and what a correct answer looks like
    Prediction   what your pipeline produced for that Example
    RunResult    every metric, per example and aggregated
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """One retrieved passage.

    Args:
        id: Stable identifier used to match against ``Example.relevant_ids``.
            Whatever your index calls a document; it only has to be consistent.
        text: The passage itself, used by groundedness metrics.
        score: Retriever score, if you have one. Not used for ranking metrics —
            list order is what counts.
        metadata: Anything else you want to carry into the report.
    """

    id: str
    text: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> Chunk:
        """Accept a Chunk, a dict, or a bare id string."""
        if isinstance(value, Chunk):
            return value
        if isinstance(value, str):
            return cls(id=value)
        if isinstance(value, dict):
            return cls(
                id=str(value.get("id", "")),
                text=value.get("text", ""),
                score=value.get("score"),
                metadata=value.get("metadata", {}) or {},
            )
        raise TypeError(f"cannot read a chunk from {type(value).__name__}")


@dataclass(slots=True)
class Example:
    """One test case.

    Args:
        id: Stable identifier. Regression comparison matches on this, so keep
            it stable across dataset edits or you lose the history.
        question: The user's question.
        relevant_ids: Chunk ids a correct retrieval should surface. Required
            for retrieval metrics.
        reference_answer: A known-good answer, if you have one.
        reference_contexts: Passage texts a correct answer can be built from.
            Groundedness falls back to the retrieved chunks when absent.
        tags: Free-form labels. Reports break scores down by tag, which is how
            you notice that only multi-hop questions regressed.
    """

    id: str
    question: str
    relevant_ids: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Example:
        missing = {"id", "question"} - raw.keys()
        if missing:
            raise ValueError(f"example is missing {', '.join(sorted(missing))}")
        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            relevant_ids=[str(x) for x in raw.get("relevant_ids", [])],
            reference_answer=raw.get("reference_answer"),
            reference_contexts=list(raw.get("reference_contexts", [])),
            tags=[str(t) for t in raw.get("tags", [])],
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Prediction:
    """What the pipeline produced for one example.

    Args:
        answer: The generated answer. Leave empty for retrieval-only pipelines.
        chunks: Retrieved passages **in rank order**. Order is the signal for
            MRR and nDCG.
        latency_ms: Wall-clock time, filled in by the runner if you do not set it.
    """

    answer: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> Prediction:
        """Accept a Prediction, a dict, or a bare answer string."""
        if isinstance(value, Prediction):
            return value
        if isinstance(value, str):
            return cls(answer=value)
        if isinstance(value, dict):
            return cls(
                answer=value.get("answer", "") or "",
                chunks=[Chunk.coerce(c) for c in value.get("chunks", [])],
                latency_ms=value.get("latency_ms"),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError(f"cannot read a prediction from {type(value).__name__}")

    @property
    def chunk_ids(self) -> list[str]:
        return [c.id for c in self.chunks]

    @property
    def contexts(self) -> list[str]:
        return [c.text for c in self.chunks if c.text]


@dataclass(slots=True)
class ExampleResult:
    """Every metric for a single example, plus enough context to debug it."""

    example_id: str
    question: str
    scores: dict[str, float]
    answer: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    """A whole evaluation run: per-example results and their aggregates."""

    results: list[ExampleResult] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def failures(self) -> list[ExampleResult]:
        """Examples where the pipeline itself raised."""
        return [r for r in self.results if r.error]

    def aggregate(self) -> dict[str, float]:
        """Mean of each metric across examples that produced a score."""
        return {name: self.mean(name) for name in self.metrics}

    def mean(self, metric: str) -> float:
        values = [r.scores[metric] for r in self.results if metric in r.scores]
        return statistics.fmean(values) if values else 0.0

    def by_tag(self) -> dict[str, dict[str, float]]:
        """Per-tag means, so you can see which slice moved.

        An overall score that barely shifts can hide a slice that fell apart.
        """
        tags: dict[str, list[ExampleResult]] = {}
        for result in self.results:
            for tag in result.tags:
                tags.setdefault(tag, []).append(result)

        out: dict[str, dict[str, float]] = {}
        for tag, group in sorted(tags.items()):
            out[tag] = {
                name: statistics.fmean(
                    [r.scores[name] for r in group if name in r.scores]
                )
                if any(name in r.scores for r in group)
                else 0.0
                for name in self.metrics
            }
        return out

    def worst(self, metric: str, limit: int = 10) -> list[ExampleResult]:
        """Lowest-scoring examples for *metric*. Where debugging starts."""
        scored = [r for r in self.results if metric in r.scores]
        return sorted(scored, key=lambda r: r.scores[metric])[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "metadata": self.metadata,
            "aggregate": self.aggregate(),
            "by_tag": self.by_tag(),
            "example_count": len(self.results),
            "failure_count": len(self.failures),
            "results": [r.to_dict() for r in self.results],
        }
