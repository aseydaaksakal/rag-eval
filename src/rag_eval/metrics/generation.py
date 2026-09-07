"""Generation metrics: is the answer supported, relevant, and correct?

All of these delegate to a :class:`~rag_eval.judges.base.Judge`, so the same
metric can be gated cheaply in CI with the lexical judge and inspected more
carefully with an LLM judge, without the metric definition changing.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..judges.base import Judge


def groundedness(judge: Judge, answer: str, contexts: Sequence[str]) -> float:
    """Share of the answer that the retrieved context supports.

    The core hallucination signal. It says nothing about whether the answer is
    *correct* — an answer faithfully derived from a wrong passage scores 1.0.
    Pair it with ``answer_correctness`` when you have reference answers.
    """
    return judge.groundedness(answer, contexts)


def hallucination(judge: Judge, answer: str, contexts: Sequence[str]) -> float:
    """``1 - groundedness``. Lower is better.

    Reported separately because a rate is what you put on a dashboard and in an
    alert, and inverting a metric at the point of display is how thresholds end
    up backwards.
    """
    return 1.0 - judge.groundedness(answer, contexts)


def answer_relevance(judge: Judge, answer: str, question: str) -> float:
    """How directly the answer addresses the question that was asked.

    Catches the fluent-but-off-topic failure, where retrieval returned
    neighbouring passages and the model wrote confidently about the wrong
    thing.
    """
    return judge.relevance(answer, question)


def answer_correctness(judge: Judge, answer: str, reference: str) -> float:
    """Agreement with a known-good reference answer.

    Only scored for examples that have ``reference_answer`` set. Writing those
    is slow, so most datasets have them for a subset — that is fine, the
    aggregate simply covers the examples that have one.
    """
    return judge.similarity(answer, reference)


def refusal(answer: str) -> float:
    """1.0 when the answer looks like a refusal or an admission of ignorance.

    Not a quality signal on its own. It is the number that explains a
    groundedness improvement: a pipeline that starts refusing everything scores
    beautifully on hallucination and helps nobody.
    """
    if not answer.strip():
        return 1.0
    lowered = answer.lower()
    markers = (
        "i don't know",
        "i do not know",
        "i cannot",
        "i can't",
        "no information",
        "not enough information",
        "insufficient context",
        "unable to answer",
        "does not contain",
        "no relevant",
    )
    return 1.0 if any(m in lowered for m in markers) else 0.0
