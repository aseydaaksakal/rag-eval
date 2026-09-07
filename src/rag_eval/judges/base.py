"""What a judge has to provide."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Anything that can score an answer against its context.

    Implement this to plug in your own scorer. Every method returns a float in
    ``[0, 1]``. :class:`~rag_eval.judges.lexical.LexicalJudge` is the offline
    reference implementation; :class:`~rag_eval.judges.llm.LLMJudge` wraps a
    model.
    """

    name: str

    def groundedness(self, answer: str, contexts: Sequence[str]) -> float:
        """Share of the answer that the contexts support."""
        ...

    def relevance(self, answer: str, question: str) -> float:
        """How much of the question the answer addresses."""
        ...

    def similarity(self, answer: str, reference: str) -> float:
        """How close the answer is to a known-good reference."""
        ...
