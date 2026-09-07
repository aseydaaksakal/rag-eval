"""Judge backed by a language model.

The library does not talk to any provider. You pass a callable that takes a
prompt and returns a string, so you keep your own client, credentials, retries
and timeouts:

    from openai import OpenAI

    client = OpenAI()

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    judge = LLMJudge(complete)

Results are cached per (prompt) within a run, because a dataset of 200
questions otherwise means 600 calls and most evaluation loops re-score the same
examples repeatedly.

An LLM judge is not free and not deterministic. Use it for depth, keep
:class:`~rag_eval.judges.lexical.LexicalJudge` as the gate in CI.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

Completion = Callable[[str], str]

_NUMBER = re.compile(r"(?:0?\.\d+|[01](?:\.0+)?|\d{1,3}%)")

GROUNDEDNESS_PROMPT = """\
You are grading whether an ANSWER is supported by the CONTEXT.

Rules:
- Judge support only. Do not judge whether the answer is helpful or well written.
- A claim that is true in general but absent from the CONTEXT is NOT supported.
- Reply with a single number between 0 and 1: the fraction of the answer's
  claims that the CONTEXT supports. No other text.

CONTEXT:
{context}

ANSWER:
{answer}

Score:"""

RELEVANCE_PROMPT = """\
Rate how directly the ANSWER addresses the QUESTION, from 0 to 1.
Ignore whether it is factually correct. Reply with the number only.

QUESTION:
{question}

ANSWER:
{answer}

Score:"""

SIMILARITY_PROMPT = """\
Rate how closely the ANSWER matches the REFERENCE in meaning, from 0 to 1.
Wording may differ. Reply with the number only.

REFERENCE:
{reference}

ANSWER:
{answer}

Score:"""


def parse_score(text: str) -> float:
    """Pull a 0-1 score out of a model reply, tolerating stray prose.

    >>> parse_score("0.8")
    0.8
    >>> parse_score("Score: 75%")
    0.75
    """
    match = _NUMBER.search(text or "")
    if not match:
        raise ValueError(f"no score found in judge reply: {text[:120]!r}")
    token = match.group(0)
    value = float(token.rstrip("%")) / 100 if token.endswith("%") else float(token)
    return max(0.0, min(1.0, value))


class LLMJudge:
    """Score with a model of your choosing.

    Args:
        complete: Callable taking a prompt, returning the model's text.
        name: Recorded in the run metadata so a report says which judge
            produced its numbers.
        on_error: ``"raise"`` to fail loudly, or ``"zero"`` to score 0.0 and
            keep going. Use ``"zero"`` for long unattended runs.
    """

    def __init__(
        self,
        complete: Completion,
        *,
        name: str = "llm",
        on_error: str = "raise",
    ) -> None:
        if on_error not in ("raise", "zero"):
            raise ValueError("on_error must be 'raise' or 'zero'")
        self.complete = complete
        self.name = name
        self.on_error = on_error
        self._cache: dict[str, float] = {}
        self.calls = 0

    def _score(self, prompt: str) -> float:
        if prompt in self._cache:
            return self._cache[prompt]
        try:
            self.calls += 1
            value = parse_score(self.complete(prompt))
        except Exception:
            if self.on_error == "raise":
                raise
            value = 0.0
        self._cache[prompt] = value
        return value

    def groundedness(self, answer: str, contexts: Sequence[str]) -> float:
        if not answer.strip():
            return 0.0
        if not contexts:
            return 0.0
        joined = "\n\n---\n\n".join(contexts)
        return self._score(GROUNDEDNESS_PROMPT.format(context=joined, answer=answer))

    def relevance(self, answer: str, question: str) -> float:
        if not answer.strip():
            return 0.0
        return self._score(RELEVANCE_PROMPT.format(question=question, answer=answer))

    def similarity(self, answer: str, reference: str) -> float:
        if not answer.strip() or not reference.strip():
            return 0.0
        return self._score(SIMILARITY_PROMPT.format(reference=reference, answer=answer))
