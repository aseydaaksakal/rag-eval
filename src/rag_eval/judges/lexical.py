"""A deterministic judge that needs no model.

Generation metrics need something that can say whether a sentence is supported
by a passage. The usual answer is another LLM, which costs money, varies run to
run, and cannot run in CI without a network call.

This judge answers with token overlap instead. It is worse than an LLM judge at
recognising paraphrase, and better at everything a regression test needs: it is
free, offline, and returns the same number every time. Use it as the gate in
CI, and an LLM judge for the occasional deeper read.

**What it will get wrong.** A claim restated entirely in synonyms scores low
even though the context supports it. A fluent sentence assembled from words
scattered across the context scores high even though the context does not
support it. Both are visible in the score distribution rather than hidden, and
neither moves between runs, so a *change* in the score still means a change in
the pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Function words carry no evidence, so overlap on them is not support.
# noqa on the split(): a wrapped string is far easier to review and edit than a
# 70-element list literal, and this runs once at import.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her his how i if in into is it its me my no not of on or our out she should
    so than that the their them then there these they this those to too us was we
    were what when where which who why will with would you your
    """.split()  # noqa: SIM905
)


_SUFFIXES = ("ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Strip one common inflectional suffix, conservatively.

    Enough to make ``refunds`` match ``refund`` and ``accepted`` match
    ``accept``, which is most of what an overlap score needs. It is not a real
    stemmer and it will occasionally merge unrelated words (``less`` becomes
    ``les``); the cost is a slightly noisier score, not a nondeterministic one.

    >>> stem("refunds"), stem("accepted"), stem("window")
    ('refund', 'accept', 'window')
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase content tokens: stopwords removed, suffixes stripped."""
    return [
        stem(token)
        for token in (m.group(0).lower() for m in _WORD.finditer(text))
        if token not in STOPWORDS
    ]


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation and newlines.

    Naive on purpose. Abbreviations get over-split, which costs a little
    precision and no determinism.
    """
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def overlap(claim: str, context: str) -> float:
    """Share of *claim*'s content tokens that also appear in *context*."""
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 1.0  # nothing asserted, nothing unsupported
    context_tokens = set(tokenize(context))
    return len(claim_tokens & context_tokens) / len(claim_tokens)


class LexicalJudge:
    """Score claims against context by token overlap.

    Args:
        threshold: Overlap at or above which a sentence counts as supported.
            0.6 is a reasonable default: high enough that a sentence sharing
            only a topic word fails, low enough to tolerate rewording.
    """

    name = "lexical"

    def __init__(self, threshold: float = 0.6) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def supported(self, claim: str, contexts: Sequence[str]) -> float:
        """Best overlap between *claim* and any single context passage.

        Best-of rather than pooled: a claim is supported when one passage
        supports it, not when its words happen to be scattered across five.
        """
        if not contexts:
            return 0.0
        return max(overlap(claim, context) for context in contexts)

    def groundedness(self, answer: str, contexts: Sequence[str]) -> float:
        """Share of the answer's sentences that clear ``threshold``.

        >>> judge = LexicalJudge()
        >>> judge.groundedness("The refund window is 30 days.",
        ...                    ["Refunds are accepted within a 30 day window."])
        1.0
        """
        sentences = split_sentences(answer)
        if not sentences:
            return 0.0
        supported = sum(
            1 for s in sentences if self.supported(s, contexts) >= self.threshold
        )
        return supported / len(sentences)

    def unsupported_sentences(self, answer: str, contexts: Sequence[str]) -> list[str]:
        """The sentences that failed. This is what you read when a score drops."""
        return [
            s
            for s in split_sentences(answer)
            if self.supported(s, contexts) < self.threshold
        ]

    def relevance(self, answer: str, question: str) -> float:
        """Share of the question's content tokens the answer picks up.

        A weak proxy for "did it answer the question", but it does catch the
        common failure of a fluent answer about the wrong subject.
        """
        return overlap(question, answer)

    def similarity(self, answer: str, reference: str) -> float:
        """Symmetric token overlap between an answer and a reference answer."""
        left, right = set(tokenize(answer)), set(tokenize(reference))
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
