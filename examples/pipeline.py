"""A toy RAG pipeline, so the examples run with nothing installed.

Retrieval is bag-of-words overlap and generation is a template. Neither is
worth copying; they exist so `rag-eval run --pipeline examples.pipeline:build`
produces real numbers you can watch move.

Swap `build()` for your own factory and the rest of the workflow is unchanged.
"""

from __future__ import annotations

from rag_eval.judges.lexical import tokenize
from rag_eval.types import Chunk, Prediction

CORPUS = {
    "policy-refund": "Refunds are accepted within a 30 day window from the delivery date.",
    "policy-cancel": "To cancel an order, email support before it enters fulfilment.",
    "policy-warranty": "The warranty covers manufacturing defects for two years.",
    "policy-restocking": "A 15 percent restocking fee applies to opened electronics.",
    "policy-gift": "Gift returns without a receipt are issued as store credit.",
    "shipping-express": "Express shipping is available in nine countries.",
    "shipping-address": "The delivery address can be changed until the parcel is picked up.",
    "shipping-times": "Standard parcels arrive in three to five working days.",
}


class ToyPipeline:
    """Overlap retrieval plus a template generator.

    Args:
        top_k: How many passages to retrieve.
        use_context: When False the generator ignores what was retrieved,
            which is how the examples demonstrate a hallucination regression
            while retrieval metrics stay flat.
    """

    def __init__(self, top_k: int = 3, use_context: bool = True) -> None:
        self.top_k = top_k
        self.use_context = use_context

    def retrieve(self, question: str) -> list[Chunk]:
        wanted = set(tokenize(question))
        scored = [
            (len(wanted & set(tokenize(text))) / (len(wanted) or 1), chunk_id, text)
            for chunk_id, text in CORPUS.items()
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            Chunk(id=chunk_id, text=text, score=round(score, 3))
            for score, chunk_id, text in scored[: self.top_k]
            if score > 0
        ]

    def __call__(self, question: str) -> Prediction:
        chunks = self.retrieve(question)
        if not self.use_context:
            answer = "Our courier partner delivers on Tuesdays in seventeen cities."
        elif chunks:
            answer = chunks[0].text
        else:
            answer = "I don't know based on the available documentation."
        return Prediction(answer=answer, chunks=chunks)


def build() -> ToyPipeline:
    """Factory the CLI can import: `--pipeline examples.pipeline:build`."""
    return ToyPipeline()


def build_broken() -> ToyPipeline:
    """The same retriever with a generator that ignores its context."""
    return ToyPipeline(use_context=False)
