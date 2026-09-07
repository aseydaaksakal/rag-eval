import pytest

from rag_eval.types import Chunk, Example, Prediction

CONTEXTS = {
    "p1": "Refunds are accepted within a 30 day window from the delivery date.",
    "p2": "To cancel an order, email support before it enters fulfilment.",
    "p3": "Express shipping is available in nine countries.",
}


@pytest.fixture
def examples():
    return [
        Example(
            id="q1",
            question="What is the refund window?",
            relevant_ids=["p1"],
            reference_answer="Thirty days from delivery.",
            tags=["policy"],
        ),
        Example(
            id="q2",
            question="How do I cancel an order?",
            relevant_ids=["p2"],
            tags=["policy", "howto"],
        ),
    ]


@pytest.fixture
def good_pipeline():
    """Retrieves the right passage and answers from it."""
    answers = {
        "What is the refund window?": ("Refunds are accepted within a 30 day window.", "p1"),
        "How do I cancel an order?": ("Email support before fulfilment to cancel an order.", "p2"),
    }

    def pipeline(question: str) -> Prediction:
        answer, chunk_id = answers[question]
        return Prediction(
            answer=answer,
            chunks=[Chunk(id=chunk_id, text=CONTEXTS[chunk_id], score=0.9)],
        )

    return pipeline


@pytest.fixture
def hallucinating_pipeline():
    """Retrieves the right passage and then ignores it."""

    def pipeline(question: str) -> Prediction:
        chunk_id = "p1" if "refund" in question else "p2"
        return Prediction(
            answer="Our courier partner delivers on Tuesdays in seventeen cities.",
            chunks=[Chunk(id=chunk_id, text=CONTEXTS[chunk_id])],
        )

    return pipeline
