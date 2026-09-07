from rag_eval.ragas_adapter import from_ragas_scores, to_ragas_dataset
from rag_eval.types import Chunk, Prediction


def test_rows_carry_texts_not_ids(examples):
    rows = to_ragas_dataset(
        examples,
        {"q1": Prediction(answer="Thirty days.", chunks=[Chunk("p1", "A 30 day window.")])},
    )
    assert len(rows) == 1
    assert rows[0]["contexts"] == ["A 30 day window."]
    assert rows[0]["ground_truth"] == "Thirty days from delivery."
    assert rows[0]["example_id"] == "q1"


def test_examples_without_predictions_are_skipped(examples):
    assert to_ragas_dataset(examples, {}) == []


def test_ragas_names_map_onto_ours(examples):
    run = from_ragas_scores(
        examples,
        [
            {"example_id": "q1", "answer": "a", "faithfulness": 0.8, "answer_relevancy": 0.9},
            {"example_id": "q2", "answer": "b", "faithfulness": 0.5, "answer_relevancy": 0.7},
        ],
    )
    assert set(run.metrics) == {"groundedness", "hallucination", "answer_relevance"}
    assert run.aggregate()["groundedness"] == 0.65
    assert run.aggregate()["hallucination"] == 0.35


def test_rows_match_on_question_when_id_is_absent(examples):
    run = from_ragas_scores(
        examples, [{"question": "What is the refund window?", "faithfulness": 1.0}]
    )
    assert run.results[0].example_id == "q1"


def test_non_numeric_columns_are_ignored(examples):
    run = from_ragas_scores(
        examples, [{"example_id": "q1", "faithfulness": 1.0, "notes": "fine", "ok": True}]
    )
    assert set(run.results[0].scores) == {"groundedness", "hallucination"}
