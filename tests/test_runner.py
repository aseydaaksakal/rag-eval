import pytest

from rag_eval import evaluate, evaluate_predictions
from rag_eval.types import Example


def test_good_pipeline_scores_well(examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5", "mrr@5", "groundedness"])
    scores = run.aggregate()
    assert scores["recall@5"] == 1.0
    assert scores["mrr@5"] == 1.0
    assert scores["groundedness"] == 1.0
    assert not run.failures


def test_hallucination_is_detected(examples, hallucinating_pipeline):
    run = evaluate(examples, hallucinating_pipeline, metrics=["recall@5", "hallucination"])
    assert run.aggregate()["recall@5"] == 1.0      # retrieval was fine
    assert run.aggregate()["hallucination"] == 1.0  # the answer was not


def test_metrics_that_need_missing_fields_are_skipped_not_zeroed(good_pipeline):
    # q2 has no reference_answer, so answer_correctness must ignore it rather
    # than score it 0 and halve the mean.
    dataset = [
        Example(id="q1", question="What is the refund window?",
                relevant_ids=["p1"], reference_answer="Refunds within 30 days."),
        Example(id="q2", question="How do I cancel an order?", relevant_ids=["p2"]),
    ]
    run = evaluate(dataset, good_pipeline, metrics=["answer_correctness"])
    scored = [r for r in run.results if "answer_correctness" in r.scores]
    assert len(scored) == 1
    assert run.aggregate()["answer_correctness"] == scored[0].scores["answer_correctness"]


def test_retrieval_metrics_skip_examples_without_relevant_ids(good_pipeline):
    dataset = [Example(id="q1", question="What is the refund window?")]
    run = evaluate(dataset, good_pipeline, metrics=["recall@5"])
    assert run.results[0].scores == {}


def test_pipeline_errors_are_recorded_not_raised(examples):
    def broken(question):
        raise RuntimeError("index unavailable")

    run = evaluate(examples, broken, metrics=["recall@5"])
    assert len(run.failures) == 2
    assert "index unavailable" in run.failures[0].error
    assert run.aggregate()["recall@5"] == 0.0


def test_on_error_raise_propagates(examples):
    def broken(question):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        evaluate(examples, broken, on_error="raise")


def test_invalid_on_error_is_rejected(examples, good_pipeline):
    with pytest.raises(ValueError):
        evaluate(examples, good_pipeline, on_error="explode")


def test_pipeline_may_return_a_dict_or_a_string(examples):
    def dict_pipeline(question):
        return {"answer": "Thirty days.", "chunks": [{"id": "p1", "text": "30 day window"}]}

    run = evaluate(examples, dict_pipeline, metrics=["recall@5"])
    assert run.results[0].retrieved_ids == ["p1"]

    def string_pipeline(question):
        return "Thirty days."

    run = evaluate(examples, string_pipeline, metrics=["refusal_rate"])
    assert run.results[0].answer == "Thirty days."


def test_latency_is_recorded(examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"])
    assert all(r.latency_ms is not None and r.latency_ms >= 0 for r in run.results)


def test_by_tag_splits_the_aggregate(examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"])
    by_tag = run.by_tag()
    assert set(by_tag) == {"policy", "howto"}
    assert by_tag["howto"]["recall@5"] == 1.0


def test_worst_sorts_ascending(examples, hallucinating_pipeline):
    run = evaluate(examples, hallucinating_pipeline, metrics=["groundedness"])
    worst = run.worst("groundedness", limit=1)
    assert len(worst) == 1


def test_progress_hook_sees_every_example(examples, good_pipeline):
    seen = []
    evaluate(examples, good_pipeline, metrics=["recall@5"],
             progress=lambda i, total, r: seen.append((i, total)))
    assert seen == [(1, 2), (2, 2)]


def test_metadata_is_carried_into_the_run(examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"], metadata={"commit": "abc123"})
    assert run.metadata["commit"] == "abc123"
    assert run.metadata["judge"] == "lexical"


def test_precomputed_predictions(examples):
    run = evaluate_predictions(
        examples,
        {
            "q1": {"answer": "Refunds within a 30 day window.", "chunks": [{"id": "p1"}]},
            "q2": {"answer": "Email support.", "chunks": [{"id": "p2"}]},
        },
        metrics=["recall@5"],
    )
    assert run.aggregate()["recall@5"] == 1.0
    assert run.metadata["source"] == "precomputed"


def test_missing_prediction_is_a_failure_not_a_silent_drop(examples):
    partial = {"q1": {"answer": "x", "chunks": []}}
    run = evaluate_predictions(examples, partial, metrics=["recall@5"])
    assert len(run) == 2
    assert len(run.failures) == 1


def test_unknown_metric_name_is_rejected(examples, good_pipeline):
    with pytest.raises(KeyError, match="unknown metric"):
        evaluate(examples, good_pipeline, metrics=["nonsense"])
