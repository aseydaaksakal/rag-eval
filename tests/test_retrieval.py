import pytest

from rag_eval.metrics.retrieval import (
    context_waste,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RETRIEVED = ["a", "x", "b", "y", "z"]
RELEVANT = ["a", "b", "c"]


def test_recall_counts_only_the_top_k():
    assert recall_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT) == pytest.approx(2 / 3)


def test_precision_divides_by_k_not_by_what_was_returned():
    # Only two results, but the caller asked for five: the three missing slots
    # count against precision rather than being ignored.
    assert precision_at_k(["a", "b"], RELEVANT, k=5) == pytest.approx(2 / 5)
    assert precision_at_k(["a", "b"], RELEVANT, k=2) == pytest.approx(1.0)


def test_mrr_is_the_rank_of_the_first_hit():
    assert mrr(["a", "x"], RELEVANT) == 1.0
    assert mrr(["x", "a"], RELEVANT) == 0.5
    assert mrr(["x", "y"], RELEVANT) == 0.0
    assert mrr(["x", "a"], RELEVANT, k=1) == 0.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k(["a", "b", "x"], RELEVANT, k=3)
    late = ndcg_at_k(["x", "a", "b"], RELEVANT, k=3)
    assert early > late
    assert ndcg_at_k(["a", "b", "c"], RELEVANT, k=3) == pytest.approx(1.0)


def test_ndcg_ideal_accounts_for_a_short_result_list():
    # Two relevant documents exist but only one slot: a perfect single result
    # is a perfect score for that slot.
    assert ndcg_at_k(["a"], ["a", "b"], k=1) == pytest.approx(1.0)


def test_hit_rate_is_binary():
    assert hit_rate(["x", "a"], RELEVANT) == 1.0
    assert hit_rate(["x", "y"], RELEVANT) == 0.0


def test_context_waste_is_one_minus_precision():
    assert context_waste(["a", "x"], RELEVANT) == pytest.approx(0.5)
    assert context_waste(["a", "b"], RELEVANT) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "func", [recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate, context_waste]
)
def test_empty_inputs_score_zero_rather_than_dividing_by_zero(func):
    assert func([], [], k=5) == 0.0
    assert func(["a"], [], k=5) == 0.0


def test_ranking_metrics_respect_order_not_set_membership():
    assert mrr(["b", "a"], RELEVANT) == 1.0
    assert recall_at_k(["b", "a"], RELEVANT, k=2) == recall_at_k(["a", "b"], RELEVANT, k=2)
