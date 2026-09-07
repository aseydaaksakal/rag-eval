import json

import pytest

from rag_eval import compare, evaluate, load_baseline, save_baseline
from rag_eval.baseline import Threshold, load_thresholds
from rag_eval.types import ExampleResult, RunResult


def make_run(**scores):
    return RunResult(
        results=[ExampleResult(example_id="q1", question="?", scores=dict(scores))],
        metrics=list(scores),
    )


def test_baseline_round_trip(tmp_path, examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"])
    path = save_baseline(run, tmp_path / "b.json", label="v1")
    loaded = load_baseline(path)
    assert loaded["aggregate"]["recall@5"] == 1.0
    assert loaded["label"] == "v1"
    assert "results" not in loaded  # baselines stay readable in review


def test_missing_baseline_explains_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="baseline save"):
        load_baseline(tmp_path / "nope.json")


def test_no_regression_when_scores_hold():
    result = compare(make_run(**{"recall@5": 0.80}), {"aggregate": {"recall@5": 0.80}})
    assert result.passed
    assert result.deltas["recall@5"] == 0.0


def test_small_moves_are_within_tolerance():
    result = compare(make_run(**{"recall@5": 0.795}), {"aggregate": {"recall@5": 0.80}},
                     tolerance=0.01)
    assert result.passed


def test_a_real_drop_is_caught():
    result = compare(make_run(**{"recall@5": 0.70}), {"aggregate": {"recall@5": 0.80}},
                     tolerance=0.01)
    assert not result.passed
    assert result.findings[0].kind == "drop"
    assert "0.8000 -> 0.7000" in result.findings[0].message


def test_direction_is_read_from_the_registry_not_the_config():
    # hallucination going UP is the regression, with no sign flip in the config.
    worse = compare(make_run(hallucination=0.20), {"aggregate": {"hallucination": 0.10}},
                    tolerance=0.01)
    assert not worse.passed

    better = compare(make_run(hallucination=0.05), {"aggregate": {"hallucination": 0.10}},
                     tolerance=0.01)
    assert better.passed


def test_absolute_floor_without_a_baseline():
    result = compare(make_run(**{"recall@5": 0.60}), None,
                     [Threshold(metric="recall@5", minimum=0.80)])
    assert not result.passed
    assert result.findings[0].kind == "floor"
    assert "below the required" in result.findings[0].message


def test_floor_on_a_lower_is_better_metric_is_a_ceiling():
    result = compare(make_run(hallucination=0.30), None,
                     [Threshold(metric="hallucination", minimum=0.10)])
    assert not result.passed
    assert "above the required" in result.findings[0].message


def test_threshold_for_an_uncomputed_metric_is_reported():
    result = compare(make_run(**{"recall@5": 0.9}), None,
                     [Threshold(metric="ndcg@5", minimum=0.5)])
    assert not result.passed
    assert result.findings[0].kind == "missing"


def test_thresholds_load_from_toml(tmp_path):
    path = tmp_path / "rag-eval.toml"
    path.write_text(
        """
        [thresholds."recall@5"]
        minimum = 0.80
        max_drop = 0.02

        [thresholds.hallucination]
        minimum = 0.10
        """,
        encoding="utf-8",
    )
    thresholds = {t.metric: t for t in load_thresholds(path)}
    assert thresholds["recall@5"].minimum == 0.80
    assert thresholds["recall@5"].max_drop == 0.02
    assert thresholds["hallucination"].max_drop is None


def test_malformed_threshold_table_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[thresholds]\n"recall@5" = 0.8\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a table"):
        load_thresholds(path)


def test_comparison_serializes(tmp_path):
    result = compare(make_run(**{"recall@5": 0.7}), {"aggregate": {"recall@5": 0.9}})
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] is False
    assert payload["deltas"]["recall@5"] == pytest.approx(-0.2)
