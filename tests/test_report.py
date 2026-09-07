import json

from rag_eval import compare, evaluate, to_html, to_json, to_text


def test_text_report_shows_metrics_and_worst_examples(examples, hallucinating_pipeline):
    run = evaluate(examples, hallucinating_pipeline, metrics=["groundedness", "recall@5"])
    text = to_text(run)
    assert "groundedness" in text
    assert "By tag" in text
    assert "Lowest groundedness" in text
    assert "q1" in text


def test_text_report_shows_deltas_and_verdict(examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"])
    passing = to_text(run, compare(run, {"aggregate": {"recall@5": 1.0}}))
    assert "PASS" in passing

    failing = to_text(run, compare(run, {"aggregate": {"recall@5": 1.0}},
                                   [__import__("rag_eval").Threshold("recall@5", minimum=1.5)]))
    assert "FAIL" in failing


def test_text_report_lists_pipeline_failures(examples):
    def broken(question):
        raise RuntimeError("index unavailable")

    text = to_text(evaluate(examples, broken, metrics=["recall@5"]))
    assert "2 example(s) failed to run" in text
    assert "index unavailable" in text


def test_json_report_round_trips(tmp_path, examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5"])
    path = to_json(run, tmp_path / "r.json")
    payload = json.loads(path.read_text())
    assert payload["aggregate"]["recall@5"] == 1.0
    assert payload["example_count"] == 2
    assert len(payload["results"]) == 2


def test_html_report_is_self_contained(tmp_path, examples, good_pipeline):
    run = evaluate(examples, good_pipeline, metrics=["recall@5", "groundedness"])
    comparison = compare(run, {"aggregate": {"recall@5": 0.9}})
    path = to_html(run, tmp_path / "r.html", comparison=comparison)
    document = path.read_text()
    assert document.startswith("<!doctype html>")
    assert "<script" not in document
    assert "src=" not in document           # no external assets
    assert "recall@5" in document
    assert "No threshold breached" in document


def test_html_escapes_example_text(tmp_path, examples):
    def injecting(question):
        return {"answer": "<script>alert(1)</script>", "chunks": [{"id": "p1"}]}

    run = evaluate(examples, injecting, metrics=["groundedness"])
    document = to_html(run, tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document
