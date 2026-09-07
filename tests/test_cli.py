import json
from pathlib import Path

import pytest

from rag_eval.cli import main

DATASET = [
    {
        "id": "q1",
        "question": "What is the refund window?",
        "relevant_ids": ["p1"],
        "tags": ["policy"],
    },
    {
        "id": "q2",
        "question": "How do I cancel an order?",
        "relevant_ids": ["p2"],
        "tags": ["howto"],
    },
]

GOOD_PIPELINE = '''
from rag_eval.types import Chunk, Prediction

TEXTS = {
    "p1": "Refunds are accepted within a 30 day window from the delivery date.",
    "p2": "To cancel an order, email support before it enters fulfilment.",
}


def answer(question: str) -> Prediction:
    chunk_id = "p1" if "refund" in question else "p2"
    return Prediction(answer=TEXTS[chunk_id], chunks=[Chunk(chunk_id, TEXTS[chunk_id])])
'''

# Retrieval still works; the generator has stopped using what it retrieved.
BAD_PIPELINE = '''
from rag_eval.types import Chunk, Prediction


def answer(question: str) -> Prediction:
    chunk_id = "p1" if "refund" in question else "p2"
    return Prediction(answer="Our courier delivers on Tuesdays.", chunks=[Chunk(chunk_id, "")])
'''


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in DATASET), encoding="utf-8"
    )
    (tmp_path / "pipe.py").write_text(GOOD_PIPELINE, encoding="utf-8")
    (tmp_path / "pipe_bad.py").write_text(BAD_PIPELINE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(argv):
    return main(argv)


def test_run_writes_results(workspace, capsys):
    assert run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
                "--metric", "recall@5", "--out", "r.json", "--quiet"]) == 0
    payload = json.loads(Path("r.json").read_text())
    assert payload["aggregate"]["recall@5"] == 1.0
    assert "recall@5" in capsys.readouterr().out


def test_full_baseline_workflow(workspace, capsys):
    run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
         "--metric", "recall@5", "--metric", "groundedness", "--out", "r.json", "--quiet"])
    assert run(["baseline", "save", "r.json", "--to", "b.json", "--label", "v1"]) == 0
    capsys.readouterr()

    # Same pipeline, no change: compare passes.
    assert run(["compare", "r.json", "--baseline", "b.json"]) == 0
    assert "PASS" in capsys.readouterr().out

    # A pipeline whose generator ignores its context: compare exits 1.
    run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe_bad:answer",
         "--metric", "recall@5", "--metric", "groundedness", "--out", "r2.json", "--quiet"])
    capsys.readouterr()
    assert run(["compare", "r2.json", "--baseline", "b.json"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_compare_with_thresholds_and_html(workspace, capsys):
    run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
         "--metric", "recall@5", "--out", "r.json", "--quiet"])
    Path("t.toml").write_text('[thresholds."recall@5"]\nminimum = 1.5\n', encoding="utf-8")
    assert run(["compare", "r.json", "--thresholds", "t.toml", "--html", "out.html"]) == 1
    assert Path("out.html").exists()


def test_compare_without_baseline_or_thresholds_is_a_usage_error(workspace, capsys):
    run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
         "--metric", "recall@5", "--out", "r.json", "--quiet"])
    assert run(["compare", "r.json", "--baseline", "missing.json"]) == 2
    assert "nothing to check against" in capsys.readouterr().err


def test_report_renders_html(workspace, capsys):
    run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
         "--metric", "recall@5", "--out", "r.json", "--quiet"])
    assert run(["report", "r.json", "--html", "rep.html", "--title", "Nightly"]) == 0
    assert "Nightly" in Path("rep.html").read_text()


def test_describe_reports_coverage(workspace, capsys):
    assert run(["describe", "d.jsonl"]) == 0
    out = capsys.readouterr().out
    assert "2 examples" in out
    assert "with_relevant_ids" in out
    assert "policy" in out


def test_bad_pipeline_spec_is_a_usage_error(workspace, capsys):
    assert run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe.answer", "--quiet"]) == 2
    assert "module:attribute" in capsys.readouterr().err


def test_missing_attribute_is_a_usage_error(workspace, capsys):
    assert run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:nope", "--quiet"]) == 2
    assert "no attribute" in capsys.readouterr().err


def test_unknown_metric_is_a_usage_error(workspace, capsys):
    assert run(["run", "--dataset", "d.jsonl", "--pipeline", "pipe:answer",
                "--metric", "nonsense", "--quiet"]) == 2
    assert "unknown metric" in capsys.readouterr().err
