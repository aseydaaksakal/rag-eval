import json

import pytest

from rag_eval import load_dataset, save_dataset, summarize
from rag_eval.dataset import DatasetError
from rag_eval.types import Example

ROWS = [
    {
        "id": "q1",
        "question": "What is the refund window?",
        "relevant_ids": ["p1"],
        "tags": ["policy"],
    },
    {
        "id": "q2",
        "question": "How do I cancel?",
        "relevant_ids": ["p2"],
        "reference_answer": "Email support.",
        "tags": ["policy", "howto"],
    },
]


def write(tmp_path, rows, name="d.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_round_trip(tmp_path):
    loaded = load_dataset(write(tmp_path, ROWS))
    assert [e.id for e in loaded] == ["q1", "q2"]
    assert loaded[1].reference_answer == "Email support."

    out = save_dataset(loaded, tmp_path / "out.jsonl")
    assert [e.id for e in load_dataset(out)] == ["q1", "q2"]


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        "# policy questions\n\n" + json.dumps(ROWS[0]) + "\n\n", encoding="utf-8"
    )
    assert len(load_dataset(path)) == 1


def test_bad_json_names_the_line(tmp_path):
    path = write(tmp_path, ROWS)
    path.write_text(path.read_text() + "{not json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r":3: invalid JSON"):
        load_dataset(path)


def test_missing_required_field_names_the_line(tmp_path):
    with pytest.raises(DatasetError, match=r":1: example is missing question"):
        load_dataset(write(tmp_path, [{"id": "q1"}]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(write(tmp_path, [ROWS[0], ROWS[0]]))


def test_empty_and_missing_files(tmp_path):
    empty = tmp_path / "e.jsonl"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no examples"):
        load_dataset(empty)
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_summarize_reports_field_coverage(tmp_path):
    summary = summarize(load_dataset(write(tmp_path, ROWS)))
    assert summary["examples"] == 2
    assert summary["with_relevant_ids"] == 2
    assert summary["with_reference_answer"] == 1
    assert summary["tags"] == {"howto": 1, "policy": 2}


def test_save_omits_empty_optional_fields(tmp_path):
    out = save_dataset([Example(id="q", question="?")], tmp_path / "o.jsonl")
    written = json.loads(out.read_text())
    assert written == {"id": "q", "question": "?"}
