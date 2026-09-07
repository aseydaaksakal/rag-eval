"""Load and save evaluation datasets.

The format is JSONL: one JSON object per line, one line per question. It diffs
cleanly in review, streams without loading everything, and survives a bad line
in the middle without taking the file with it.

    {"id": "q1", "question": "What is the refund window?",
     "relevant_ids": ["policy-3"], "tags": ["policy"]}

``id`` and ``question`` are required. Everything else is optional, and which
optional fields you fill in decides which metrics can be computed:

    relevant_ids        retrieval metrics
    reference_answer    answer_correctness
    reference_contexts  groundedness for generator-only runs
    tags                per-slice breakdowns in the report
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .types import Example


class DatasetError(ValueError):
    """A dataset file could not be read. The message names the line."""


def load(path: str | Path) -> list[Example]:
    """Read a JSONL dataset.

    Blank lines and ``#`` comment lines are skipped so a dataset can carry
    section headers.

    Raises:
        DatasetError: On malformed JSON or a missing required field, quoting
            the line number. Fixing one bad line in a 500-line file should not
            require a bisect.
    """
    file = Path(path)
    if not file.exists():
        raise DatasetError(f"dataset not found: {file}")

    examples: list[Example] = []
    seen: set[str] = set()

    for number, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{file}:{number}: invalid JSON ({exc.msg})") from exc
        try:
            example = Example.from_dict(payload)
        except ValueError as exc:
            raise DatasetError(f"{file}:{number}: {exc}") from exc

        if example.id in seen:
            raise DatasetError(
                f"{file}:{number}: duplicate id {example.id!r}; ids are how "
                "regression comparison matches examples across runs"
            )
        seen.add(example.id)
        examples.append(example)

    if not examples:
        raise DatasetError(f"{file} contains no examples")
    return examples


def save(examples: Iterable[Example], path: str | Path) -> Path:
    """Write examples as JSONL, dropping empty optional fields."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {k: v for k, v in example.to_dict().items() if v not in ([], {}, None)}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return file


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream raw records without building Examples. For large files."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield json.loads(line)


def summarize(examples: list[Example]) -> dict[str, object]:
    """Describe a dataset: size, coverage of optional fields, tag counts.

    Run this before trusting a number. A recall score computed over the eight
    examples that happen to have ``relevant_ids`` is not a recall score for
    your dataset.
    """
    tags: dict[str, int] = {}
    for example in examples:
        for tag in example.tags:
            tags[tag] = tags.get(tag, 0) + 1
    return {
        "examples": len(examples),
        "with_relevant_ids": sum(1 for e in examples if e.relevant_ids),
        "with_reference_answer": sum(1 for e in examples if e.reference_answer),
        "with_reference_contexts": sum(1 for e in examples if e.reference_contexts),
        "tags": dict(sorted(tags.items())),
    }
