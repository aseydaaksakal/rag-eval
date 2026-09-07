# Contributing

## Setup

```bash
git clone https://github.com/aseydaaksakal/rag-eval
cd rag-eval
pip install -e ".[dev]"
pytest
```

## Before opening a pull request

```bash
pytest
ruff check .
pytest examples/test_rag_regression.py     # the examples must keep working
```

CI runs these on Python 3.11 through 3.13.

## What makes a change easy to merge

**A test that fails without it.** For a metric, that means a case where the old
behaviour gives the wrong number, not just a case where the new code runs.

**Numeric behaviour pinned by hand-checkable examples.** `mrr(["x", "a"],
["a"]) == 0.5` is reviewable. A snapshot of a float is not.

**A reason in the commit message.** What the code does is in the diff.

## Adding a metric

1. Write the function in `metrics/retrieval.py` or `metrics/generation.py`.
   Return `[0, 1]`. Handle empty inputs without dividing by zero.
2. Register it in `metrics/registry.py`. Set `higher_is_better=False` if lower
   is better — the regression checker reads it, and getting it wrong silently
   inverts a gate. Set `needs_reference` or `needs_relevant` so examples that
   cannot be scored are skipped rather than zeroed.
3. Document it in `docs/metrics.md`, including **when it misleads**. Every
   metric misleads somewhere, and a metric whose failure modes are undocumented
   will be trusted past the point where it should be.
4. Test the edge cases: empty retrieval, empty relevant set, `k` larger than
   the result list, perfect and zero scores.

## Adding a judge

Implement `groundedness`, `relevance` and `similarity`, all returning
`[0, 1]`, and set `name`. Document what it gets wrong in `docs/judges.md`.

Judges that need a network call must not be the default and must not be needed
by the test suite.

## Scope

In scope: metrics, judges, reporting, regression tooling, interop with other
evaluation libraries.

Ask first: async or concurrent evaluation, multi-turn conversations,
significance testing on deltas. All wanted, all architectural.

Out of scope: retrievers, vector stores, chunking strategies, prompt
templates. This library measures pipelines; it is not one.
