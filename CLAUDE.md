# Working in this repository

Python package in `src/rag_eval` (stdlib only — no runtime dependencies), docs in `docs/` (mkdocs-material, published to Pages), in-browser demo in `docs/playground.html` (Pyodide loads the wheel built in CI), runnable toy pipeline in `examples/`.

## Before you finish any task
- `pytest`, `ruff check .` and `pytest examples/test_rag_regression.py` pass.
- Numeric behaviour is pinned by hand-checkable examples (`mrr(["x","a"], ["a"]) == 0.5`), not float snapshots. Edge cases covered: empty retrieval, empty relevant set, k larger than the list, perfect and zero scores.
- Docs updated in the same change; a new metric is documented in `docs/metrics.md` **including when it misleads**. `CHANGELOG.md` gets a line under Unreleased.

## Design rules that do not change
- Metric direction lives in the registry (`higher_is_better`); thresholds never flip signs.
- Examples missing what a metric needs are skipped, never scored zero.
- The default judge stays offline and deterministic. Anything that needs a network call is opt-in and never required by the test suite.
- Optional integrations (Ragas, LangChain, vector stores) must import lazily and be testable without the dependency installed, with the real integration exercised in a separate CI job that installs it.
- The core keeps zero runtime dependencies; integrations go in `rag_eval/integrations/` with their own optional extra in `pyproject.toml`.

## Style
Ruff, line length 100, Python 3.11+. Comments explain why, not what. No emoji, no marketing language in docs.
