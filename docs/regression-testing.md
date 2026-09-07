# Regression testing

An evaluation that prints numbers is a report. An evaluation that fails a build
is a test. The difference is a stored baseline and a threshold.

## The loop

```bash
# 1. Score the pipeline
rag-eval run --dataset eval/questions.jsonl --pipeline myapp.rag:build --out results.json

# 2. Once, on a commit you trust
rag-eval baseline save results.json --to eval/baseline.json --label "$(git rev-parse --short HEAD)"

# 3. On every change
rag-eval compare results.json --baseline eval/baseline.json --thresholds eval/rag-eval.toml
```

Step 3 exits 1 on a breach.

## Baselines

A baseline holds aggregates and per-tag means. Not per-example results — it is
checked into the repository and read in review, so it has to stay legible.

**Commit it.** A baseline that lives on someone's laptop is not a shared
standard.

**Update it deliberately, in its own commit.** When a change is meant to move
the numbers, regenerating the baseline is how the team agrees that it was
supposed to. A commit that changes the pipeline and the baseline together hides
the very thing review is for.

**Rebuild it when the dataset changes.** Adding ten hard questions lowers every
average. That is not a regression, and comparing across it will look like one.

## Thresholds

Two checks, and you usually want both.

**Floors** are absolute. They catch slow drift that a per-commit delta never
trips: twenty commits each losing 0.005 pass every delta check and lose 0.10.

**Deltas** are relative to the baseline. They catch what this pull request did,
without waiting for it to cross a floor.

```toml
[thresholds."recall@5"]
minimum = 0.80
max_drop = 0.02

[thresholds.groundedness]
minimum = 0.85
max_drop = 0.02

[thresholds.hallucination]
minimum = 0.15      # a ceiling: direction comes from the metric
max_drop = 0.02

[thresholds.refusal_rate]
minimum = 0.25
```

`minimum` on a lower-is-better metric is a ceiling. Direction is read from the
metric registry, not from the config, so no sign needs flipping and a threshold
cannot be silently inverted by a typo.

Without a `--thresholds` file, `compare` checks every metric shared with the
baseline against `--tolerance` (default 0.01).

## Choosing tolerances

Start by measuring noise rather than guessing. Run the same pipeline three
times and look at the spread. With a deterministic retriever and the lexical
judge it will be zero; with a sampling model it will not be.

Set `max_drop` above the noise floor and below what you would consider a real
problem. Too tight and the check cries wolf until someone disables it; too
loose and it never fires.

Small datasets are noisy. On 20 examples, one question flipping moves recall by
0.05, so a 0.02 tolerance is not measuring anything you can act on. Either
grow the dataset or widen the tolerance to match it.

## In CI

```yaml
name: rag-eval

on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - name: Evaluate
        run: |
          rag-eval run \
            --dataset eval/questions.jsonl \
            --pipeline myapp.rag:build \
            --out results.json \
            --meta commit=${{ github.sha }}
      - name: Check for regressions
        run: |
          rag-eval compare results.json \
            --baseline eval/baseline.json \
            --thresholds eval/rag-eval.toml \
            --html report.html
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rag-eval-report
          path: report.html
```

`if: always()` matters — the report is most useful on the run that failed.

The HTML report is self-contained, so it opens from an artifact download or a
`file://` URL with nothing else available.

If the run needs a model, the lexical judge does not — keep the LLM judge for
scheduled deeper runs and let pull requests use the offline one.

## As a pytest case

Sometimes better than a separate CI step, because it lands in the suite your
team already runs:

```python
from pathlib import Path
import pytest
from rag_eval import compare, evaluate, load_baseline, load_dataset, to_text
from rag_eval.baseline import load_thresholds

EVAL = Path("eval")
METRICS = ["recall@5", "mrr@5", "groundedness", "hallucination", "refusal_rate"]


@pytest.fixture(scope="module")
def run():
    from myapp.rag import build
    return evaluate(load_dataset(EVAL / "questions.jsonl"), build(), metrics=METRICS)


def test_no_example_failed(run):
    assert not run.failures, [f.error for f in run.failures]


def test_no_regression(run):
    result = compare(run, load_baseline(EVAL / "baseline.json"),
                     load_thresholds(EVAL / "rag-eval.toml"))
    assert result.passed, "\n" + to_text(run, result)
```

Putting `to_text(run, result)` in the assertion message means the failure output
is the report, rather than `assert False`.

See [`examples/test_rag_regression.py`](https://github.com/aseydaaksakal/rag-eval/blob/main/examples/test_rag_regression.py) for
a working version, including a test that proves the harness catches a generator
which ignores its context.

## Recording provenance

```bash
rag-eval run ... --meta commit=abc123 --meta model=gpt-4o-mini --meta index=v7
```

Stored in the run and carried into the baseline. A score without the commit,
the model and the index version is not reproducible, and six weeks later that
is the difference between "we fixed it" and "we think we fixed it".
