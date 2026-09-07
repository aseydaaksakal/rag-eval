# rag-eval

Regression testing for RAG pipelines: retrieval metrics, hallucination tracking, and a baseline you can fail a build against.

<div class="grid cards" markdown>

-   **[Playground →](playground.html)**

    Runs the real library in your browser. Paste a dataset and your pipeline's output, get scores. Nothing to install.

-   **[Latest report →](report.html)**

    The HTML report the CLI produces, regenerated from the toy pipeline on every push.

-   **[Source on GitHub →](https://github.com/aseydaaksakal/rag-eval)**

    MIT licensed. `pip install -e ".[dev]" && pytest`.

</div>

## What it does

```bash
rag-eval run --dataset questions.jsonl --pipeline myapp.rag:build --out results.json
rag-eval baseline save results.json          # once, on a commit you trust
rag-eval compare results.json                # in CI — exits 1 on regression
```

```
metric         score   delta
-------------  ------  -------
recall@5       0.8125  -0.0625
groundedness   0.7200  -0.1800
hallucination  0.2800  +0.1800

FAIL: 1 threshold breached.
  - groundedness moved 0.1800 the wrong way (0.9000 -> 0.7200), over the 0.0200 allowance
```

Three things make it a test rather than a report:

- **A stored baseline.** A score is only meaningful against a previous score.
- **Direction awareness.** `hallucination` rising is a regression; `recall` rising is not. That lives in the metric, not in a config file where a typo can invert a gate.
- **Retrieval and generation measured separately.** A generator that stops using its context leaves every retrieval metric flat and only moves groundedness — exactly the failure a retrieval-only dashboard cannot see.

The default judge is deterministic and offline, so the whole thing runs in CI without a model call. Start with [Getting started](getting-started.md).
