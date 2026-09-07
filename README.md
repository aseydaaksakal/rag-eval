# rag-eval

Regression testing for RAG pipelines. Retrieval metrics, hallucination tracking, and a baseline you can fail a build against.

The default judge is deterministic and offline, so the whole thing runs in CI without a model call.

```bash
rag-eval run --dataset questions.jsonl --pipeline myapp.rag:build --out results.json
rag-eval baseline save results.json          # once, on a commit you trust
rag-eval compare results.json                # in CI — exits 1 on regression
```

```
metric         score   delta
-------------  ------  -------
recall@5       0.8125  -0.0625
mrr@5          0.8750  -0.0125
groundedness   0.7200  -0.1800
hallucination  0.2800  +0.1800

FAIL: 1 threshold breached.
  - groundedness moved 0.1800 the wrong way (0.9000 -> 0.7200), over the 0.0200 allowance
```

---

## Why this exists

Most RAG evaluation is a notebook that prints an average. It tells you the number today and nothing about whether today is worse than last week.

Three things make the difference between a report and a test:

**A stored baseline.** A score is only meaningful against a previous score. Without one, "groundedness 0.72" is a fact with no consequence.

**Direction awareness.** `hallucination` going up is a regression; `recall@5` going up is not. Encoding that in the metric rather than in a config file means nobody has to remember to flip a sign, and a threshold cannot be silently inverted.

**Metrics that separate retrieval from generation.** When a pipeline gets worse, the first question is which half broke. A generator that stops using its context leaves every retrieval metric flat and only moves groundedness — which is exactly the failure a retrieval-only dashboard cannot see.

## Install

```bash
pip install rag-eval          # once published
pip install -e ".[dev]"       # from a clone, with test extras
```

Python 3.11+ (uses `tomllib` from the standard library). No runtime dependencies.

## The dataset

JSONL, one question per line. It diffs cleanly in review and survives a bad line without taking the file with it.

```jsonl
{"id": "q1", "question": "How long do I have to request a refund?", "relevant_ids": ["policy-refund"], "reference_answer": "Thirty days from delivery.", "tags": ["policy"]}
{"id": "q2", "question": "How do I return a gift without a receipt?", "relevant_ids": ["policy-refund", "policy-gift"], "tags": ["policy", "multi-hop"]}
```

Only `id` and `question` are required. Which optional fields you fill in decides which metrics can be computed:

| Field | Unlocks |
|---|---|
| `relevant_ids` | recall, precision, MRR, nDCG, hit rate |
| `reference_answer` | `answer_correctness` |
| `reference_contexts` | groundedness for generator-only runs |
| `tags` | per-slice breakdowns |

Examples missing what a metric needs are **skipped, not scored zero**. Scoring them zero drags the aggregate down in proportion to how incompletely the dataset is annotated, which looks exactly like a regression and is not one.

Check coverage before you trust a number:

```bash
$ rag-eval describe questions.jsonl
questions.jsonl: 8 examples
  with_relevant_ids              8  (100%)
  with_reference_answer          4  (50%)
  tags
    policy                       5
    shipping                     3
    multi-hop                    1
```

## The pipeline

Any callable taking a question. Return a `Prediction`, a dict, or a bare string — whichever you already have:

```python
from rag_eval import Chunk, Prediction

def answer(question: str) -> Prediction:
    hits = my_retriever.search(question, k=5)
    return Prediction(
        answer=my_llm.generate(question, hits),
        chunks=[Chunk(id=h.doc_id, text=h.text, score=h.score) for h in hits],
    )
```

`Chunk.id` has to match the ids in `relevant_ids` — whatever your index calls a document, as long as it is consistent. `Chunk.text` is what groundedness reads. Chunk order is the ranking, so pass the retriever's output as it came.

If the pipeline runs somewhere this process cannot reach, score its output instead:

```python
from rag_eval import evaluate_predictions
run = evaluate_predictions(dataset, json.load(open("nightly.json")))
```

## Metrics

**Retrieval** — all take an optional `@k`:

| Metric | Question it answers |
|---|---|
| `recall@k` | Did the passages we need make it into the context? This is the ceiling on answer quality. |
| `mrr@k` | How far down does the first useful passage sit? |
| `ndcg@k` | Position-weighted quality; the most sensitive to reranking changes. |
| `precision@k` | How much of the context window is wasted? |
| `hit_rate@k` | Did we find anything at all? |
| `context_waste@k` | `1 - precision`, reported directly because that is what costs tokens. |

**Generation** — these use a judge:

| Metric | Question it answers |
|---|---|
| `groundedness` | What share of the answer does the retrieved context support? |
| `hallucination` | `1 - groundedness`, so a dashboard threshold reads the right way round. |
| `answer_relevance` | Did it answer the question that was asked? |
| `answer_correctness` | Does it agree with the reference answer? |
| `refusal_rate` | How often does it decline? |

`refusal_rate` is not a quality signal on its own — it is the metric that explains a groundedness improvement. A pipeline that starts refusing everything scores beautifully on hallucination and helps nobody.

Watch `recall@k` and `groundedness` together. Recall drops mean retrieval broke. Groundedness dropping while recall holds means the generator stopped using what it was given.

## Judges

The default is `LexicalJudge`: it splits the answer into sentences and asks whether each one's content words appear in some retrieved passage. Free, offline, and returns the same number every time.

It is worse than a model at recognising paraphrase and better at everything a regression test needs. A claim restated entirely in synonyms scores low; a fluent sentence assembled from scattered words scores high. Both show up in the score distribution rather than hiding, and neither moves between runs — so a *change* still means a change in the pipeline.

For a deeper read, bring your own model:

```python
from rag_eval import LLMJudge

judge = LLMJudge(lambda prompt: client.complete(prompt))
run = evaluate(dataset, pipeline, judge=judge)
```

You pass a callable, so your client, credentials, retries and timeouts stay yours. Prompts are cached within a run.

Already using [Ragas](https://github.com/explodinggradients/ragas)? Keep it, and keep one gate:

```python
from rag_eval.ragas_adapter import to_ragas_dataset, from_ragas_scores

rows = to_ragas_dataset(dataset, predictions)
scores = ragas.evaluate(Dataset.from_list(rows), metrics=[faithfulness])
run = from_ragas_scores(dataset, scores.to_pandas().to_dict("records"))
compare(run, load_baseline("rag-eval-baseline.json"))
```

`faithfulness` maps onto `groundedness`, so a baseline recorded natively still lines up. Ragas is not a dependency — the adapter only moves dicts around.

## Thresholds

Two kinds of check, and you usually want both. Floors catch slow drift that a per-commit delta never trips; deltas catch what this pull request did, without waiting for it to cross a floor.

```toml
# rag-eval.toml
[thresholds."recall@5"]
minimum = 0.80      # absolute floor
max_drop = 0.02     # tolerated move away from the baseline

[thresholds.hallucination]
minimum = 0.20      # a ceiling here — direction comes from the metric
max_drop = 0.02

[thresholds.refusal_rate]
minimum = 0.25
```

```bash
rag-eval compare results.json --thresholds rag-eval.toml --html report.html
```

## In CI

```yaml
- run: pip install -e ".[dev]"
- run: |
    rag-eval run --dataset eval/questions.jsonl \
      --pipeline myapp.rag:build --out results.json \
      --meta commit=${{ github.sha }}
- run: rag-eval compare results.json --thresholds eval/rag-eval.toml --html report.html
- uses: actions/upload-artifact@v4
  if: always()
  with: { name: rag-eval-report, path: report.html }
```

`compare` exits 1 on a breach, so the step fails on its own. The HTML report is self-contained — no external assets — so it opens from an artifact download or a `file://` URL.

Or keep it in the suite your team already runs:

```python
def test_no_rag_regression():
    run = evaluate(load_dataset("eval/questions.jsonl"), build(), metrics=METRICS)
    result = compare(run, load_baseline("eval/baseline.json"),
                     load_thresholds("eval/rag-eval.toml"))
    assert result.passed, "\n" + to_text(run, result)
```

See [`examples/test_rag_regression.py`](examples/test_rag_regression.py) for the full version, including a test that proves the harness catches a generator which ignores its context.

## Reading a failure

Start with the tag breakdown. An overall score that barely moves can hide a slice that fell apart:

```
By tag
tag        recall@3  groundedness
---------  --------  ------------
howto      1.0000    1.0000
multi-hop  0.5000    1.0000
policy     0.9000    1.0000
```

Then read the worst examples — `to_text` and the HTML report both lead with them, because the mean tells you something broke and the failures tell you what:

```
Lowest recall@3:
  0.500  q6           How do I return a gift without a receipt?
```

Programmatically:

```python
for result in run.worst("groundedness", limit=10):
    print(result.example_id, result.scores["groundedness"], result.answer[:100])
```

## Try it

The repository ships a toy pipeline so the examples run with nothing installed:

```bash
python examples/evaluate.py           # scores it, writes examples/report.html
pytest examples/test_rag_regression.py
```

## Documentation

| Guide | Covers |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Dataset, pipeline, first run |
| [docs/metrics.md](docs/metrics.md) | What each metric measures and when it misleads |
| [docs/regression-testing.md](docs/regression-testing.md) | Baselines, thresholds, CI |
| [docs/judges.md](docs/judges.md) | Lexical vs LLM, writing your own |

## Status

Early. Tested and working; not yet stable across minor versions. Pre-1.0 releases may break the API, and the changelog will say so.

Not implemented: async pipelines, concurrent evaluation, multi-turn conversations, statistical significance testing on deltas.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
