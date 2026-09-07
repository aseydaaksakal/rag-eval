# Getting started

## Install

```bash
pip install rag-eval
```

From a clone:

```bash
git clone https://github.com/aseydaaksakal/rag-eval
cd rag-eval
pip install -e ".[dev]"
pytest
```

Python 3.11 or newer, for `tomllib`. No runtime dependencies.

## 1. Write a dataset

One JSON object per line. Start with twenty questions you already know the
answers to — questions support has actually been asked beat questions you
invented, because the failures you care about are the ones users hit.

```jsonl
{"id": "q1", "question": "How long do I have to request a refund?", "relevant_ids": ["policy-refund"], "tags": ["policy"]}
{"id": "q2", "question": "How do I cancel an order?", "relevant_ids": ["policy-cancel"], "tags": ["policy", "howto"]}
```

`id` and `question` are required. `relevant_ids` are the chunk ids a correct
retrieval should surface — whatever your index calls a document, as long as it
matches what your pipeline returns.

Keep `id` stable. Regression comparison matches examples on it, so renaming one
loses its history.

Check what you have:

```bash
rag-eval describe questions.jsonl
```

If `with_relevant_ids` is 40%, your recall number describes 40% of the dataset.
That is fine as long as you know it.

## 2. Wrap your pipeline

Any callable taking a question:

```python
# myapp/rag.py
from rag_eval import Chunk, Prediction


def build():
    retriever = load_retriever()
    llm = load_llm()

    def answer(question: str) -> Prediction:
        hits = retriever.search(question, k=5)
        return Prediction(
            answer=llm.generate(question, [h.text for h in hits]),
            chunks=[Chunk(id=h.doc_id, text=h.text, score=h.score) for h in hits],
        )

    return answer
```

Three things matter:

- **`Chunk.id` must match `relevant_ids`.** If retrieval metrics come back at
  zero on a pipeline you know works, this is why.
- **`Chunk.text` is what groundedness reads.** Empty text means every
  generation metric scores zero.
- **Order is the ranking.** MRR and nDCG read position, so pass the retriever's
  output as it came — do not sort or deduplicate first.

A factory named `build`, `make` or `create` is called once by the CLI. A
function named anything else is used directly.

## 3. Run it

```bash
rag-eval run --dataset questions.jsonl --pipeline myapp.rag:build --out results.json
```

```
metric         score
-------------  ------
recall@5       0.8125
mrr@5          0.8750
ndcg@5         0.8402
groundedness   0.9000
hallucination  0.1000

By tag
tag        recall@5  groundedness
---------  --------  ------------
howto      1.0000    1.0000
multi-hop  0.5000    0.7500
policy     0.9000    0.9500

Lowest recall@5:
  0.500  q6           How do I return a gift without a receipt?
```

From Python if you prefer:

```python
from rag_eval import evaluate, load_dataset, to_text

run = evaluate(load_dataset("questions.jsonl"), build(), metrics=["recall@5", "groundedness"])
print(to_text(run))
```

## 4. Store a baseline

Once the numbers look reasonable, freeze them:

```bash
rag-eval baseline save results.json --to baseline.json --label "$(git rev-parse --short HEAD)"
```

Commit `baseline.json`. It holds aggregates and per-tag means, not per-example
results, so it stays readable in a pull request — which matters, because
changing it is how a team agrees the numbers were supposed to move.

## 5. Fail the build on a regression

```bash
rag-eval compare results.json --baseline baseline.json --thresholds rag-eval.toml
```

Exit code 1 when a threshold is breached. That is the whole integration.

## What to do when it fails

**Read the tag breakdown first.** An aggregate that moves 0.02 can be one slice
falling off a cliff while the rest holds.

**Then read the worst examples.** The report leads with them. The mean says
something broke; the failures say what.

**Then work out which half.** Recall down means retrieval. Recall flat and
groundedness down means the generator stopped using its context — a chunking
change, a prompt change, a model swap.

**Check `refusal_rate` before celebrating.** Hallucination improving while
refusals climb is a pipeline that has learned to say "I don't know".

## Next

- [metrics.md](metrics.md) — what each metric measures and when it misleads
- [regression-testing.md](regression-testing.md) — baselines, thresholds, CI
- [judges.md](judges.md) — lexical vs LLM, writing your own
