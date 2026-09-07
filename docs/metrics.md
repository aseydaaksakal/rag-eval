# Metrics

Every metric returns a float in `[0, 1]`. Retrieval metrics take an optional
`@k` suffix; without one the whole retrieved list is scored.

Examples that lack what a metric needs are **skipped**, not scored zero. A
metric's aggregate covers the examples it could actually be computed for.

## Retrieval

### `recall@k`

Share of an example's `relevant_ids` that appear in the top *k*.

The ceiling on everything downstream: a passage that was never retrieved cannot
be cited, however good the model is. If you gate one metric, gate this one.

Sensitive to how completely your dataset is annotated. If `relevant_ids` lists
one of three passages that would have answered the question, a retriever that
finds the other two scores zero. Annotate all of them, or accept that the
absolute number is pessimistic and only watch the delta.

### `precision@k`

Share of the top *k* that is relevant. Divides by *k*, not by how many results
came back, so a retriever returning two chunks when asked for five is not
rewarded for it.

Matters when you pay per token, or when near-miss passages distract the model.

### `mrr@k`

Reciprocal rank of the first relevant chunk: 1.0 if it is first, 0.5 if second,
0.0 if none of the top *k*.

Falls when a reranker degrades even though recall holds — the right passages are
still there, just lower.

### `ndcg@k`

Position-weighted, normalised against the best possible ordering. The most
sensitive metric to reordering, and the one that moves first when you change
how results are ranked.

The ideal is computed over `min(len(relevant), k)`, so a perfect single result
in a single slot scores 1.0 rather than being penalised for slots that do not
exist.

### `hit_rate@k`

1.0 if anything relevant is in the top *k*. Averaged, it reads as "the share of
questions we had a chance at".

### `context_waste@k`

`1 - precision`. Lower is better. Reported directly because it is the number
that translates into tokens paid for.

## Generation

These delegate to a [judge](judges.md), so the same metric can be gated cheaply
in CI and inspected more carefully with a model.

### `groundedness`

Share of the answer that the retrieved context supports.

The core hallucination signal — and it says nothing about correctness. An
answer faithfully derived from a wrong passage scores 1.0. Pair it with
`answer_correctness` when you have reference answers.

Scored against what the pipeline retrieved. When a prediction carries no chunk
texts it falls back to `reference_contexts`, which lets you evaluate a
generator with retrieval held fixed.

### `hallucination`

`1 - groundedness`. Lower is better.

Exists separately because a rate is what goes on a dashboard and in an alert,
and inverting a metric at the point of display is how thresholds end up
backwards.

### `answer_relevance`

How much of the question the answer picks up. Catches the fluent-but-off-topic
failure, where retrieval returned neighbouring passages and the model wrote
confidently about the wrong thing.

### `answer_correctness`

Agreement with `reference_answer`. Only scored for examples that have one.

### `refusal_rate`

Share of answers that look like a refusal. Lower is better.

Not a quality signal on its own — it is the metric that explains a groundedness
improvement. A pipeline that starts refusing everything scores beautifully on
hallucination and helps nobody. Put a threshold on it whenever you put one on
hallucination.

## Reading them together

| Recall | Groundedness | Likely cause |
|---|---|---|
| down | down | Retrieval broke: index, embeddings, chunking |
| flat | down | Generator stopped using its context: prompt, model, context assembly |
| down | flat | Retrieval degraded but the model is still faithful to what it got — often paired with rising refusals |
| flat | flat, correctness down | Retrieval and grounding fine, answers wrong: the source documents themselves |

The middle row is the one a retrieval-only dashboard cannot see, and it is the
most common regression after a prompt change.

## Choosing `k`

Use the `k` your pipeline actually passes to the model. `recall@20` on a
pipeline that puts five chunks in the prompt measures a retriever nobody is
running.

## Defaults

`recall@5`, `mrr@5`, `ndcg@5`, `groundedness`, `hallucination` when you name no
metrics. Reasonable to start with, not a recommendation to stop there — at
minimum add `refusal_rate`.
