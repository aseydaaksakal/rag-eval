# Judges

A judge decides whether a sentence is supported by a passage. Generation
metrics delegate to one, so the same metric definition can be gated cheaply in
CI and inspected more carefully with a model.

## The interface

```python
class Judge(Protocol):
    name: str
    def groundedness(self, answer: str, contexts: Sequence[str]) -> float: ...
    def relevance(self, answer: str, question: str) -> float: ...
    def similarity(self, answer: str, reference: str) -> float: ...
```

Three methods, all returning `[0, 1]`.

## LexicalJudge (default)

Splits the answer into sentences and scores each one by content-word overlap
with the best-matching passage. A sentence clearing `threshold` (default 0.6)
counts as supported; `groundedness` is the share that do.

```python
from rag_eval import LexicalJudge

judge = LexicalJudge(threshold=0.6)
judge.groundedness("The refund window is 30 days.",
                   ["Refunds are accepted within a 30 day window."])   # 1.0
```

Stopwords are dropped, because overlap on "the" is not evidence. A light
stemmer strips one inflectional suffix, so `refunds` matches `refund` and
`accepted` matches `accept`.

**Best-of, not pooled.** A claim is supported when one passage supports it, not
when its words happen to be scattered across five. Pooling makes a large
retrieved set look like evidence for anything.

**What it gets wrong.** A claim restated entirely in synonyms scores low even
though the context supports it. A fluent sentence assembled from words scattered
across a passage scores high even though the passage does not support the claim.

Both are worth knowing and neither is disqualifying, because the failures are
*stable*. The absolute number is a lower bound on quality; a **change** in it
still means a change in the pipeline, which is what a regression test needs.

Debug a drop by asking which sentences failed:

```python
judge.unsupported_sentences(answer, contexts)
# ["Express shipping costs nine euros."]
```

## LLMJudge

```python
from rag_eval import LLMJudge
from openai import OpenAI

client = OpenAI()

def complete(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

run = evaluate(dataset, pipeline, judge=LLMJudge(complete, name="gpt-4o-mini"))
```

You pass a callable, so your client, credentials, retries and timeouts stay
yours. This library never talks to a provider.

Prompts are cached within a run. `temperature=0` helps but does not make a
model deterministic — expect small movement between runs, and set tolerances
above it.

For long unattended runs, `on_error="zero"` scores a failed call as 0.0 and
keeps going instead of losing the run to one timeout. Check `judge.calls`
afterwards to see what it cost.

## Which to use

| | Lexical | LLM |
|---|---|---|
| Cost | none | per call |
| Speed | instant | seconds per example |
| Deterministic | yes | approximately |
| Recognises paraphrase | no | yes |
| Works offline | yes | no |

Use lexical as the gate on every pull request and an LLM judge on a schedule,
or when a lexical drop needs a second opinion. Keep separate baselines: their
scales are different and comparing across them is meaningless.

## Writing your own

Anything with the three methods works. A cross-encoder NLI model, for instance:

```python
class NLIJudge:
    name = "nli"

    def __init__(self, model):
        self.model = model

    def groundedness(self, answer, contexts):
        from rag_eval.judges.lexical import split_sentences
        sentences = split_sentences(answer)
        if not sentences or not contexts:
            return 0.0
        entailed = sum(
            1 for s in sentences
            if max(self.model.entailment(c, s) for c in contexts) > 0.5
        )
        return entailed / len(sentences)

    def relevance(self, answer, question):
        return self.model.similarity(question, answer)

    def similarity(self, answer, reference):
        return self.model.similarity(reference, answer)
```

Set `name` — it is recorded in the run metadata, and a report that does not say
which judge produced its numbers is a report you cannot compare to anything.
