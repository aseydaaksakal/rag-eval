"""Score the toy pipeline and print a report.

    python examples/evaluate.py
"""

from pathlib import Path

from pipeline import build

from rag_eval import evaluate, load_dataset, to_html, to_text

HERE = Path(__file__).parent

dataset = load_dataset(HERE / "questions.jsonl")
run = evaluate(
    dataset,
    build(),
    metrics=["recall@3", "mrr@3", "ndcg@3", "groundedness", "hallucination", "refusal_rate"],
    metadata={"pipeline": "toy", "top_k": 3},
)

print(to_text(run))
to_html(run, HERE / "report.html", title="Toy pipeline")
print(f"\nHTML report: {HERE / 'report.html'}")
