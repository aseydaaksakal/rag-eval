# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/), with the pre-1.0 caveat that
minor releases may break the API.

## [Unreleased]

## [0.1.0] - 2026-09-07

First release.

### Added

- Retrieval metrics: `recall@k`, `precision@k`, `mrr@k`, `ndcg@k`,
  `hit_rate@k`, `context_waste@k`.
- Generation metrics: `groundedness`, `hallucination`, `answer_relevance`,
  `answer_correctness`, `refusal_rate`.
- `LexicalJudge`: offline, deterministic groundedness scoring with stopword
  removal and light stemming. `LLMJudge`: bring-your-own-model, with prompt
  caching and optional error swallowing.
- `evaluate()` over any callable pipeline, accepting `Prediction`, dict or
  string returns; `evaluate_predictions()` for output produced elsewhere.
- Examples missing `relevant_ids` or `reference_answer` are skipped by the
  metrics that need them rather than scored zero.
- Baselines and thresholds: absolute floors and deltas, with direction read
  from the metric registry so lower-is-better metrics cannot be inverted by a
  config typo. TOML threshold files.
- Reporting: terminal tables with deltas and worst examples, JSON, and a
  self-contained HTML page with per-tag breakdowns.
- `rag-eval` CLI: `run`, `baseline save`, `compare`, `report`, `describe`.
  `compare` exits 1 on a regression.
- Ragas interop: dataset conversion in both directions, with `faithfulness`
  mapped onto `groundedness` so baselines stay comparable. Ragas is not a
  dependency.
- Toy pipeline, dataset and pytest example under `examples/`.

[Unreleased]: https://github.com/aseydaaksakal/rag-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aseydaaksakal/rag-eval/releases/tag/v0.1.0
