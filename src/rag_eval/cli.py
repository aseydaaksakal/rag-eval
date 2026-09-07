"""``rag-eval`` on the command line.

    rag-eval run --dataset q.jsonl --pipeline myapp.rag:build --out results.json
    rag-eval baseline save results.json
    rag-eval compare results.json --thresholds rag-eval.toml
    rag-eval report results.json --html report.html
    rag-eval describe q.jsonl

``compare`` exits 1 when a threshold is breached, which is what makes it usable
as a CI step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .baseline import Comparison, compare, load_baseline, load_thresholds, save_baseline
from .dataset import load as load_dataset
from .dataset import summarize
from .report import to_html, to_json, to_text
from .runner import evaluate
from .types import ExampleResult, RunResult

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


def load_callable(spec: str) -> Any:
    """Import ``package.module:attribute``.

    If the attribute is a zero-argument factory it is called once and its
    return value used, so both of these work:

        --pipeline myapp.rag:answer      # a function taking a question
        --pipeline myapp.rag:build       # a factory returning one
    """
    if ":" not in spec:
        raise ValueError(
            f"expected 'module:attribute', got {spec!r} "
            "(for example: myapp.rag:build)"
        )
    module_name, attribute = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {attribute!r}") from exc

    if callable(target) and getattr(target, "__name__", "").startswith(("build", "make", "create")):
        try:
            return target()
        except TypeError:
            return target
    return target


def _rebuild_run(payload: dict[str, Any]) -> RunResult:
    """Reconstruct a RunResult from a results file."""
    run = RunResult(metrics=payload.get("metrics", []), metadata=payload.get("metadata", {}))
    for raw in payload.get("results", []):
        run.results.append(
            ExampleResult(
                example_id=raw["example_id"],
                question=raw.get("question", ""),
                scores=raw.get("scores", {}),
                answer=raw.get("answer", ""),
                retrieved_ids=raw.get("retrieved_ids", []),
                relevant_ids=raw.get("relevant_ids", []),
                tags=raw.get("tags", []),
                latency_ms=raw.get("latency_ms"),
                error=raw.get("error"),
            )
        )
    return run


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    pipeline = load_callable(args.pipeline)

    def progress(index: int, total: int, _: ExampleResult) -> None:
        if not args.quiet:
            print(f"\r  {index}/{total}", end="", file=sys.stderr, flush=True)

    run = evaluate(
        dataset,
        pipeline,
        metrics=args.metric or None,
        progress=progress,
        metadata={"dataset": str(args.dataset), "pipeline": args.pipeline, **_kv(args.meta)},
    )
    if not args.quiet:
        print(file=sys.stderr)

    if args.out:
        to_json(run, args.out)
        print(f"wrote {args.out}", file=sys.stderr)

    print(to_text(run))
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace) -> int:
    run = _rebuild_run(json.loads(Path(args.results).read_text(encoding="utf-8")))
    path = save_baseline(run, args.to, label=args.label)
    print(f"baseline written to {path}")
    for metric, value in run.aggregate().items():
        print(f"  {metric:<22} {value:.4f}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    run = _rebuild_run(json.loads(Path(args.results).read_text(encoding="utf-8")))
    baseline = load_baseline(args.baseline) if Path(args.baseline).exists() else None
    thresholds = load_thresholds(args.thresholds) if args.thresholds else None

    if baseline is None and thresholds is None:
        print(
            f"no baseline at {args.baseline} and no --thresholds given; "
            "nothing to check against",
            file=sys.stderr,
        )
        return EXIT_USAGE

    result = compare(run, baseline, thresholds, tolerance=args.tolerance)
    print(to_text(run, result))
    if args.html:
        to_html(run, args.html, comparison=result, title="RAG evaluation")
        print(f"\nwrote {args.html}", file=sys.stderr)
    return EXIT_OK if result.passed else EXIT_REGRESSION


def cmd_report(args: argparse.Namespace) -> int:
    run = _rebuild_run(json.loads(Path(args.results).read_text(encoding="utf-8")))
    comparison: Comparison | None = None
    if args.baseline and Path(args.baseline).exists():
        comparison = compare(run, load_baseline(args.baseline), tolerance=args.tolerance)
    if args.html:
        to_html(run, args.html, comparison=comparison, title=args.title)
        print(f"wrote {args.html}", file=sys.stderr)
    print(to_text(run, comparison))
    return EXIT_OK


def cmd_describe(args: argparse.Namespace) -> int:
    examples = load_dataset(args.dataset)
    summary = summarize(examples)
    total = summary["examples"]
    print(f"{args.dataset}: {total} examples")
    for field in ("with_relevant_ids", "with_reference_answer", "with_reference_contexts"):
        count = summary[field]
        share = f"{100 * count / total:.0f}%" if total else "0%"
        print(f"  {field:<26} {count:>5}  ({share})")
    if summary["tags"]:
        print("  tags")
        for tag, count in summary["tags"].items():
            print(f"    {tag:<24} {count:>5}")
    return EXIT_OK


def _kv(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--meta expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate a pipeline over a dataset")
    run.add_argument("--dataset", required=True, help="path to a JSONL dataset")
    run.add_argument("--pipeline", required=True, help="module:attribute to import")
    run.add_argument("--metric", action="append", help="metric name; repeatable")
    run.add_argument("--out", help="write full results as JSON")
    run.add_argument("--meta", action="append", help="key=value recorded in the run")
    run.add_argument("--quiet", action="store_true", help="suppress progress output")
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser("baseline", help="manage the reference scores")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    save = baseline_sub.add_parser("save", help="store a run as the baseline")
    save.add_argument("results", help="results JSON from `rag-eval run --out`")
    save.add_argument("--to", default="rag-eval-baseline.json")
    save.add_argument("--label", help="note stored alongside, e.g. a commit sha")
    save.set_defaults(func=cmd_baseline)

    comparison = sub.add_parser("compare", help="check a run for regressions; exits 1 if any")
    comparison.add_argument("results")
    comparison.add_argument("--baseline", default="rag-eval-baseline.json")
    comparison.add_argument("--thresholds", help="TOML file of per-metric rules")
    comparison.add_argument("--tolerance", type=float, default=0.01)
    comparison.add_argument("--html", help="also write an HTML report")
    comparison.set_defaults(func=cmd_compare)

    report = sub.add_parser("report", help="render a run")
    report.add_argument("results")
    report.add_argument("--html")
    report.add_argument("--baseline")
    report.add_argument("--tolerance", type=float, default=0.01)
    report.add_argument("--title", default="RAG evaluation")
    report.set_defaults(func=cmd_report)

    describe = sub.add_parser("describe", help="summarise a dataset's field coverage")
    describe.add_argument("dataset")
    describe.set_defaults(func=cmd_describe)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"rag-eval: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except BrokenPipeError:
        # `rag-eval report ... | head` closes the pipe early. Point stdout at
        # devnull so the interpreter's shutdown flush does not raise again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
