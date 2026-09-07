"""Turn a score into a test.

An evaluation that prints numbers is a report. An evaluation that fails a
build is a test. The difference is a stored baseline and a threshold, which is
all this module is.

    rag-eval run --dataset q.jsonl --pipeline app:build --out results.json
    rag-eval baseline save results.json           # once, on a good commit
    rag-eval compare results.json                 # in CI, exits 1 on regression

Two kinds of check, and you usually want both:

* **Absolute floors** — "recall@5 must be at least 0.80". Catches slow drift
  that a per-commit delta never trips.
* **Deltas against the baseline** — "recall@5 may not drop by more than 0.02".
  Catches the change this pull request made, without waiting for it to cross a
  floor.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .metrics.registry import build
from .types import RunResult

#: Below this, a delta is noise from a small dataset rather than a regression.
DEFAULT_TOLERANCE = 0.01


@dataclass(slots=True)
class Threshold:
    """Rules for one metric.

    Args:
        metric: Metric name, e.g. ``"recall@5"``.
        minimum: Absolute floor. ``None`` disables the floor check.
        max_drop: Largest tolerated move away from the baseline in the bad
            direction. ``None`` disables the delta check.
    """

    metric: str
    minimum: float | None = None
    max_drop: float | None = None


@dataclass(slots=True)
class Finding:
    """One threshold that was not met."""

    metric: str
    kind: str  # "floor" | "drop" | "missing"
    current: float
    reference: float
    message: str


@dataclass(slots=True)
class Comparison:
    """The outcome of checking a run against a baseline."""

    findings: list[Finding] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)
    current: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "current": self.current,
            "baseline": self.baseline,
            "deltas": self.deltas,
            "findings": [asdict(f) for f in self.findings],
        }


def save_baseline(run: RunResult, path: str | Path, *, label: str | None = None) -> Path:
    """Write aggregate scores as the reference for later runs.

    Only the aggregates and per-tag means are stored, not per-example results:
    a baseline is checked into the repository and reviewed in pull requests, so
    it should stay readable.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "metrics": run.metrics,
        "aggregate": run.aggregate(),
        "by_tag": run.by_tag(),
        "example_count": len(run),
        "metadata": run.metadata,
    }
    file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Read a baseline file."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"no baseline at {file}; create one with: rag-eval baseline save <results.json>"
        )
    return json.loads(file.read_text(encoding="utf-8"))


def load_thresholds(path: str | Path) -> list[Threshold]:
    """Read thresholds from TOML.

        [thresholds."recall@5"]
        minimum = 0.80
        max_drop = 0.02

        [thresholds.hallucination]
        minimum = 0.0
        max_drop = 0.03
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    section = raw.get("thresholds", raw)
    thresholds: list[Threshold] = []
    for metric, rules in section.items():
        if not isinstance(rules, dict):
            raise ValueError(f"threshold for {metric!r} must be a table")
        thresholds.append(
            Threshold(
                metric=metric,
                minimum=rules.get("minimum"),
                max_drop=rules.get("max_drop"),
            )
        )
    return thresholds


def compare(
    run: RunResult,
    baseline: dict[str, Any] | None = None,
    thresholds: list[Threshold] | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Comparison:
    """Check *run* against a baseline and a set of thresholds.

    Direction is read from the metric registry, so ``hallucination`` going up
    is a regression while ``recall@5`` going up is not. Nobody has to remember
    to invert a sign in a config file.

    Args:
        run: The run to check.
        baseline: A loaded baseline, or ``None`` to skip delta checks.
        thresholds: Rules to apply. Without them, every shared metric is
            checked against the baseline using *tolerance*.
        tolerance: Default ``max_drop`` when a threshold does not set one.
    """
    current = run.aggregate()
    reference = dict((baseline or {}).get("aggregate", {}))
    comparison = Comparison(current=current, baseline=reference)

    for metric, value in current.items():
        if metric in reference:
            comparison.deltas[metric] = round(value - reference[metric], 6)

    rules = thresholds or [
        Threshold(metric=name, max_drop=tolerance) for name in current if name in reference
    ]

    for rule in rules:
        if rule.metric not in current:
            comparison.findings.append(
                Finding(
                    metric=rule.metric,
                    kind="missing",
                    current=0.0,
                    reference=0.0,
                    message=(
                        f"{rule.metric} has a threshold but was not computed; "
                        "add it to the metrics list or drop the threshold"
                    ),
                )
            )
            continue

        value = current[rule.metric]
        higher_is_better = _direction(rule.metric)

        if rule.minimum is not None:
            breached = value < rule.minimum if higher_is_better else value > rule.minimum
            if breached:
                relation = "below" if higher_is_better else "above"
                comparison.findings.append(
                    Finding(
                        metric=rule.metric,
                        kind="floor",
                        current=value,
                        reference=rule.minimum,
                        message=(
                            f"{rule.metric} is {value:.4f}, {relation} the "
                            f"required {rule.minimum:.4f}"
                        ),
                    )
                )

        if rule.max_drop is not None and rule.metric in reference:
            previous = reference[rule.metric]
            # A "drop" is movement in the bad direction, whichever way that is.
            drop = (previous - value) if higher_is_better else (value - previous)
            if drop > rule.max_drop:
                comparison.findings.append(
                    Finding(
                        metric=rule.metric,
                        kind="drop",
                        current=value,
                        reference=previous,
                        message=(
                            f"{rule.metric} moved {drop:.4f} the wrong way "
                            f"({previous:.4f} -> {value:.4f}), over the "
                            f"{rule.max_drop:.4f} allowance"
                        ),
                    )
                )

    return comparison


def _direction(metric: str) -> bool:
    try:
        return build(metric).higher_is_better
    except KeyError:
        # An unknown metric is assumed to be one where more is better, which
        # is the common case and fails loudly in the right direction.
        return True
