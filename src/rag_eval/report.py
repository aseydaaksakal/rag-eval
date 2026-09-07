"""Turn a run into something a person reads.

Three shapes for three audiences: a terminal table while you iterate, JSON for
whatever comes next in the pipeline, and a self-contained HTML page for the
pull request.

All of them lead with the worst examples rather than the aggregate. The mean
tells you whether something broke; the failures tell you what.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .baseline import Comparison
from .types import RunResult


def to_json(run: RunResult, path: str | Path, *, indent: int = 2) -> Path:
    """Write the full run, per-example results included."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(run.to_dict(), indent=indent, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return file


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a run written by :func:`to_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    return "\n".join([line, rule, *body])


def to_text(run: RunResult, comparison: Comparison | None = None, *, worst: int = 5) -> str:
    """Render a run as plain text."""
    out: list[str] = []
    aggregate = run.aggregate()

    headers = ["metric", "score"]
    rows = [[name, f"{value:.4f}"] for name, value in aggregate.items()]
    if comparison and comparison.deltas:
        headers.append("delta")
        for row in rows:
            delta = comparison.deltas.get(row[0])
            row.append(f"{delta:+.4f}" if delta is not None else "")
    out.append(_table(headers, rows))

    by_tag = run.by_tag()
    if by_tag:
        out.append("\nBy tag")
        tag_rows = [
            [tag, *[f"{scores.get(m, 0.0):.4f}" for m in run.metrics]]
            for tag, scores in by_tag.items()
        ]
        out.append(_table(["tag", *run.metrics], tag_rows))

    if run.failures:
        out.append(f"\n{len(run.failures)} example(s) failed to run:")
        for result in run.failures[:worst]:
            out.append(f"  {result.example_id}: {result.error}")

    if run.metrics and worst:
        primary = run.metrics[0]
        lowest = [r for r in run.worst(primary, worst) if primary in r.scores]
        if lowest:
            out.append(f"\nLowest {primary}:")
            for result in lowest:
                question = result.question[:70]
                out.append(
                    f"  {result.scores[primary]:.3f}  {result.example_id:<12} {question}"
                )

    if comparison is not None:
        out.append("")
        if comparison.passed:
            out.append("PASS: no threshold breached.")
        else:
            out.append(f"FAIL: {len(comparison.findings)} threshold(s) breached.")
            for finding in comparison.findings:
                out.append(f"  - {finding.message}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       margin: 0 auto; padding: 2.5rem 1.5rem; max-width: 62rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.5rem 0 .75rem; }
.sub { color: #6b7280; margin: 0 0 2rem; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #e5e7eb; }
th { font-weight: 600; color: #6b7280; font-size: .8rem; text-transform: uppercase;
     letter-spacing: .03em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.up { color: #15803d; } .down { color: #b91c1c; }
.banner { padding: .8rem 1rem; border-radius: 6px; margin-bottom: 1.5rem; font-weight: 600; }
.banner.pass { background: #dcfce7; color: #14532d; }
.banner.fail { background: #fee2e2; color: #7f1d1d; }
.banner ul { font-weight: 400; margin: .5rem 0 0; padding-left: 1.2rem; }
.q { max-width: 34rem; }
.bar { display: inline-block; height: .5rem; background: #93c5fd; border-radius: 2px;
       vertical-align: middle; margin-right: .4rem; }
@media (prefers-color-scheme: dark) {
  body { background: #0b0f19; color: #e5e7eb; }
  th, td { border-color: #1f2937; }
  .banner.pass { background: #14532d; color: #dcfce7; }
  .banner.fail { background: #7f1d1d; color: #fee2e2; }
}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def to_html(
    run: RunResult,
    path: str | Path,
    *,
    comparison: Comparison | None = None,
    title: str = "RAG evaluation",
    worst: int = 15,
) -> Path:
    """Write a self-contained HTML report.

    No external assets, so it can be attached to a CI job or opened from a
    file:// URL without anything else being available.
    """
    parts: list[str] = []

    if comparison is not None:
        state = "pass" if comparison.passed else "fail"
        label = "No threshold breached" if comparison.passed else "Regression detected"
        items = "".join(f"<li>{_e(f.message)}</li>" for f in comparison.findings)
        extra = f"<ul>{items}</ul>" if items else ""
        parts.append(f'<div class="banner {state}">{label}{extra}</div>')

    rows = []
    for metric, value in run.aggregate().items():
        delta = (comparison.deltas.get(metric) if comparison else None)
        if delta is None:
            cell = "<td class='num'>-</td>"
        else:
            css = "up" if delta >= 0 else "down"
            cell = f"<td class='num {css}'>{delta:+.4f}</td>"
        width = max(2, round(value * 160))
        bar = f"<span class='bar' style='width:{width}px'></span>"
        rows.append(
            f"<tr><td>{_e(metric)}</td><td class='num'>{bar}{value:.4f}</td>{cell}</tr>"
        )
    parts.append(
        "<h2>Metrics</h2><table><tr><th>Metric</th><th class='num'>Score</th>"
        f"<th class='num'>vs baseline</th></tr>{''.join(rows)}</table>"
    )

    by_tag = run.by_tag()
    if by_tag:
        head = "".join(f"<th class='num'>{_e(m)}</th>" for m in run.metrics)
        body = "".join(
            f"<tr><td>{_e(tag)}</td>"
            + "".join(f"<td class='num'>{scores.get(m, 0.0):.4f}</td>" for m in run.metrics)
            + "</tr>"
            for tag, scores in by_tag.items()
        )
        parts.append(f"<h2>By tag</h2><table><tr><th>Tag</th>{head}</tr>{body}</table>")

    if run.metrics:
        primary = run.metrics[0]
        lowest = [r for r in run.worst(primary, worst) if primary in r.scores]
        if lowest:
            body = "".join(
                f"<tr><td class='num'>{r.scores[primary]:.3f}</td>"
                f"<td>{_e(r.example_id)}</td>"
                f"<td class='q'>{_e(r.question)}</td>"
                f"<td class='q'>{_e(r.answer[:200])}</td></tr>"
                for r in lowest
            )
            parts.append(
                f"<h2>Lowest {_e(primary)}</h2><table>"
                f"<tr><th class='num'>{_e(primary)}</th><th>Example</th>"
                "<th>Question</th><th>Answer</th></tr>"
                f"{body}</table>"
            )

    if run.failures:
        body = "".join(
            f"<tr><td>{_e(r.example_id)}</td><td>{_e(r.error)}</td></tr>"
            for r in run.failures
        )
        parts.append(f"<h2>Failures</h2><table><tr><th>Example</th><th>Error</th></tr>{body}</table>")

    meta = " &middot; ".join(f"{_e(k)}: {_e(v)}" for k, v in sorted(run.metadata.items()))
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{_e(title)}</h1><p class='sub'>{len(run)} examples &middot; {meta}</p>"
        f"{''.join(parts)}</body></html>"
    )

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(document, encoding="utf-8")
    return file
