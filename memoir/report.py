"""Turning a results file into the accuracy grid and the efficiency table.

    python -m memoir.report runs/*.jsonl

Accuracy and efficiency are reported side by side on purpose. The claim a
benchmark like this can make is not just "more accurate" but "more accurate *and*
cheaper" (fewer round-trips, lower latency, fewer tokens), and that claim is only
available if the cost was measured from the start.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from memoir.scoring.score import Aggregate, Score, aggregate
from memoir.tasks import TASK_LABELS, TASKS


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL results file."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scores_from_rows(rows: Iterable[dict[str, Any]]) -> list[Score]:
    """Recover the :class:`Score` records written alongside each answer."""
    return [
        Score(
            question_id=r["question_id"],
            task=r["task"],
            scorer=r["scorer"],
            score=float(r["score"]),
            question_kind=r.get("question_kind"),
            needle_kind=r.get("needle_kind"),
            rationale=r.get("rationale", ""),
            date_grounding=r.get("date_grounding"),
            entity_grounding=r.get("entity_grounding"),
            error=r.get("error"),
        )
        for r in rows
    ]


def efficiency(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean cost per question, plus cost per unit of score actually earned.

    ``per_correct`` divides by total score rather than by a count of correct
    answers, so partial credit is spent rather than rounded away.
    """
    if not rows:
        return {}
    tokens = [float(r.get("total_tokens") or 0) for r in rows]
    latency = [float(r.get("latency_ms") or 0) / 1000.0 for r in rows]
    calls = [float(r.get("tool_calls") or 0) for r in rows]
    earned = sum(float(r.get("score") or 0.0) for r in rows) or 1.0
    return {
        "total_tokens": round(statistics.fmean(tokens), 1),
        "latency_s": round(statistics.fmean(latency), 2),
        "tool_calls": round(statistics.fmean(calls), 2),
        "tokens_per_correct": round(sum(tokens) / earned, 1),
        "calls_per_correct": round(sum(calls) / earned, 2),
    }


def _fmt_k(n: float) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else f"{n:.0f}"


def format_report(agg: Aggregate, eff: dict[str, float] | None = None) -> str:
    """A single substrate's numbers, as plain text."""
    # Say how many tasks the headline actually covers. A filtered run reports an
    # "overall" over the tasks it ran, which is not comparable to a full one.
    covered = len(agg.per_task)
    basis = (
        "mean of the four task scores"
        if covered == len(TASKS)
        else f"mean of {covered} of {len(TASKS)} tasks, partial run"
    )
    lines = [
        "",
        f"{agg.substrate or 'substrate'}: {agg.n} questions"
        + (f", {agg.errors} errored" if agg.errors else ""),
        "",
        f"  {'Overall':<26} {agg.overall:.3f}   ({basis})",
    ]
    for task in TASKS:
        if task in agg.per_task:
            label = TASK_LABELS[task]
            n = agg.per_task_n.get(task, 0)
            lines.append(f"  {label:<26} {agg.per_task[task]:.3f}   n={n}")
    if agg.per_kind:
        lines.append("")
        for kind, value in agg.per_kind.items():
            n = agg.per_kind_n.get(kind, 0)
            lines.append(f"    {kind:<24} {value:.3f}   n={n}")
    if agg.per_needle_kind:
        lines.append("")
        for kind, value in agg.per_needle_kind.items():
            n = agg.per_needle_kind_n.get(kind, 0)
            lines.append(f"    needle/{kind:<17} {value:.3f}   n={n}")
    if agg.date_grounding is not None or agg.entity_grounding is not None:
        lines.append("")
        if agg.date_grounding is not None:
            lines.append(f"  {'date grounding':<26} {agg.date_grounding:.3f}")
        if agg.entity_grounding is not None:
            lines.append(f"  {'entity grounding':<26} {agg.entity_grounding:.3f}")
    if eff:
        lines += [
            "",
            f"  {'tokens / question':<26} {_fmt_k(eff['total_tokens'])}",
            f"  {'latency / question':<26} {eff['latency_s']:.1f}s",
            f"  {'tool calls / question':<26} {eff['tool_calls']:.2f}",
            f"  {'tokens / correct':<26} {_fmt_k(eff['tokens_per_correct'])}",
            f"  {'tool calls / correct':<26} {eff['calls_per_correct']:.2f}",
        ]
    return "\n".join(lines) + "\n"


def format_grid(entries: list[tuple[Aggregate, dict[str, float]]]) -> str:
    """The accuracy grid and efficiency table across substrates, as markdown."""
    if not entries:
        return "no results\n"
    entries = sorted(entries, key=lambda e: e[0].overall, reverse=True)

    head = ["Substrate", "Overall"] + [TASK_LABELS[t] for t in TASKS]
    rows = [
        [a.substrate, f"{a.overall:.3f}"]
        + [f"{a.per_task[t]:.3f}" if t in a.per_task else "n/a" for t in TASKS]
        for a, _ in entries
    ]
    out = [_markdown_table(head, rows), ""]

    eff_head = [
        "Substrate",
        "Tokens",
        "Latency",
        "Tool calls",
        "Tokens / correct",
        "Calls / correct",
    ]
    eff_rows = [
        [
            a.substrate,
            _fmt_k(e["total_tokens"]),
            f"{e['latency_s']:.1f}s",
            f"{e['tool_calls']:.2f}",
            _fmt_k(e["tokens_per_correct"]),
            f"{e['calls_per_correct']:.2f}",
        ]
        for a, e in entries
        if e
    ]
    if eff_rows:
        out += [_markdown_table(eff_head, eff_rows), ""]

    counts = entries[0][0].per_task_n
    out.append(
        "Overall is the mean of the four task scores, equally weighted, not a mean "
        "across all questions, so no single task's size drives it. Task counts: "
        + " · ".join(f"{TASK_LABELS[t]} {counts[t]:,}" for t in TASKS if t in counts)
        + "."
    )
    return "\n".join(out) + "\n"


def _markdown_table(head: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(head[i]), *(len(r[i]) for r in rows)) if rows else len(head[i])
        for i in range(len(head))
    ]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    return "\n".join(
        [line(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        + [line(r) for r in rows]
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m memoir.report", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("results", nargs="+", type=Path, help="One or more JSONL run files.")
    p.add_argument(
        "--grid",
        action="store_true",
        help="Markdown comparison across files instead of a per-file breakdown.",
    )
    args = p.parse_args(argv)

    entries: list[tuple[Aggregate, dict[str, float]]] = []
    for path in args.results:
        rows = load_results(path)
        if not rows:
            continue
        name = rows[0].get("substrate") or path.stem
        entries.append((aggregate(scores_from_rows(rows), substrate=name), efficiency(rows)))

    if not entries:
        print("no results found")
        return 1
    if args.grid or len(entries) > 1:
        print(format_grid(entries))
    else:
        agg, eff = entries[0]
        print(format_report(agg, eff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
