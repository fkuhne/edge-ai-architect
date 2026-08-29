"""Reporting: tables always, plots when matplotlib is installed.

    uv run python -m edgebench.report
    uv run python -m edgebench.report --project p01-compress-cifar
    uv run python -m edgebench.report --plot results/pareto.png \\
        --x lat_p50_ms --y quality_value

The table path has no third-party dependencies on purpose -- inspecting results
should never require installing a plotting stack into a phase environment.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .store import Store

#: Columns worth seeing at a glance. Everything else stays in the database.
DEFAULT_COLUMNS = (
    "label",
    "backend",
    "device",
    "quantization",
    "lat_p50_ms",
    "lat_p95_ms",
    "tokens_per_s",
    "peak_rss_mb",
    "avg_power_w",
    "quality_value",
)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.4g}"
    return str(value)


def render_table(rows: list[sqlite3.Row], columns: tuple[str, ...] = DEFAULT_COLUMNS) -> str:
    """Fixed-width table. Right-aligns numbers so columns stay comparable."""
    if not rows:
        return "(no results yet -- run a benchmark first)"

    present = [c for c in columns if c in rows[0].keys()]
    table = [[_fmt(r[c]) for c in present] for r in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in table))
        for i, header in enumerate(present)
    ]
    numeric = [
        all(_is_number(row[i]) for row in table)
        for i in range(len(present))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        )

    out = [line(list(present)), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in table)
    return "\n".join(out)


def _is_number(text: str) -> bool:
    if text == "-":
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def plot_pareto(
    rows: list[sqlite3.Row],
    out_path: str | Path,
    x: str = "lat_p50_ms",
    y: str = "quality_value",
) -> Path | None:
    """Scatter `y` against `x`, labelled by run.

    The characteristic edge-ML plot: cost on one axis, capability on the other,
    with the frontier showing which variants are actually worth keeping.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print(
            "matplotlib is not installed in this environment.\n"
            "  uv pip install -e '.[viz]'",
            file=sys.stderr,
        )
        return None

    points = [(r[x], r[y], r["label"]) for r in rows if r[x] is not None and r[y] is not None]
    if not points:
        print(f"no rows have both {x} and {y} populated", file=sys.stderr)
        return None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs, ys, labels = zip(*points)
    ax.scatter(xs, ys, s=64, zorder=3)
    for xi, yi, label in points:
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=8)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report edgebench results")
    parser.add_argument("--db", default=None, help="path to bench.db")
    parser.add_argument("--project", default=None, help="filter by project, e.g. p01-compress-cifar")
    parser.add_argument("--label", default=None, help="filter by label (SQL LIKE)")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--plot", default=None, metavar="PATH", help="also write a scatter plot")
    parser.add_argument("--x", default="lat_p50_ms")
    parser.add_argument("--y", default="quality_value")
    args = parser.parse_args(argv)

    with Store(args.db) as db:
        rows = db.query(project=args.project, label=args.label, limit=args.limit)
        total = db.count()
        projects = db.projects()

    print(render_table(rows))
    print()
    print(f"{len(rows)} of {total} run(s) shown", end="")
    print(f" | projects: {', '.join(projects)}" if projects else "")

    if args.plot:
        written = plot_pareto(rows, args.plot, x=args.x, y=args.y)
        if written:
            print(f"plot written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
