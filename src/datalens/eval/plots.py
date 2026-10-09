"""Report figures, rendered in a light and a dark variant.

Colour carries one job per chart. In the comparison charts it encodes the model
group (fine-tuned, frontier, base) using the first three slots of a palette
validated for colour-vision deficiency across all pairs; every chart also
labels its marks directly, so identity never depends on colour alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from datalens.data.benchmarks import benchmark_label
from datalens.eval.score import CATEGORIES

if TYPE_CHECKING:
    from datalens.eval.report import Run


@dataclass(frozen=True)
class Theme:
    suffix: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    series: tuple[str, ...]
    ramp: tuple[str, str, str]


LIGHT = Theme(
    suffix="",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    ramp=("#fcfcfb", "#86b6ef", "#184f95"),
)
DARK = Theme(
    suffix="-dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ramp=("#1a1a19", "#1c5cab", "#9ec5f4"),
)

# Fixed group -> slot mapping: colour follows the entity, never its rank.
GROUP_SLOT = {"fine-tuned": 0, "frontier": 1, "base": 2}
GROUP_LABEL = {
    "fine-tuned": "Fine-tuned (small)",
    "frontier": "Frontier (large, zero-shot)",
    "base": "Base (small, zero-shot)",
}


def _style(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor(theme.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, length=0)
    ax.grid(True, color=theme.grid, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)
    ax.title.set_color(theme.ink)


def _title(ax: Axes, theme: Theme, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=12, pad=12, color=theme.ink)


def _usd_ticks(ax: Axes) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}" if v >= 0.1 else f"${v:.3f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _figure(theme: Theme, size: tuple[float, float]) -> tuple[Any, Axes]:
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(theme.surface)
    _style(ax, theme)
    return fig, ax


def _legend(ax: Axes, theme: Theme, ncol: int = 3) -> None:
    """One-row legend along the bottom of the figure, so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) < 2:
        return  # a single series is named by the title
    fig: Any = ax.get_figure()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=ncol,
        handlelength=1.2,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)
    fig._datalens_legend = True


def _save(fig: Any, out_dir: Path, name: str, theme: Theme) -> None:
    # Reserve a fixed ~0.45in strip at the bottom for the figure legend.
    bottom = 0.45 / fig.get_figheight() if getattr(fig, "_datalens_legend", False) else 0
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out_dir / f"{name}{theme.suffix}.png", facecolor=theme.surface)
    plt.close(fig)


def plot_ex(runs: list[Run], theme: Theme) -> Any:
    """Execution accuracy per model with 95% CIs, coloured by group."""
    rows = sorted((r for r in runs if r.summary.get("greedy")), key=lambda r: r.ex() or 0)
    fig, ax = _figure(theme, (8, 0.42 * len(rows) + 1.9))
    seen: set[str] = set()
    for i, r in enumerate(rows):
        g = r.summary["greedy"]
        color = theme.series[GROUP_SLOT[r.group]]
        label = GROUP_LABEL[r.group] if r.group not in seen else None
        seen.add(r.group)
        ax.barh(i, 100 * g["ex"], height=0.34, color=color, label=label)
        lo, hi = g["ex_ci95"]
        ax.plot(
            [100 * lo, 100 * hi], [i, i], color=theme.ink_secondary, linewidth=1.5, solid_capstyle="round"
        )
        ax.text(100 * hi + 0.8, i, f"{100 * g['ex']:.1f}", va="center", color=theme.ink, fontsize=9)
    ax.set_yticks(range(len(rows)), [r.name for r in rows])
    ax.grid(False, axis="y")
    ax.set_xlim(0, 100)
    benchmarks = ", ".join(sorted({benchmark_label(r.config.benchmark) for r in rows}))
    ax.set_xlabel(f"Execution accuracy on {benchmarks} (%), greedy, with 95% CI")
    _title(ax, theme, "Accuracy by model")
    _legend(ax, theme)
    return fig


def _pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both cost (lower) and accuracy (higher)."""
    frontier = []
    best = -1.0
    for cost, acc in sorted(points):
        if acc > best:
            frontier.append((cost, acc))
            best = acc
    return frontier


def plot_cost(runs: list[Run], theme: Theme) -> Any | None:
    """Accuracy vs dollars per 1,000 queries, with the Pareto frontier."""
    points = []
    for r in runs:
        for key, ex_key, suffix in (("greedy", "ex", ""), ("sampled", "ex_self_consistency", " +SC")):
            block = r.summary.get(key)
            if block and block["cost"]["usd_per_1k_queries"]:
                points.append(
                    (block["cost"]["usd_per_1k_queries"], 100 * block[ex_key], r.name + suffix, r.group)
                )
    if len(points) < 2:
        return None
    fig, ax = _figure(theme, (8, 5))
    frontier = _pareto([(c, a) for c, a, _, _ in points])
    ax.plot(*zip(*frontier, strict=True), color=theme.muted, linewidth=1.5, drawstyle="steps-post", zorder=1)
    seen: set[str] = set()
    for cost, acc, name, group in points:
        label = GROUP_LABEL[group] if group not in seen else None
        seen.add(group)
        ax.scatter(
            cost,
            acc,
            s=80,
            color=theme.series[GROUP_SLOT[group]],
            edgecolors=theme.surface,
            linewidths=2,
            zorder=3,
            label=label,
        )
        ax.annotate(
            name,
            (cost, acc),
            xytext=(6, 4),
            textcoords="offset points",
            color=theme.ink_secondary,
            fontsize=8,
        )
    ax.set_xscale("log")
    _usd_ticks(ax)
    ax.set_xlabel("Cost per 1,000 queries (log scale)")
    ax.set_ylabel("Execution accuracy (%)")
    _title(ax, theme, "Accuracy vs. cost (grey line: Pareto frontier)")
    _legend(ax, theme)
    return fig


def plot_scaling(runs: list[Run], theme: Theme) -> Any | None:
    """Self-consistency accuracy as the number of voting samples grows."""
    sampled = [r for r in runs if r.summary.get("sampled")]
    if not sampled:
        return None
    fig, ax = _figure(theme, (8, 5))
    for slot, r in enumerate(sampled):
        maj = r.summary["sampled"]["maj_at_k"]
        ks = sorted(int(k) for k in maj)
        ys = [100 * maj[str(k)] for k in ks]
        color = theme.series[slot % len(theme.series)]
        ax.plot(
            ks,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            label=r.name,
            solid_capstyle="round",
        )
        ax.annotate(
            f"{ys[-1]:.1f}",
            (ks[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
            fontsize=9,
        )
    for r in runs:
        if r.group == "frontier" and r.ex() is not None:
            y = 100 * (r.ex() or 0)
            ax.axhline(y, color=theme.muted, linewidth=1)
            ax.annotate(
                f"{r.name} (greedy)",
                (1, y),
                xytext=(2, 3),
                textcoords="offset points",
                color=theme.ink_secondary,
                fontsize=8,
            )
    ax.set_xscale("log", base=2)
    all_ks = sorted({int(k) for r in sampled for k in r.summary["sampled"]["maj_at_k"]})
    ax.set_xticks(all_ks, [str(k) for k in all_ks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Samples per question (k)")
    ax.set_ylabel("Execution accuracy (%)")
    _title(ax, theme, "Self-consistency: majority vote over k executed samples")
    _legend(ax, theme)
    return fig


def plot_errors(runs: list[Run], theme: Theme) -> Any | None:
    """Error taxonomy as a heatmap: models x failure category, % of questions."""
    rows = [r for r in runs if r.summary.get("greedy")]
    if not rows:
        return None
    cats = [c for c in CATEGORIES if c != "correct"]
    data = []
    for r in rows:
        errors = r.summary["greedy"]["errors"]
        n = max(sum(errors.values()), 1)
        data.append([100 * errors[c] / n for c in cats])
    fig, ax = _figure(theme, (9, 0.45 * len(rows) + 2))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("seq", list(theme.ramp))
    vmax = max(max(row) for row in data) or 1
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if value == 0:
                continue  # blank cells read faster than a grid of zeros
            share = value / vmax
            dark_cell = share > 0.55 if theme is LIGHT else share < 0.45
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if dark_cell else "#0b0b0b",
            )
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), [r.name for r in rows])
    ax.spines["bottom"].set_visible(False)
    _title(ax, theme, "Why queries fail (% of all questions, greedy)")
    return fig


def plot_learning_curve(runs: list[Run], theme: Theme) -> Any | None:
    """Accuracy vs share of BIRD train used, against frontier reference lines."""
    curves: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        frac = r.config.model.train_fraction
        if frac is not None and r.ex() is not None:
            key = (
                f"{r.config.model.training} · {r.config.model.params_b:g}B"
                if r.config.model.params_b
                else r.config.model.training
            )
            curves.setdefault(key, []).append((100 * frac, 100 * (r.ex() or 0)))
    if not curves or all(len(v) < 2 for v in curves.values()):
        return None
    fig, ax = _figure(theme, (8, 5))
    for slot, (key, pts) in enumerate(sorted(curves.items())):
        pts.sort()
        xs, ys = zip(*pts, strict=True)
        ax.plot(
            xs,
            ys,
            color=theme.series[slot % len(theme.series)],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            label=key,
        )
        ax.annotate(
            f"{ys[-1]:.1f}",
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
            fontsize=9,
        )
    for r in runs:
        if r.group == "frontier" and r.ex() is not None:
            y = 100 * (r.ex() or 0)
            ax.axhline(y, color=theme.muted, linewidth=1)
            ax.annotate(
                r.name,
                (ax.get_xlim()[0], y),
                xytext=(2, 3),
                textcoords="offset points",
                color=theme.ink_secondary,
                fontsize=8,
            )
    ax.set_xlabel("Share of BIRD train used (%)")
    ax.set_ylabel("Execution accuracy (%)")
    _title(ax, theme, "How much task data does the small model need?")
    _legend(ax, theme)
    return fig


PLOTS: dict[str, Callable[[list[Run], Theme], Any]] = {
    "ex_by_model": plot_ex,
    "accuracy_vs_cost": plot_cost,
    "self_consistency": plot_scaling,
    "error_taxonomy": plot_errors,
    "learning_curve": plot_learning_curve,
}


def render_all(runs: list[Run], out_dir: Path) -> list[str]:
    """Render every chart the runs have data for; returns the base names written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in PLOTS.items():
        made = False
        for theme in (LIGHT, DARK):
            fig = fn(runs, theme)
            if fig is None:
                break
            _save(fig, out_dir, name, theme)
            made = True
        if made:
            written.append(name)
    return written


def plot_training_log(trainer_state: str | Path, out_dir: str | Path) -> list[Path]:
    """Plot reward and SQL accuracy over GRPO steps from TRL's ``trainer_state.json``."""
    import json

    history = json.loads(Path(trainer_state).read_text(encoding="utf-8"))["log_history"]
    series = {
        "Execution accuracy of rollouts": "sql/exec_accuracy",
        "Rollouts that execute": "sql/executable",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in (LIGHT, DARK):
        fig, ax = _figure(theme, (8, 4.5))
        plotted = False
        for slot, (label, key) in enumerate(series.items()):
            pts = [(h["step"], 100 * h[key]) for h in history if key in h]
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, color=theme.series[slot], linewidth=2, label=label)
                plotted = True
        if not plotted:
            plt.close(fig)
            return []
        ax.set_xlabel("Training step")
        ax.set_ylabel("% of rollouts")
        _title(ax, theme, "GRPO training: rollouts scored by execution")
        _legend(ax, theme)
        _save(fig, out, "grpo_training", theme)
        paths.append(out / f"grpo_training{theme.suffix}.png")
    return paths
