"""Command-line interface: ``datalens data|train|eval|serve|ask``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="DataLens: GRPO-trained small models vs frontier LLMs on BIRD text-to-SQL.", no_args_is_help=True
)
data_app = typer.Typer(help="Download and prepare benchmarks.", no_args_is_help=True)
train_app = typer.Typer(help="Fine-tune models (SFT, GRPO).", no_args_is_help=True)
eval_app = typer.Typer(help="Generate, score and compare evaluation runs.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------- data


@data_app.command("download")
def data_download(
    dest: Annotated[Path, typer.Option(help="Where to put BIRD.")] = Path("data/bird"),
    split: Annotated[list[str], typer.Option(help="Splits to fetch (dev, train).")] = ["dev"],  # noqa: B006
    force: Annotated[bool, typer.Option(help="Download again even if the split is in place.")] = False,
) -> None:
    """Download BIRD and lay it out as data/bird/<split>/<split>.json + <split>_databases/."""
    from datalens.data.download import download_bird

    for path in download_bird(dest, split, force=force):
        typer.echo(f"ready: {path}")


@data_app.command("exclude-solved")
def data_exclude_solved(
    scored: Annotated[Path, typer.Argument(help="scored.jsonl of a sampled run on BIRD train.")],
    out: Annotated[Path, typer.Option(help="Where to write the id list.")] = Path(
        "data/grpo_exclude_ids.txt"
    ),
    low: float = 0.0,
    high: float = 1.0,
) -> None:
    """List train questions the base model always (or never) solves, to skip in GRPO."""
    from datalens.training.data import exclusion_ids_from_run

    ids = exclusion_ids_from_run(scored, low=low, high=high)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n", encoding="utf-8")
    typer.echo(f"wrote {len(ids)} ids to {out}")


# ---------------------------------------------------------------- train


SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config value, e.g. --set trainer.learning_rate=2e-5."),
]


def _train(config: Path, sets: list[str] | None, method: str) -> None:
    from datalens.training.config import TrainConfig

    cfg = TrainConfig.from_yaml(config, sets or [])
    if cfg.method != method:
        raise typer.BadParameter(f"{config} is a {cfg.method} config, not {method}")

    from datalens.training.train import run

    typer.echo(f"done: {run(cfg)}")


@train_app.command("sft")
def train_sft(
    config: Annotated[Path, typer.Argument(help="Training YAML with method: sft.")],
    set_: SetOption = None,
) -> None:
    """Supervised fine-tuning on gold SQL."""
    _train(config, set_, "sft")


@train_app.command("grpo")
def train_grpo(
    config: Annotated[Path, typer.Argument(help="Training YAML with method: grpo.")],
    set_: SetOption = None,
) -> None:
    """GRPO with the execution reward."""
    _train(config, set_, "grpo")


@train_app.command("plot")
def train_plot(
    trainer_state: Annotated[Path, typer.Argument(help="trainer_state.json from a GRPO run.")],
    out: Annotated[Path, typer.Option()] = Path("reports/figures"),
) -> None:
    """Plot rollout accuracy over GRPO training steps."""
    from datalens.eval.plots import plot_training_log

    for path in plot_training_log(trainer_state, out):
        typer.echo(f"wrote {path}")


# ---------------------------------------------------------------- eval


@eval_app.command("generate")
def eval_generate(
    config: Annotated[Path, typer.Argument(help="Eval YAML for one model.")],
    runs: Annotated[Path, typer.Option(help="Root folder for run outputs.")] = Path("runs"),
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    benchmark: Annotated[str | None, typer.Option(help="Override, e.g. spider-dev.")] = None,
    data_root: Annotated[str | None, typer.Option(help="Override the data folder.")] = None,
    run_name: Annotated[str | None, typer.Option(help="Override the run name.")] = None,
    set_: SetOption = None,
) -> None:
    """Generate completions (needs the model; resumable)."""
    from datalens.eval.config import EvalConfig
    from datalens.eval.generate import generate

    cfg = EvalConfig.from_yaml(
        config, set_ or [], limit=limit, benchmark=benchmark, data_root=data_root, run_name=run_name
    )
    typer.echo(f"run: {generate(cfg, runs)}")


@eval_app.command("score")
def eval_score(
    run_dir: Annotated[Path, typer.Argument(help="A run folder with generations.jsonl.")],
    workers: int = 8,
    timeout: Annotated[float, typer.Option(help="Per-query timeout in seconds.")] = 30.0,
    ves: Annotated[int, typer.Option(help="R-VES timing iterations (0 = skip).")] = 0,
) -> None:
    """Execute and grade a run's completions, then write metrics.json."""
    from datalens.eval.metrics import write_metrics
    from datalens.eval.score import score_run

    score_run(run_dir, workers=workers, timeout_s=timeout, ves_iterations=ves)
    summary = write_metrics(run_dir)
    greedy = summary.get("greedy")
    if greedy:
        lo, hi = greedy["ex_ci95"]
        typer.echo(f"{summary['model']}: EX {100 * greedy['ex']:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f})")
    sampled = summary.get("sampled")
    if sampled:
        typer.echo(
            f"  self-consistency@{sampled['num_samples']}: {100 * sampled['ex_self_consistency']:.1f}%"
        )


@eval_app.command("run")
def eval_run(
    config: Annotated[Path, typer.Argument(help="Eval YAML for one model.")],
    runs: Annotated[Path, typer.Option()] = Path("runs"),
    limit: Annotated[int | None, typer.Option()] = None,
    benchmark: Annotated[str | None, typer.Option()] = None,
    data_root: Annotated[str | None, typer.Option()] = None,
    run_name: Annotated[str | None, typer.Option()] = None,
    workers: int = 8,
    ves: int = 0,
    set_: SetOption = None,
) -> None:
    """Generate, then score, in one go."""
    from datalens.eval.config import EvalConfig
    from datalens.eval.generate import generate

    cfg = EvalConfig.from_yaml(
        config, set_ or [], limit=limit, benchmark=benchmark, data_root=data_root, run_name=run_name
    )
    run_dir = generate(cfg, runs)
    eval_score(run_dir, workers=workers, timeout=30.0, ves=ves)


@eval_app.command("report")
def eval_report(
    run_dirs: Annotated[list[Path], typer.Argument(help="Run folders to compare.")],
    out: Annotated[Path, typer.Option(help="Report folder.")] = Path("reports/latest"),
    no_plots: Annotated[bool, typer.Option("--no-plots")] = False,
) -> None:
    """Compare runs: headline table, paired significance tests, cost, figures."""
    from datalens.eval.report import build_report

    typer.echo(f"wrote {build_report(run_dirs, out, plots=not no_plots)}")


# ---------------------------------------------------------------- serve / ask


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run the HTTP API and demo page (configured with DATALENS_* env vars)."""
    import uvicorn

    uvicorn.run("datalens.serving.api:create_app", factory=True, host=host, port=port)


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    db: Annotated[Path, typer.Option(help="Path to a .sqlite file.")],
    evidence: Annotated[str, typer.Option(help="Optional external knowledge.")] = "",
    samples: Annotated[int, typer.Option(help="Self-consistency samples (1 = greedy).")] = 5,
) -> None:
    """Answer one question against a SQLite file using the configured model."""
    from datalens.serving.engine import Engine
    from datalens.serving.settings import Settings

    engine = Engine.from_settings(Settings())
    answer = engine.answer(db, question, evidence=evidence, samples=samples)
    typer.echo(answer.sql or "(no query)")
    if answer.result is not None and answer.result.ok:
        typer.echo(" | ".join(answer.result.columns or []))
        for row in (answer.result.rows or [])[:20]:
            typer.echo(" | ".join(str(v) for v in row))
    elif answer.result is not None:
        typer.echo(f"error: {answer.result.error}")


if __name__ == "__main__":
    app()
