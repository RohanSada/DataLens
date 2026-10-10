# DataLens

**Can a small open model, trained with reinforcement learning, beat frontier LLMs at text-to-SQL?**

DataLens trains Qwen2.5-Coder models (1.5B to 7B) with **GRPO and an execution reward**: the model
writes SQL, the SQL runs against a real database, and the model is rewarded only when the result is
right. It then measures them against Claude Opus 5.5, Claude Sonnet 5.5 and Qwen2.5-Coder-32B on the
[BIRD](https://bird-bench.github.io/) benchmark, with paired significance tests and cost per correct
answer, and serves the trained model behind an API.

[![CI](https://github.com/RohanSada/DataLens/actions/workflows/ci.yml/badge.svg)](https://github.com/RohanSada/DataLens/actions/workflows/ci.yml)

```
question + schema ──▶ small model ──▶ 8 SQL candidates ──▶ run each (read-only) ──▶ vote by result ──▶ answer
                         ▲                                        │
                         └──────── GRPO: reward = correct? ◀──────┘   (training only)
```

## Results

> **Status: 3B runs done, frontier runs pending.** The numbers below come from
> [reports/bird-dev/results.md](reports/bird-dev/results.md), written by `datalens eval report` from
> the [Colab notebook](notebooks/colab_train_and_eval.ipynb) runs. Claude and the larger Qwen models
> haven't been run yet, so the small-vs-frontier question is still open.

| Model | Params | Training | EX % (95% CI) | Simple | Moderate | Challenging | EX with self-consistency | $ / 1k queries |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-3B | 3B | none | 37.2 (34.7–39.6) | 44.4 | 26.9 | 23.4 | 54.4 | $0.025 |
| Qwen2.5-Coder-3B | 3B | SFT | 50.8 (48.3–53.3) | 58.1 | 40.9 | 35.9 | 56.5 | $0.015 |
| Qwen2.5-Coder-3B | 3B | **GRPO** | **51.8** (49.2–54.2) | **58.4** | **42.5** | **39.3** | **60.0** | $0.013 |
| Qwen2.5-Coder-7B | 7B | **GRPO** | not run yet | | | | | |
| Qwen2.5-Coder-32B | 32B | none | not run yet | | | | | |
| Claude Sonnet 5.5 | undisclosed | none | not run yet | | | | | |
| Claude Opus 5.5 | undisclosed | none | not run yet | | | | | |

BIRD dev, 1,534 questions. EX = execution accuracy (the official BIRD metric), greedy decoding.
Self-consistency is a majority vote over 16 sampled queries by their execution result. Cost is
greedy decoding on one Colab G4 GPU (RTX PRO 6000), priced at an assumed $1.50 per GPU hour. The
full report adds Soft-F1, maj@k / pass@k curves, an error taxonomy and latency; paired McNemar tests
appear once a frontier model is in the report, and R-VES needs `datalens eval score --ves` (not run
here). See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the hypotheses and protocol.

What the 3B runs show so far:

* **Fine-tuning adds about 15 points.** GRPO lifts the 3B model from 37.2% to 51.8% EX, with
  non-overlapping confidence intervals. The biggest change is in schema errors (a table or column
  that doesn't exist), which fall from 27.0% of questions to 8.9%.
* **GRPO and SFT are close at greedy decoding.** 51.8% vs 50.8% lies inside both confidence
  intervals, so these runs don't separate them. GRPO is ahead on challenging questions (39.3% vs
  35.9%), but there are only 145 of those.
* **The gap grows with sampling.** With 16 votes GRPO reaches 60.0%, against 56.5% for SFT and
  54.4% for the base model, for $0.20 per 1k queries. The report has no paired test of GRPO
  against SFT yet.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/bird-dev/figures/self_consistency-dark.png">
  <img alt="Execution accuracy against number of voting samples for the base, SFT and GRPO 3B models" src="reports/bird-dev/figures/self_consistency.png" width="640">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/bird-dev/figures/grpo_training-dark.png">
  <img alt="Share of GRPO rollouts that execute and that are correct over 400 training steps" src="reports/bird-dev/figures/grpo_training.png" width="640">
</picture>

## What's in here

| | |
|---|---|
| **Method** | GRPO with an execution reward (Arctic-Text2SQL-R1 recipe), DAPO-style clipping and loss, LoRA, vLLM rollouts. [docs/METHOD.md](docs/METHOD.md) |
| **Data pipeline** | BIRD download and normalisation, schema rendering with example values, gold-query filtering, nested subsets for learning curves |
| **Evaluation** | Official EX semantics, Soft-F1, R-VES, bootstrap CIs, McNemar, maj@k / pass@k, error taxonomy, schema-link recall, $ and latency |
| **Comparison** | One harness, one prompt, any model: local (vLLM), OpenAI-compatible servers, Claude (with prompt caching) |
| **Serving** | FastAPI service with execution-based self-consistency, read-only sandboxed SQL, demo page, Docker + vLLM compose |
| **Quality** | Tests for every stage, incl. CPU smoke runs of SFT and GRPO through real TRL and the Claude client against the real SDK; ruff, mypy; CI on every PR |

## Quickstart

```bash
pip install -e ".[dev]"            # add [train,vllm] on a GPU machine

datalens data download --split dev --split train     # BIRD into data/bird/

# Score a model (generate on GPU, then score on CPU)
datalens eval run configs/eval/base-qwen2.5-coder-3b.yaml

# Train with GRPO, then score the result
datalens train grpo configs/train/grpo-qwen2.5-coder-3b.yaml
datalens eval run configs/eval/grpo-qwen2.5-coder-3b.yaml

# Frontier baselines (needs ANTHROPIC_API_KEY)
datalens eval run configs/eval/frontier-claude-sonnet-5-5.yaml

# Compare everything: tables, significance tests, figures
datalens eval report runs/* --out reports/bird-dev
```

No GPU? Open [`notebooks/colab_train_and_eval.ipynb`](notebooks/colab_train_and_eval.ipynb) in
Colab with an A100 runtime. Any config value can be overridden from the command line, e.g.
`--set trainer.learning_rate=2e-5` or `--set data.train_fraction=0.25`.

## Serve it

```bash
# GPU: serve your trained model with vLLM, plus the API and demo page
DATALENS_MODEL_PATH=/checkpoints/grpo-qwen2.5-coder-7b/merged docker compose --profile gpu up

# No GPU: same API, Claude as the model
DATALENS_BACKEND=anthropic DATALENS_MODEL=claude-sonnet-5-5 ANTHROPIC_API_KEY=... docker compose up api
```

Open http://localhost:8000 for the demo page, or call the API:

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"db_id": "music_store", "question": "Which country spends the most per customer?", "samples": 5}'
```

The response includes the chosen SQL, the rows, and every candidate query with its vote count.
Generated SQL runs on a read-only connection with a timeout and row cap, and anything other than a
single `SELECT` is rejected before it reaches the database. A demo database ships in the image;
mount your own SQLite files to query them.

## Repository layout

```
src/datalens/
  data/        BIRD/Spider loading, download, schema introspection and rendering
  sql/         read-only execution with timeouts, result comparison (EX, Soft-F1, R-VES), SQL parsing
  prompts.py   the one prompt every model sees
  rewards.py   execution reward for GRPO
  training/    SFT and GRPO (TRL + PEFT), data filtering
  inference/   model backends (vLLM, OpenAI-compatible, Anthropic) and self-consistency voting
  eval/        generate -> score -> metrics -> report, statistics, figures
  serving/     FastAPI app, engine, demo page
configs/       one YAML per training run and per evaluated model
notebooks/     Colab runbook for the full experiment
docs/          method and experiment plan
```

## Design decisions

* **The database is the reward model.** No learned reward, no preference data. Correctness is
  checked the same way BIRD grades it, so the training signal and the metric agree.
* **Filter the data before tuning the algorithm.** Gold queries that fail or return nothing are
  removed (an empty gold result would reward any empty query), and questions the base model always
  or never solves can be excluded because they carry no GRPO advantage.
* **One prompt for every model.** Differences in accuracy come from the models, not from prompt
  engineering per model.
* **Statistics, not leaderboard deltas.** Every model answers the same questions, so comparisons are
  paired (McNemar, paired bootstrap) and corrected for multiple tests.
* **Accuracy is reported next to cost.** The interesting claim is never "X beats Y" alone but "X beats
  Y at a tenth of the price", or doesn't.
* **Generation and scoring are separate stages.** The GPU job only writes completions (resumable);
  everything else runs on a laptop.

## Limitations

BIRD dev has been public since 2023, so frontier models may have seen it; the Spider transfer run is
there partly to check for this. Some BIRD gold queries are noisy, which caps every model below 100%.
Results come from one prompt and one training seed per model size. All of these are discussed in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#threats-to-validity).

## References

* Li et al., *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database
  Grounded Text-to-SQLs* (BIRD), NeurIPS 2023. [arXiv:2305.03111](https://arxiv.org/abs/2305.03111)
* Snowflake AI Research, *Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL*, 2025.
* Shao et al., *DeepSeekMath* (introduces GRPO), 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
* Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025.
* Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR 2023.
  [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
* Chen et al., *Evaluating Large Language Models Trained on Code* (unbiased pass@k), 2021.
  [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
* Li et al., *OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale*, 2025.
* Pourreza et al., *CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in
  Text-to-SQL*, 2024.
