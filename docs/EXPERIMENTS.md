# Experiments: do small fine-tuned models beat large frontier models at text-to-SQL?

This document is the experiment plan, written before the runs: the question, the hypotheses,
the models, the metrics, the statistics and the threats to validity. Fixing these up front keeps
the analysis honest when the numbers come in.

## Question

On one well-defined task, does a small open model (1.5B to 7B parameters) fine-tuned for that
task match or beat much larger general-purpose models used zero-shot? And if it does, at what cost,
with how much task data, and does the advantage survive outside the training distribution?

The task is BIRD: natural-language questions over 95 real SQLite databases (11 in dev), graded by
executing the predicted query.

## Hypotheses

| | Hypothesis | Measured by |
|---|---|---|
| H1 | A GRPO-trained 7B model reaches the execution accuracy of a frontier model on BIRD dev. | EX with 95% CIs; paired McNemar test |
| H2 | It does so at a small fraction of the cost per query. | $ per 1k queries, accuracy-vs-cost Pareto frontier |
| H3 | Extra test-time compute (self-consistency) helps the small model more than it costs. | maj@k for k = 1 to 16 vs cost |
| H4 | The small model needs only part of BIRD train to cross the frontier model. | Learning curve over 10/25/50/100% of train |
| H5 | The advantage shrinks on a different benchmark (Spider), i.e. it is task-specific. | EX on Spider dev, same models, no retraining |

Each is falsifiable, and a "no" is a publishable answer too.

## Models

| Group | Models | How they are used |
|---|---|---|
| Base (small) | Qwen2.5-Coder 1.5B / 3B / 7B Instruct | zero-shot, the starting point |
| Fine-tuned (small) | the same, after SFT and after GRPO | `configs/train/*.yaml` |
| Frontier (large) | Claude Opus 5.5, Claude Sonnet 5.5 (API); Qwen2.5-Coder-32B (open weights) | zero-shot |

All models get the same system prompt, the same schema rendering (DDL with three example values
per column) and BIRD's external-knowledge "evidence". The one deliberate difference: SFT models are
asked for the SQL directly, because that is the only format they were trained on; every other model
is asked to reason first and then answer.

## Runs

| Run | Config | Needs |
|---|---|---|
| Zero-shot baselines | `configs/eval/base-*.yaml` | GPU |
| SFT 3B (+ 1.5B learning curve) | `configs/train/sft-*.yaml`, `configs/eval/sft-*.yaml` | GPU |
| GRPO 1.5B / 3B / 7B | `configs/train/grpo-*.yaml`, `configs/eval/grpo-*.yaml` | GPU (80GB for 7B) |
| Frontier, API | `configs/eval/frontier-claude-*.yaml` | `ANTHROPIC_API_KEY` |
| Frontier, open weights | `configs/eval/frontier-qwen2.5-coder-32b.yaml` | 80GB GPU |
| Transfer | any eval config with `--benchmark spider-dev --data-root data/spider` | Spider downloaded manually |

Learning-curve runs reuse one config with overrides, for example:

```bash
for f in 0.1 0.25 0.5; do
  datalens train sft configs/train/sft-qwen2.5-coder-1.5b.yaml \
    --set data.train_fraction=$f --set output_dir=checkpoints/sft-1.5b-f$f --set run_name=sft-1.5b-f$f
  datalens eval run configs/eval/base-qwen2.5-coder-1.5b.yaml --run-name sft-1.5b-f$f \
    --set model.model=checkpoints/sft-1.5b-f$f/merged --set model.group=fine-tuned \
    --set model.training=sft --set model.train_fraction=$f --set model.name="Qwen2.5-Coder-1.5B SFT ($f)" \
    --set prompt.reasoning=false
done
```

Subsets are nested (the 10% subset is inside the 25% subset, and so on) so the curve measures data
quantity, not which questions happened to be drawn.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| **EX** (execution accuracy) | Predicted and gold query return the same *set* of rows (official BIRD criterion). | The leaderboard metric. |
| EX by difficulty | EX on BIRD's simple / moderate / challenging labels. | Where the gains come from. |
| **Soft-F1** | Cell-level partial credit (BIRD mini-dev). | Separates "nearly right" from "unrelated". |
| **R-VES** | Reward-based valid efficiency: correct queries scored by speed relative to gold. | Correct but slow SQL is a production problem. |
| maj@k | Accuracy when k samples vote by execution result. | Value of test-time compute. |
| pass@k | Unbiased estimate that one of k samples is correct. | Ceiling for any reranker or verifier. |
| Error taxonomy | Each wrong answer is no SQL / syntax / schema / timeout / runtime / empty / wrong result. | What fine-tuning fixes. |
| Schema-link recall | Share of gold tables and columns the predicted query uses. | Separates linking errors from logic errors. |
| Cost | Tokens, p50/p95 latency, $ per 1k queries and per 1k *correct* answers. | The practical question. |

## Statistics

* **95% confidence intervals** on every EX number: percentile bootstrap over questions (10,000
  resamples). On 1,534 questions the interval is roughly ±2.5 points, so differences smaller than
  that need the paired test to mean anything.
* **Paired comparisons**: every model answers the same questions, so small-vs-large comparisons use
  McNemar's exact test on the discordant questions (one model right, the other wrong) and a paired
  bootstrap CI on the accuracy difference. This is much more sensitive than comparing two
  independent intervals.
* **Multiple comparisons**: several small-vs-large pairs are tested, so p-values are Holm-Bonferroni
  adjusted.
* **Sampling noise**: maj@k is averaged over 32 random size-k subsets of the 16 samples rather than
  taking the first k.

`datalens eval report` computes all of this; nothing in the README is computed by hand.

## Cost model

* API models: real token counts from the API response, priced from the config, including prompt
  cache reads and writes (the schema prefix is cached per database).
* Local models: GPU-seconds per question at an hourly rate set in the config. With vLLM batching this
  is throughput cost, which is the right number for batch analytics workloads; single-request
  latency is reported separately.

Prices change; the configs carry the prices used so a reader can recompute.

## Threats to validity

* **Contamination.** BIRD dev has been public since 2023. Frontier models may have seen it, which
  would inflate their scores. Spider transfer (H5) and the error analysis help, but cannot rule it out.
* **Label noise.** Some BIRD gold queries are wrong or ambiguous. All models are graded against the
  same gold, so comparisons stay fair, but absolute numbers carry a ceiling below 100%.
* **Prompt sensitivity.** One prompt is used for every model. A prompt tuned for one model family
  could move its score. The prompt is in `src/datalens/prompts.py` for anyone to vary.
* **Reasoning budget.** Claude models think internally before answering (effort `high` here); the
  small models reason in visible `<think>` tags within 2,048 tokens. Neither is capped to match the
  other; the cost columns make the trade-off visible.
* **Single seed.** GRPO results come from one training seed per size. Seed variance is a known issue
  in RL fine-tuning; re-running the 3B config with two more seeds is the cheapest robustness check.
