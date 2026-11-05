"""Agentar-Scale-SQL style orchestrated test-time scaling around a small trained model.

The pipeline replicates Wang et al., *Agentar-Scale-SQL: Advancing Text-to-SQL
through Orchestrated Test-Time Scaling* (2025), which ranked first on the BIRD
leaderboard. It has three steps:

1. **Task understanding**: find the database values the question mentions and
   retrieve few-shot examples with a similar question skeleton.
2. **SQL generation scaling**: candidates from two complementary generators
   (the RL-trained reasoning generator and an in-context-learning generator),
   then an iterative fixer and reviser driven by execution results.
3. **SQL selection scaling**: group candidates by execution result and run a
   pairwise round-robin tournament judged by a (GRPO-trained) selector.

See ``docs/AGENT.md`` for how each part maps onto the paper.
"""
