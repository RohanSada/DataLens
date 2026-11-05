"""A scripted model that plays every agent role, and a BIRD layout with a train split."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conftest import QUESTIONS, make_shop_db
from datalens.agent import prompts as agent_prompts
from datalens.agent.config import AgentConfig
from datalens.prompts import SYSTEM_PROMPT, PromptParts

GOLD = {q["question"]: q["SQL"] for q in QUESTIONS}
WRONG = "SELECT name FROM customers"  # runs, but answers none of the questions
BROKEN = "SELECT nme FROM customers"  # no such column
ROLES = ("understanding", "reasoning_generator", "icl_generator", "fixer", "revisor", "selector")


def make_agent_bird(root: Path) -> Path:
    """``root`` with a dev split (database ``shop``) and a train split (``shop2``)."""
    make_shop_db(root / "dev" / "dev_databases" / "shop" / "shop.sqlite")
    (root / "dev" / "dev.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    make_shop_db(root / "train" / "train_databases" / "shop2" / "shop2.sqlite")
    train = [{**q, "db_id": "shop2"} for q in QUESTIONS]
    (root / "train" / "train.json").write_text(json.dumps(train), encoding="utf-8")
    return root


def target_question(prompt: PromptParts) -> str:
    """The question a prompt asks about (not the questions of its few-shot examples)."""
    block = "\n" + prompt.question_block
    for marker in ("\n# Question:\n", "\n# Question\n"):
        if marker in block:
            return block.rsplit(marker, 1)[1].split("\n", 1)[0].strip()
    return prompt.question_block.split("Question: ", 1)[1].split("\n", 1)[0].strip()


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _norm(sql: str) -> str:
    return " ".join(sql.split())


@dataclass
class Script:
    """What each role answers; the defaults behave like a decent model."""

    reasoning: Callable[[str, int], str] = lambda q, i: GOLD[q] if i % 2 == 0 else BROKEN
    icl: Callable[[str, int], str] = lambda q, i: GOLD[q]
    fixer: Callable[[str, str], str] = lambda q, sql: GOLD[q]
    revisor: Callable[[str, str], str] = lambda q, sql: sql
    selector: Callable[[str, str, str], int] = lambda q, a, b: 1 if _norm(a) == _norm(GOLD[q]) else 2
    literals: Callable[[str], list[str]] = lambda q: ["Lamp"] if "Lamp" in q else []
    json_ok: bool = True
    calls: Counter[str] = field(default_factory=Counter)

    def __call__(self, prompt: PromptParts, i: int) -> str:
        q = target_question(prompt)
        system = prompt.system
        if system == agent_prompts.UNDERSTANDING_SYSTEM:
            self.calls["understanding"] += 1
            if not self.json_ok:
                return "I think the skeleton is: something"
            payload = {
                "database_literals": self.literals(q),
                "question_skeleton": q.replace("Lamp", "<VALUE>"),
            }
            return "```json\n" + json.dumps(payload) + "\n```"
        if system == SYSTEM_PROMPT:
            self.calls["reasoning"] += 1
            return f"<think>plan</think>\n```sql\n{self.reasoning(q, i)}\n```"
        if system.startswith("# Your Role"):
            self.calls["icl"] += 1
            return f"Step 1: think.\n```sql\n{self.icl(q, i)}\n```"
        if system == agent_prompts.FIXER_SYSTEM:
            self.calls["fixer"] += 1
            sql = _between(prompt.question_block, "# Original SQL\n```sql\n", "\n```")
            return f"```sql\n{self.fixer(q, sql)}\n```"
        if system == agent_prompts.REVISOR_SYSTEM:
            self.calls["revisor"] += 1
            sql = _between(prompt.question_block, "# SQL\n```sql\n", "\n```")
            return f"Checked.\n```sql\n{self.revisor(q, sql)}\n```"
        if system == agent_prompts.SELECTOR_SYSTEM:
            self.calls["selector"] += 1
            body = prompt.question_block.split("# SQL Candidates:\n", 1)[1]
            first = _between(body, "## Candidate 1\n```sql\n", "\n```")
            second = _between(body, "## Candidate 2\n```sql\n", "\n```")
            choice = self.selector(q, first, second)
            return f"Candidate {choice} is right. \\boxed{{{choice}}}"
        raise AssertionError(f"unexpected prompt: {system[:60]!r}")


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def agent_config(bird_root: Path, tmp_path: Path, **overrides: Any) -> AgentConfig:
    base: dict[str, Any] = {
        "run_name": "agent-test",
        "name": "Fake 3B + agent",
        "params_b": 3,
        "data_root": str(bird_root),
        "batch_size": 3,
        "models": {"fake": {"backend": "openai", "model": "fake"}},
        "roles": {role: {"model": "fake"} for role in ROLES},
        "understanding": {"embedding_model": None},
        "reasoning": {"n": 4},
        "icl": {
            "variants": [
                {"style": "direct", "temperature": 0.2},
                {"style": "cot", "temperature": 0.6, "n": 2},
            ]
        },
        "pricing": {"gpu_hourly_usd": 1.5},
        "schema_cache_dir": None,
        "index_dir": str(tmp_path / "agent-index"),
    }
    return AgentConfig.model_validate(_merge(base, overrides))
