from __future__ import annotations

import pickle

import pytest

from datalens.inference.voting import group_candidates, majority_index
from datalens.rewards import ExecutionReward, RewardConfig, completion_text, format_reward

GOLD = "SELECT name FROM customers WHERE country = 'DE'"


def _completion(sql: str, think: str = "reasoning") -> list[dict[str, str]]:
    return [{"role": "assistant", "content": f"<think>{think}</think>\n```sql\n{sql}\n```"}]


class TestExecutionReward:
    def test_scores_correct_executable_and_invalid(self, shop_db):
        reward = ExecutionReward()
        completions = [
            _completion("SELECT name FROM customers WHERE country = 'DE' ORDER BY name DESC"),  # correct
            _completion("SELECT name FROM customers"),  # runs, wrong
            _completion("SELECT nme FROM customers"),  # fails
            [{"role": "assistant", "content": "no idea"}],  # no SQL
        ]
        db = [str(shop_db)] * 4
        scores = reward(prompts=None, completions=completions, db_path=db, gold_sql=[GOLD] * 4)
        assert scores == [1.0, 0.1, 0.0, 0.0]

    def test_accepts_plain_string_completions_and_logs_metrics(self, shop_db):
        logged: dict[str, float] = {}
        reward = ExecutionReward(RewardConfig(executable=0.2))
        scores = reward(
            completions=["```sql\nSELECT name FROM customers\n```"],
            db_path=[str(shop_db)],
            gold_sql=[GOLD],
            log_metric=logged.__setitem__,
            trainer_state=object(),
        )
        assert scores == [0.2]
        assert logged == {"sql/exec_accuracy": 0.0, "sql/executable": 1.0}

    def test_gold_results_are_cached(self, shop_db):
        reward = ExecutionReward()
        for _ in range(3):
            reward(completions=[_completion(GOLD)], db_path=[str(shop_db)], gold_sql=[GOLD])
        assert len(reward._gold) == 1

    def test_runaway_results_are_capped_and_not_correct(self, shop_db):
        # Gold has 3 rows, so the cap is max(2, 2 * 3) = 6 rows.
        reward = ExecutionReward(RewardConfig(max_rows=2))
        gold = "SELECT name FROM customers"
        completions = [
            _completion("SELECT c.name FROM customers c, products p"),  # 6 rows, same set as gold
            _completion("SELECT c.name FROM customers c, customers d"),  # 9 rows: over the cap
        ]
        db = [str(shop_db)] * 2
        assert reward(completions=completions, db_path=db, gold_sql=[gold] * 2) == [1.0, 0.1]

    def test_failed_gold_never_rewards(self, shop_db):
        reward = ExecutionReward()
        assert reward(completions=[_completion("SELECT 1")], db_path=[str(shop_db)], gold_sql=["SELEC"]) == [
            0.1
        ]

    def test_is_picklable(self, shop_db):
        reward = ExecutionReward()
        clone = pickle.loads(pickle.dumps(reward))
        assert clone(completions=[_completion(GOLD)], db_path=[str(shop_db)], gold_sql=[GOLD]) == [1.0]
        assert clone.__name__ == "execution_reward"

    def test_length_mismatch_raises(self, shop_db):
        with pytest.raises(ValueError):
            ExecutionReward()(completions=[_completion(GOLD)], db_path=[], gold_sql=[])


def test_completion_text_variants():
    assert completion_text("x") == "x"
    assert completion_text([{"role": "assistant", "content": "y"}]) == "y"


def test_format_reward():
    good = "<think>plan</think>\n```sql\nSELECT 1\n```"
    assert format_reward(completions=[good, "SELECT 1", good + " trailing"]) == [1.0, 0.0, 0.0]


class TestVoting:
    def test_majority_wins(self):
        assert majority_index(["a", "b", "b", None, "b", "a"]) == 1

    def test_failures_never_vote(self):
        assert majority_index([None, None]) is None
        assert majority_index([None, "x"]) == 1

    def test_ties_prefer_non_empty_then_earliest(self):
        keys = ["empty", "rows", "empty", "rows"]
        empty = [True, False, True, False]
        assert majority_index(keys, empty) == 1
        assert majority_index(["a", "b"]) == 0

    def test_groups_are_ordered_by_votes(self):
        groups = group_candidates(["a", "b", "b", "c", "b", "c"])
        assert [(g.key, g.votes) for g in groups] == [("b", 3), ("c", 2), ("a", 1)]
