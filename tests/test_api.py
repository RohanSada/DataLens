from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from datalens.inference.backends import FakeBackend
from datalens.prompts import PromptConfig, PromptParts
from datalens.serving.api import create_app, discover_databases
from datalens.serving.engine import READ_ONLY_ERROR, Engine
from datalens.serving.settings import Settings


def voter(prompt: PromptParts, i: int) -> str:
    """Three of five samples agree on the German customers."""
    if "drop" in prompt.question_block.lower():
        return "```sql\nDROP TABLE customers\n```"
    answers = [
        "SELECT name FROM customers WHERE country = 'DE'",
        "SELECT name FROM customers",
        "SELECT name FROM customers WHERE country = 'DE' ORDER BY name",
        "SELECT nme FROM customers",
        "SELECT c.name FROM customers c WHERE c.country = 'DE'",
    ]
    return f"<think>...</think>\n```sql\n{answers[i % len(answers)]}\n```"


@pytest.fixture
def client(bird_root):
    settings = Settings(db_dir=bird_root / "dev" / "dev_databases", schema_cache_dir=None, samples=5)
    engine = Engine(FakeBackend(voter, name="fake-model"), prompt_config=PromptConfig(schema_cache_dir=None))
    return TestClient(create_app(settings, engine=engine))


def test_health(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_lists_databases_and_schema(client):
    dbs = client.get("/v1/databases").json()
    assert dbs == [{"db_id": "shop", "tables": ["customers", "products", "orders"]}]
    schema = client.get("/v1/databases/shop/schema").json()["schema"]
    assert "CREATE TABLE orders" in schema


def test_query_votes_by_result(client):
    res = client.post("/v1/query", json={"db_id": "shop", "question": "Customers from Germany?"})
    assert res.status_code == 200
    body = res.json()
    assert sorted(body["rows"]) == [["Ada"], ["Linus"]]
    assert body["columns"] == ["name"]
    assert body["error"] is None
    winner = next(c for c in body["candidates"] if c["chosen"])
    assert winner["votes"] == 3
    failed = [c for c in body["candidates"] if not c["ok"]]
    assert len(failed) == 1 and "no such column" in failed[0]["error"]
    assert body["model"] == "fake-model"


def test_greedy_query(client):
    body = client.post("/v1/query", json={"db_id": "shop", "question": "Germany?", "samples": 1}).json()
    assert body["sql"] == "SELECT name FROM customers WHERE country = 'DE'"
    assert len(body["candidates"]) == 1


def test_write_queries_are_refused(client):
    body = client.post("/v1/query", json={"db_id": "shop", "question": "please drop it", "samples": 2}).json()
    assert body["rows"] == []
    assert body["error"] == READ_ONLY_ERROR


def test_unknown_database_and_validation(client):
    assert client.post("/v1/query", json={"db_id": "nope", "question": "x"}).status_code == 404
    assert client.post("/v1/query", json={"db_id": "shop", "question": ""}).status_code == 422
    assert client.post("/v1/query", json={"db_id": "shop", "question": "x", "samples": 99}).status_code == 422


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200 and "<title>DataLens</title>" in res.text


@pytest.mark.parametrize(
    ("backend", "temperature", "max_tokens"),
    [("openai", 0.8, 1024), ("anthropic", None, 16_000)],
)
def test_engine_sampling_follows_the_backend(monkeypatch, shop_db, backend, temperature, max_tokens):
    """Claude rejects sampling parameters and thinks within max_tokens; local models sample."""
    created = {}
    seen = []

    class Recording:
        name = "recording"

        def generate(self, prompts, sampling):
            seen.append(sampling)
            return FakeBackend(voter).generate(prompts, sampling)

    def fake_create_backend(kind, model, **kwargs):
        created.update(kind=kind, model=model, **kwargs)
        return Recording()

    monkeypatch.setattr("datalens.serving.engine.create_backend", fake_create_backend)
    settings = Settings(_env_file=None, backend=backend, model="m", effort="low", schema_cache_dir=None)
    engine = Engine.from_settings(settings)
    assert created["kind"] == backend
    assert ("effort" in created) == (backend == "anthropic")

    engine.answer(shop_db, "Customers from Germany?", samples=1)
    engine.answer(shop_db, "Customers from Germany?", samples=3)
    greedy, sampled = seen
    assert greedy.temperature == (None if temperature is None else 0.0)
    assert sampled.temperature == temperature
    assert sampled.n == 3
    assert greedy.max_tokens == sampled.max_tokens == max_tokens


def test_discover_flat_and_nested(tmp_path, shop_db):
    (tmp_path / "flat.sqlite").write_bytes(shop_db.read_bytes())
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.sqlite").write_bytes(shop_db.read_bytes())
    assert set(discover_databases(tmp_path)) == {"flat", "nested", "shop"}
    assert discover_databases(tmp_path / "missing") == {}
