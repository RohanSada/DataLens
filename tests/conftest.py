import os

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_REQUIRE_ENABLED"] = "false"
os.environ["DEBUG_MODE"] = "true"
os.environ["REDIS_ENABLED"] = "false"

from app.dependencies import get_datalens
from app.main import app


@pytest.fixture
def client():
    get_datalens.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sqlite_db(tmp_path):
    db_path = tmp_path / "sample.sqlite"
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('Alice'), ('Bob')")
    conn.commit()
    conn.close()
    return db_path
