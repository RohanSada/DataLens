import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.dependencies import get_datalens
from app.main import app
from app.services.user_store import get_user_store


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_require_enabled", True)
    monkeypatch.setattr(settings, "user_store_path", str(tmp_path / "users.db"))
    get_user_store.cache_clear()
    get_datalens.cache_clear()
    with TestClient(app) as client:
        yield client


def test_protected_route_requires_auth(auth_client):
    response = auth_client.post("/query", json={"session_id": "x", "question": "test"})
    assert response.status_code == 401


def test_signup_login_and_me(auth_client):
    signup = auth_client.post(
        "/auth/signup",
        json={"email": "test@example.com", "password": "secret123", "tenant_name": "acme"},
    )
    assert signup.status_code == 200

    login = auth_client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "secret123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = auth_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "test@example.com"
