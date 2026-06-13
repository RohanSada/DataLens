"""HTTP client for the DataLens FastAPI backend."""
from __future__ import annotations

from typing import Any

import requests


class DataLensAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DataLensClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: int = 300,
        access_token: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.access_token = access_token
        self.api_key = api_key

    def set_access_token(self, token: str | None) -> None:
        self.access_token = token

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def health(self) -> dict[str, str]:
        return self._get("/health")

    def signup(self, email: str, password: str, tenant_name: str) -> dict[str, Any]:
        return self._post(
            "/auth/signup",
            {"email": email, "password": password, "tenant_name": tenant_name},
            auth=False,
        )

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._post("/auth/login", {"email": email, "password": password}, auth=False)

    def upload_db(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/upload",
            files={"file": (file_name, file_bytes)},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._parse(response)

    def connect(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/connect", payload)

    def disconnect(self, session_id: str) -> dict[str, Any]:
        return self._post("/disconnect", {"session_id": session_id})

    def test_connection(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/connect/test", payload)

    def query(self, session_id: str, question: str) -> dict[str, Any]:
        return self._post("/query", {"session_id": session_id, "question": question})

    def get_schema(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/schema/{session_id}")

    def _get(self, path: str, auth: bool = True) -> dict[str, Any]:
        try:
            headers = self._headers() if auth else {}
            response = requests.get(
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise DataLensAPIError(
                f"Could not reach API at {self.base_url}. Is the server running?"
            ) from exc
        return self._parse(response)

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        auth: bool = True,
    ) -> dict[str, Any]:
        try:
            headers = self._headers() if auth else {}
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise DataLensAPIError(
                f"Could not reach API at {self.base_url}. Is the server running?"
            ) from exc
        return self._parse(response)

    def _parse(self, response: requests.Response) -> dict[str, Any]:
        if response.ok:
            return response.json()

        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise DataLensAPIError(str(detail), status_code=response.status_code)
