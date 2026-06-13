"""Encrypt and decrypt database credentials at rest."""

from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings


class CredentialVault:
    def __init__(self, settings: Settings):
        self._fernet: Fernet | None = None
        if settings.credential_encryption_key:
            key = settings.credential_encryption_key.encode()
            self._fernet = Fernet(key)

    def encrypt_credentials(self, credentials: dict[str, Any]) -> str:
        payload = json.dumps(credentials).encode()
        if self._fernet is None:
            return base64.urlsafe_b64encode(payload).decode()
        return self._fernet.encrypt(payload).decode()

    def decrypt_credentials(self, token: str) -> dict[str, Any]:
        try:
            if self._fernet is None:
                raw = base64.urlsafe_b64decode(token.encode())
            else:
                raw = self._fernet.decrypt(token.encode())
            return json.loads(raw.decode())
        except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid stored credentials") from exc
