from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pymysql

from app.core.config import Settings
from app.models.requests import ConnectRequest
from app.services.credential_vault import CredentialVault
from app.services.upload_service import UploadService


@dataclass
class DatabaseSession:
    session_id: str
    tenant_id: str
    user_id: str
    db_type: str
    created_at: datetime
    expires_at: datetime
    db_id: str = ""
    sql_dialect: str = "sqlite"
    schema: dict[str, Any] = field(default_factory=dict)
    schema_ready: bool = False
    connection_id: str | None = None
    file_id: str | None = None
    encrypted_credentials: str | None = None
    connection: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("connection", None)
        data["created_at"] = self.created_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseSession:
        return cls(
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            user_id=data["user_id"],
            db_type=data["db_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            db_id=data.get("db_id", ""),
            sql_dialect=data.get("sql_dialect", "sqlite"),
            schema=data.get("schema", {}),
            schema_ready=data.get("schema_ready", False),
            connection_id=data.get("connection_id"),
            file_id=data.get("file_id"),
            encrypted_credentials=data.get("encrypted_credentials"),
            connection=None,
        )


class SessionStore(ABC):
    @abstractmethod
    def create(
        self,
        request: ConnectRequest,
        *,
        tenant_id: str,
        user_id: str,
        upload_service: UploadService,
        credential_vault: CredentialVault,
    ) -> DatabaseSession:
        ...

    @abstractmethod
    def store_schema(self, session_id: str, tenant_id: str, db_id: str, schema: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get(self, session_id: str, tenant_id: str) -> DatabaseSession:
        ...

    @abstractmethod
    def close(self, session_id: str, tenant_id: str) -> None:
        ...

    @abstractmethod
    def refresh_connection(self, session: DatabaseSession) -> Any:
        ...


class BaseSessionStore(SessionStore):
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(
        self,
        request: ConnectRequest,
        *,
        tenant_id: str,
        user_id: str,
        upload_service: UploadService,
        credential_vault: CredentialVault,
    ) -> DatabaseSession:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        sql_dialect = request.sql_dialect or self.settings.sql_dialect

        encrypted_credentials = None
        file_id = request.file_id

        if request.db_type == "sqlite":
            if request.db_path:
                raise ValueError(
                    "Direct db_path is not allowed. Upload a file and use file_id."
                )
            if not file_id:
                raise ValueError("file_id is required for sqlite connections")
            db_path = str(upload_service.resolve_path(tenant_id, file_id))
            connection = self._connect_sqlite(db_path)
        elif request.db_type in {"postgres", "mysql"}:
            self._validate_remote_credentials(request)
            encrypted_credentials = credential_vault.encrypt_credentials(
                {
                    "host": request.host,
                    "port": request.port,
                    "database": request.database,
                    "username": request.username,
                    "password": request.password,
                }
            )
            connection = self._connect_remote(request)
        else:
            raise ValueError(f"Unsupported db_type: {request.db_type}")

        session = DatabaseSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            db_type=request.db_type,
            connection=connection,
            created_at=now,
            expires_at=now + timedelta(seconds=self.settings.session_ttl_seconds),
            sql_dialect=sql_dialect,
            file_id=file_id,
            encrypted_credentials=encrypted_credentials,
        )
        return session

    def refresh_connection(self, session: DatabaseSession) -> Any:
        if session.connection is not None:
            return session.connection

        if session.db_type == "sqlite":
            if not session.file_id:
                raise ValueError("Missing file_id for sqlite session")
            from app.services.upload_service import UploadService

            upload_service = UploadService(self.settings)
            db_path = str(upload_service.resolve_path(session.tenant_id, session.file_id))
            session.connection = self._connect_sqlite(db_path)
            return session.connection

        if session.encrypted_credentials is None:
            raise ValueError("Missing credentials for remote session")

        vault = CredentialVault(self.settings)
        creds = vault.decrypt_credentials(session.encrypted_credentials)
        request = ConnectRequest(
            db_type=session.db_type,  # type: ignore[arg-type]
            host=creds["host"],
            port=creds["port"],
            database=creds["database"],
            username=creds["username"],
            password=creds["password"],
            sql_dialect=session.sql_dialect,  # type: ignore[arg-type]
        )
        session.connection = self._connect_remote(request)
        return session.connection

    def _validate_remote_credentials(self, request: ConnectRequest) -> None:
        missing = [
            name
            for name, value in {
                "host": request.host,
                "port": request.port,
                "database": request.database,
                "username": request.username,
                "password": request.password,
            }.items()
            if value in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing required fields for remote DB: {', '.join(missing)}")

    def _connect_sqlite(self, db_path: str) -> sqlite3.Connection:
        uri = f"file:{db_path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.execute("SELECT 1")
        return connection

    def _connect_remote(self, request: ConnectRequest) -> Any:
        if request.db_type == "postgres":
            sslmode = self.settings.postgres_ssl_mode
            if self.settings.is_production and sslmode == "disable":
                raise ValueError("Postgres SSL cannot be disabled in production")
            conninfo = (
                f"host={request.host} port={request.port} dbname={request.database} "
                f"user={request.username} password={request.password} "
                f"sslmode={sslmode} connect_timeout=10"
            )
            connection = psycopg.connect(conninfo)
            connection.execute("SELECT 1")
            return connection

        if request.db_type == "mysql":
            ssl = None if self.settings.mysql_ssl_disabled else {"ssl": {}}
            if self.settings.is_production and self.settings.mysql_ssl_disabled:
                raise ValueError("MySQL SSL is required in production")
            connection = pymysql.connect(
                host=request.host,
                port=request.port or 3306,
                user=request.username,
                password=request.password,
                database=request.database,
                connect_timeout=10,
                ssl=ssl,
            )
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return connection

        raise NotImplementedError(f"Unsupported db_type: {request.db_type}")

    def _close_connection(self, connection: Any) -> None:
        if connection is not None:
            connection.close()


class InMemorySessionStore(BaseSessionStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._sessions: dict[str, DatabaseSession] = {}

    def create(
        self,
        request: ConnectRequest,
        *,
        tenant_id: str,
        user_id: str,
        upload_service: UploadService,
        credential_vault: CredentialVault,
    ) -> DatabaseSession:
        session = super().create(
            request,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_service=upload_service,
            credential_vault=credential_vault,
        )
        self._sessions[session.session_id] = session
        return session

    def store_schema(self, session_id: str, tenant_id: str, db_id: str, schema: dict[str, Any]) -> None:
        session = self.get(session_id, tenant_id)
        session.db_id = db_id
        session.schema = schema
        session.schema_ready = True

    def get(self, session_id: str, tenant_id: str) -> DatabaseSession:
        self._cleanup_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Invalid or expired session: {session_id}")
        if session.tenant_id != tenant_id:
            raise ValueError(f"Invalid or expired session: {session_id}")
        session.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.settings.session_ttl_seconds
        )
        return session

    def close(self, session_id: str, tenant_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None and session.tenant_id == tenant_id:
            self._close_connection(session.connection)

    def save_session(self, session: DatabaseSession) -> None:
        self._sessions[session.session_id] = session

    def _cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired_ids = [
            sid
            for sid, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for sid in expired_ids:
            self.close(sid, self._sessions[sid].tenant_id)


class RedisSessionStore(BaseSessionStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        import redis

        self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._local_connections: dict[str, Any] = {}

    def _key(self, session_id: str) -> str:
        return f"datalens:session:{session_id}"

    def create(
        self,
        request: ConnectRequest,
        *,
        tenant_id: str,
        user_id: str,
        upload_service: UploadService,
        credential_vault: CredentialVault,
    ) -> DatabaseSession:
        session = super().create(
            request,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_service=upload_service,
            credential_vault=credential_vault,
        )
        connection = session.connection
        session.connection = None
        self.redis.setex(
            self._key(session.session_id),
            self.settings.session_ttl_seconds,
            json.dumps(session.to_dict()),
        )
        if connection is not None:
            self._local_connections[session.session_id] = connection
        return session

    def store_schema(self, session_id: str, tenant_id: str, db_id: str, schema: dict[str, Any]) -> None:
        session = self.get(session_id, tenant_id)
        session.db_id = db_id
        session.schema = schema
        session.schema_ready = True
        self._persist(session)

    def get(self, session_id: str, tenant_id: str) -> DatabaseSession:
        raw = self.redis.get(self._key(session_id))
        if raw is None:
            raise ValueError(f"Invalid or expired session: {session_id}")
        session = DatabaseSession.from_dict(json.loads(raw))
        if session.tenant_id != tenant_id:
            raise ValueError(f"Invalid or expired session: {session_id}")
        session.connection = self._local_connections.get(session_id)
        if session.connection is None:
            session.connection = self.refresh_connection(session)
            self._local_connections[session_id] = session.connection
        self.redis.expire(self._key(session_id), self.settings.session_ttl_seconds)
        return session

    def close(self, session_id: str, tenant_id: str) -> None:
        raw = self.redis.get(self._key(session_id))
        if raw is None:
            return
        session = DatabaseSession.from_dict(json.loads(raw))
        if session.tenant_id != tenant_id:
            return
        connection = self._local_connections.pop(session_id, None)
        self._close_connection(connection)
        self.redis.delete(self._key(session_id))

    def _persist(self, session: DatabaseSession) -> None:
        connection = session.connection
        session.connection = None
        self.redis.setex(
            self._key(session.session_id),
            self.settings.session_ttl_seconds,
            json.dumps(session.to_dict()),
        )
        if connection is not None:
            session.connection = connection


def create_session_store(settings: Settings) -> SessionStore:
    if settings.redis_enabled:
        return RedisSessionStore(settings)
    store = InMemorySessionStore(settings)
    return store
