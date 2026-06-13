from __future__ import annotations

from app.core.auth import AuthenticatedUser
from app.core.config import Settings
from app.models.requests import ConnectRequest, QueryRequest
from app.models.responses import ConnectResponse, QueryResponse
from app.services.audit_logger import AuditLogger
from app.services.credential_vault import CredentialVault
from app.services.db_connection import SessionStore, create_session_store
from app.services.schema_service import SchemaService
from app.services.sql_executor import SqlExecutor
from app.services.text_to_sql import TextToSqlService
from app.services.upload_service import UploadService


class DataLens:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: SessionStore = create_session_store(settings)
        self.schema_service = SchemaService()
        self.text_to_sql = TextToSqlService(settings)
        self.sql_executor = SqlExecutor(settings)
        self.upload_service = UploadService(settings)
        self.credential_vault = CredentialVault(settings)
        self.audit_logger = AuditLogger(settings)

    def connect(self, request: ConnectRequest, user: AuthenticatedUser) -> ConnectResponse:
        session = self.sessions.create(
            request,
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            upload_service=self.upload_service,
            credential_vault=self.credential_vault,
        )

        db_id = self.schema_service.resolve_db_id(request)
        schema = self.schema_service.extract_bird_schema(session, db_id)
        self.sessions.store_schema(session.session_id, user.tenant_id, db_id, schema)

        self.audit_logger.log(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            session_id=session.session_id,
            action="connect",
            details={"db_type": session.db_type, "db_id": db_id},
        )

        table_count = len(schema["table_names_original"])
        return ConnectResponse(
            session_id=session.session_id,
            db_id=db_id,
            db_type=session.db_type,
            schema_ready=True,
            table_count=table_count,
            message=(
                f"Connected to '{db_id}' and extracted schema "
                f"for {table_count} table(s). Ready for questions."
            ),
        )

    def query(self, request: QueryRequest, user: AuthenticatedUser) -> QueryResponse:
        session = self.sessions.get(request.session_id, user.tenant_id)

        if not session.schema_ready or not session.schema:
            raise ValueError(
                "Schema is not ready for this session. Call /connect first."
            )

        schema_context = self.schema_service.render_for_llm(session.schema)

        generated_sql = self.text_to_sql.generate(
            question=request.question,
            schema_context=schema_context,
            sql_dialect=session.sql_dialect,
        )

        connection = self.sessions.refresh_connection(session)
        columns, rows = self.sql_executor.execute(
            connection,
            generated_sql,
            dialect=session.sql_dialect,
        )

        self.audit_logger.log(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            session_id=session.session_id,
            action="query",
            details={
                "question": request.question,
                "generated_sql": generated_sql,
                "row_count": len(rows),
            },
        )

        return QueryResponse(
            question=request.question,
            generated_sql=generated_sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )

    def disconnect(self, session_id: str, user: AuthenticatedUser) -> None:
        self.sessions.close(session_id, user.tenant_id)
        self.audit_logger.log(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            session_id=session_id,
            action="disconnect",
            details={},
        )

    def test_connection(self, request) -> tuple[bool, str]:
        connect_request = ConnectRequest(
            db_type=request.db_type,
            host=request.host,
            port=request.port,
            database=request.database,
            username=request.username,
            password=request.password,
            sql_dialect=request.db_type,
        )
        store = create_session_store(self.settings)
        try:
            session = store.create(
                connect_request,
                tenant_id="connection-test",
                user_id="connection-test",
                upload_service=self.upload_service,
                credential_vault=self.credential_vault,
            )
            store.close(session.session_id, "connection-test")
            return True, "Connection successful"
        except Exception as exc:
            return False, str(exc)

    def health_ready(self) -> dict[str, object]:
        checks: dict[str, object] = {"model": self.text_to_sql.is_ready}
        if self.settings.redis_enabled:
            try:
                import redis

                client = redis.Redis.from_url(self.settings.redis_url)
                client.ping()
                checks["redis"] = True
            except Exception as exc:
                checks["redis"] = False
                checks["redis_error"] = str(exc)
        else:
            checks["redis"] = "disabled"
        if self.text_to_sql.load_error:
            checks["model_error"] = self.text_to_sql.load_error
        return checks
