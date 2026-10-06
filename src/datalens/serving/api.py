"""HTTP API and demo page.

Run with ``datalens serve`` or ``uvicorn datalens.serving.api:create_app --factory``.
Databases are the SQLite files under ``DATALENS_DB_DIR``, either
``<db_id>/<db_id>.sqlite`` (BIRD layout) or ``<db_id>.sqlite``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from datalens import __version__
from datalens.data.schema import load_schema
from datalens.serving.engine import Engine
from datalens.serving.settings import Settings

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    db_id: str
    question: str = Field(min_length=1, max_length=2000)
    evidence: str = Field("", max_length=4000)
    samples: int | None = Field(None, ge=1, le=16, description="Self-consistency samples.")


class CandidateOut(BaseModel):
    sql: str | None
    votes: int
    ok: bool
    error: str | None
    chosen: bool


class QueryResponse(BaseModel):
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    error: str | None
    candidates: list[CandidateOut]
    model: str
    latency_ms: float


class DatabaseOut(BaseModel):
    db_id: str
    tables: list[str]


def discover_databases(db_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not db_dir.is_dir():
        return found
    for path in sorted(db_dir.glob("*/*.sqlite")):
        if path.stem == path.parent.name:
            found[path.stem] = path
    for path in sorted(db_dir.glob("*.sqlite")):
        found.setdefault(path.stem, path)
    return found


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="DataLens",
        version=__version__,
        description="Natural-language questions to SQL over SQLite, answered by a GRPO-trained model.",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )
    state: dict[str, Any] = {"engine": engine}

    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = Engine.from_settings(settings)
        return state["engine"]

    def get_db(db_id: str) -> Path:
        path = discover_databases(settings.db_dir).get(db_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"unknown database {db_id!r}")
        return path

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/databases", response_model=list[DatabaseOut])
    def list_databases() -> list[DatabaseOut]:
        out = []
        for db_id, path in discover_databases(settings.db_dir).items():
            schema = load_schema(path, num_examples=0)
            out.append(DatabaseOut(db_id=db_id, tables=schema.table_names()))
        return out

    @app.get("/v1/databases/{db_id}/schema")
    def database_schema(db_id: str) -> dict[str, str]:
        schema = load_schema(
            get_db(db_id), num_examples=settings.num_examples, cache_dir=settings.schema_cache_dir
        )
        return {"db_id": db_id, "schema": schema.render()}

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        path = get_db(req.db_id)
        samples = req.samples or settings.samples
        try:
            answer = await run_in_threadpool(
                get_engine().answer, path, req.question, evidence=req.evidence, samples=samples
            )
        except Exception as exc:  # the model server or API is down or rejected the call
            raise HTTPException(status_code=502, detail=f"model backend error: {exc}") from exc
        result = answer.result
        ok = result is not None and result.ok
        return QueryResponse(
            sql=answer.sql,
            columns=(result.columns or []) if ok and result else [],
            rows=[[_jsonable(v) for v in row] for row in (result.rows or [])] if ok and result else [],
            truncated=bool(ok and result and result.truncated),
            error=None if ok else (result.error if result else "the model returned no SQL"),
            candidates=[CandidateOut(**c.__dict__) for c in answer.candidates],
            model=answer.model,
            latency_ms=round(answer.latency_ms, 1),
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
