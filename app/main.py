from __future__ import annotations

import uuid
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.__version__ import __version__
from app.api.routes import auth, connect, query, schema, upload
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import metrics_endpoint, metrics_middleware
from app.core.rate_limit import limiter
from app.dependencies import get_datalens

configure_logging(settings)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    docs_url="/docs" if settings.enable_openapi_docs else None,
    redoc_url="/redoc" if settings.enable_openapi_docs else None,
    openapi_url="/openapi.json" if settings.enable_openapi_docs else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.on_event("startup")
def startup_event() -> None:
    if settings.sentry_dsn:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(dsn=settings.sentry_dsn, integrations=[FastApiIntegration()])
    get_datalens()


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.middleware("http")(metrics_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("validation_error", errors=exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", error=str(exc))
    detail = str(exc) if settings.debug_mode else "Internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(connect.router)
app.include_router(query.router)
app.include_router(schema.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/health/ready")
def health_ready() -> dict[str, object]:
    datalens = get_datalens()
    checks = datalens.health_ready()
    ready = checks.get("model") is True and checks.get("redis") in (True, "disabled")
    return {"status": "ok" if ready else "degraded", "checks": checks}


@app.get("/metrics")
def metrics() -> Response:
    return metrics_endpoint()
