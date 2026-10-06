# DataLens API + demo page. CPU-only: the model runs behind an OpenAI-compatible
# server (vLLM, see docker-compose.yml) or the Anthropic API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# A small demo database so the page works before you mount your own.
COPY scripts/make_demo_db.py ./scripts/
RUN python scripts/make_demo_db.py /app/data/databases

RUN useradd --create-home --uid 10001 datalens && chown -R datalens /app
USER datalens

ENV DATALENS_DB_DIR=/app/data/databases \
    DATALENS_SCHEMA_CACHE_DIR=/tmp/datalens-schemas
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "datalens.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
