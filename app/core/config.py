from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "DataLens API"
    debug_mode: bool = False
    environment: str = "development"

    # Auth
    auth_secret_key: str = "change-me-in-production"
    auth_algorithm: str = "HS256"
    auth_access_token_expire_minutes: int = 60
    auth_require_enabled: bool = True

    # Default DB (optional — useful for local dev only)
    db_url: str | None = None
    db_password: str | None = None
    db_path: str | None = None

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    model_quantization: str = "none"  # none | 4bit | 8bit
    model_warmup_on_startup: bool = False
    max_new_tokens: int = 256
    sql_dialect: str = "sqlite"

    # Safety
    max_query_rows: int = 100
    query_timeout_seconds: int = 30
    max_question_length: int = 2000

    # Sessions
    session_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False

    # Upload storage
    upload_storage_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    # Credential encryption
    credential_encryption_key: str | None = None

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "60/minute"

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501"

    # Database connectivity
    postgres_ssl_mode: str = "prefer"
    mysql_ssl_disabled: bool = False

    # Observability
    sentry_dsn: str | None = None
    log_level: str = "INFO"
    log_json: bool = False

    # Docs
    enable_openapi_docs: bool = True

    # User store
    user_store_path: str = "./data/users.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


settings = Settings()
