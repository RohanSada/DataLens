"""Tenant-scoped file upload storage."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings

ALLOWED_EXTENSIONS = {".sqlite", ".db"}


class UploadService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage_root = Path(settings.upload_storage_dir)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def save_upload(self, tenant_id: str, upload: UploadFile) -> str:
        if not upload.filename:
            raise ValueError("Upload filename is required")

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        tenant_dir = self.storage_root / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)

        file_id = str(uuid.uuid4())
        dest = tenant_dir / f"{file_id}{suffix}"

        size = 0
        with dest.open("wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > self.settings.max_upload_size_bytes:
                    dest.unlink(missing_ok=True)
                    raise ValueError(
                        f"Upload exceeds maximum size of {self.settings.max_upload_size_mb} MB"
                    )
                out.write(chunk)

        if size == 0:
            dest.unlink(missing_ok=True)
            raise ValueError("Upload file is empty")

        return file_id

    def resolve_path(self, tenant_id: str, file_id: str) -> Path:
        tenant_dir = self.storage_root / tenant_id
        matches = list(tenant_dir.glob(f"{file_id}.*"))
        if not matches:
            raise ValueError(f"Upload not found: {file_id}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous upload id: {file_id}")
        return matches[0]

    def delete_upload(self, tenant_id: str, file_id: str) -> None:
        path = self.resolve_path(tenant_id, file_id)
        path.unlink(missing_ok=True)
