from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.auth import AuthenticatedUser, get_current_user
from app.dependencies import get_datalens
from app.models.responses import ErrorResponse, UploadResponse
from app.services.datalens import DataLens

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post(
    "",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
def upload_database_file(
    request: Request,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> UploadResponse:
    try:
        file_id = datalens.upload_service.save_upload(user.tenant_id, file)
        return UploadResponse(
            file_id=file_id,
            filename=file.filename or "upload.db",
            message="Upload successful. Use file_id when calling /connect.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
