from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import AuthenticatedUser, get_current_user
from app.dependencies import get_datalens
from app.models.responses import ErrorResponse
from app.services.datalens import DataLens

router = APIRouter(prefix="/schema", tags=["schema"])


@router.get(
    "/{session_id}",
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
def get_stored_schema(
    session_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> dict:
    try:
        session = datalens.sessions.get(session_id, user.tenant_id)
        if not session.schema_ready:
            raise ValueError("Schema not extracted yet. Call /connect first.")
        return session.schema
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
