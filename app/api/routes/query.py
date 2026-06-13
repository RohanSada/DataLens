from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import AuthenticatedUser, get_current_user
from app.core.config import settings
from app.dependencies import get_datalens
from app.models.requests import QueryRequest
from app.models.responses import ErrorResponse, QueryResponse
from app.services.datalens import DataLens

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def ask_database(
    request: Request,
    body: QueryRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> QueryResponse:
    if len(body.question) > settings.max_question_length:
        raise HTTPException(status_code=400, detail="Question exceeds maximum length")

    try:
        return datalens.query(body, user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query execution failed") from exc
