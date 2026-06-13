from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import AuthenticatedUser, get_current_user
from app.dependencies import get_datalens
from app.models.requests import ConnectionTestRequest, ConnectRequest, DisconnectRequest
from app.models.responses import (
    ConnectionTestResponse,
    ConnectResponse,
    DisconnectResponse,
    ErrorResponse,
)
from app.services.datalens import DataLens

router = APIRouter(tags=["connect"])


@router.post(
    "/connect",
    response_model=ConnectResponse,
    responses={400: {"model": ErrorResponse}},
)
def connect_database(
    request: ConnectRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> ConnectResponse:
    try:
        return datalens.connect(request, user)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/disconnect",
    response_model=DisconnectResponse,
    responses={400: {"model": ErrorResponse}},
)
def disconnect_database(
    request: DisconnectRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> DisconnectResponse:
    try:
        datalens.disconnect(request.session_id, user)
        return DisconnectResponse(session_id=request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/connect/test",
    response_model=ConnectionTestResponse,
    responses={400: {"model": ErrorResponse}},
)
def test_connection(
    request: ConnectionTestRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    datalens: DataLens = Depends(get_datalens),
) -> ConnectionTestResponse:
    success, message = datalens.test_connection(request)
    return ConnectionTestResponse(success=success, message=message)
