"""Authentication: JWT tokens and API keys."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

from app.core.config import Settings
from app.core.deps import get_settings
from app.services.user_store import User, UserStore, get_user_store

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenPayload(BaseModel):
    sub: str
    tenant_id: str
    exp: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    tenant_name: str = Field(..., min_length=1, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    user_id: str
    email: str
    tenant_id: str


class AuthenticatedUser(BaseModel):
    user_id: str
    email: str
    tenant_id: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(user: User, settings: Settings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.auth_access_token_expire_minutes
    )
    payload = {
        "sub": user.user_id,
        "tenant_id": user.tenant_id,
        "exp": expire,
    }
    return jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm)


def decode_access_token(token: str, settings: Settings) -> AuthenticatedUser:
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret_key,
            algorithms=[settings.auth_algorithm],
        )
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        if not user_id or not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return AuthenticatedUser(
            user_id=user_id,
            tenant_id=tenant_id,
            email=payload.get("email", ""),
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc


def _resolve_user_from_api_key(
    api_key: str | None,
    user_store: UserStore,
) -> AuthenticatedUser | None:
    if not api_key:
        return None
    user = user_store.get_user_by_api_key(api_key)
    if user is None:
        return None
    return AuthenticatedUser(
        user_id=user.user_id,
        email=user.email,
        tenant_id=user.tenant_id,
    )


def get_current_user_optional(
    settings: Annotated[Settings, Depends(get_settings)],
    user_store: Annotated[UserStore, Depends(get_user_store)],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> AuthenticatedUser | None:
    if not settings.auth_require_enabled:
        return AuthenticatedUser(
            user_id="dev-user",
            email="dev@localhost",
            tenant_id="dev-tenant",
        )

    api_user = _resolve_user_from_api_key(api_key, user_store)
    if api_user is not None:
        return api_user

    if credentials is not None:
        user = decode_access_token(credentials.credentials, settings)
        stored = user_store.get_user(user.user_id)
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        user.email = stored.email
        return user

    return None


def get_current_user(
    user: Annotated[AuthenticatedUser | None, Depends(get_current_user_optional)],
) -> AuthenticatedUser:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_session_owner(session_tenant_id: str, user: AuthenticatedUser) -> None:
    if session_tenant_id != user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )
