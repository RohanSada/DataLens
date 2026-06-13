from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import (
    AuthenticatedUser,
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.core.config import Settings
from app.core.deps import get_settings
from app.services.user_store import UserStore, get_user_store

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse)
def signup(
    request: SignupRequest,
    user_store: UserStore = Depends(get_user_store),
) -> UserResponse:
    if user_store.get_user_by_email(request.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = user_store.create_user(
        email=request.email,
        password_hash=hash_password(request.password),
        tenant_id=None,
    )
    return UserResponse(user_id=user.user_id, email=user.email, tenant_id=user.tenant_id)


@router.post("/login", response_model=TokenResponse)
def login(
    request: LoginRequest,
    settings: Settings = Depends(get_settings),
    user_store: UserStore = Depends(get_user_store),
) -> TokenResponse:
    user = user_store.get_user_by_email(request.email)
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user, settings)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(user: AuthenticatedUser = Depends(get_current_user)) -> UserResponse:
    return UserResponse(user_id=user.user_id, email=user.email, tenant_id=user.tenant_id)
