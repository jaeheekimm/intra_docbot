"""
src/api/auth.py
─────────────────────────────────────────────────────────────────────────────
FastAPI Bearer Token 인증 의존성.

VALID_TOKENS 환경변수에 쉼표 구분으로 복수 토큰 지원.
예: VALID_TOKENS=dev-secret-token,prod-token-abc
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()
logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=True)

# 유효 토큰 집합 (시작 시 1회 로드)
_VALID_TOKENS: set[str] = {
    t.strip()
    for t in os.getenv("VALID_TOKENS", "dev-secret-token").split(",")
    if t.strip()
}

if not _VALID_TOKENS:
    logger.warning("VALID_TOKENS 미설정 → 모든 요청이 거부됩니다.")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    Bearer 토큰을 검증하는 FastAPI 의존성 함수.

    Authorization: Bearer {token} 헤더를 파싱하고
    VALID_TOKENS에 포함 여부를 확인한다.

    Args:
        credentials: FastAPI HTTPBearer가 파싱한 자격증명

    Returns:
        검증된 토큰 문자열

    Raises:
        HTTPException 401: 토큰 없음 또는 유효하지 않음
    """
    token = credentials.credentials
    if token not in _VALID_TOKENS:
        logger.warning("유효하지 않은 토큰 시도: %s...", token[:8] if token else "")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# 선택적 인증 (토큰 없어도 통과, 있으면 검증)
def verify_token_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[str]:
    """
    선택적 Bearer 토큰 검증.
    토큰이 없으면 None 반환, 있으면 검증 후 반환.
    """
    if credentials is None:
        return None
    token = credentials.credentials
    if token not in _VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 인증 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
