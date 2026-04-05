"""
src/api/schemas.py
─────────────────────────────────────────────────────────────────────────────
FastAPI 요청/응답 Pydantic 스키마.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 질의응답
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """단일 대화 메시지."""

    role: str = Field(..., description="'user' 또는 'assistant'", examples=["user"])
    content: str = Field(..., description="메시지 본문")


class QueryRequest(BaseModel):
    """POST /api/query 및 POST /api/query/stream 요청 바디."""

    question: str = Field(..., min_length=1, max_length=2000, description="사용자 질문")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="이전 대화 (멀티턴)",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="검색 청크 수")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="LLM 생성 온도")
    dense_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Dense 검색 최소 유사도"
    )


class SourceItem(BaseModel):
    """출처 정보 단일 항목."""

    source: str = Field(..., description="파일 경로 or 소스 식별자")
    doc_type: str = Field(default="", description="문서 유형 (pdf, pptx 등)")
    page: Optional[int] = Field(default=None, description="페이지 번호")
    slide: Optional[int] = Field(default=None, description="슬라이드 번호")
    sheet: Optional[str] = Field(default=None, description="시트명")
    content_preview: str = Field(default="", description="내용 미리보기 (200자)")
    rerank_score: Optional[float] = Field(default=None, description="Reranker 점수")


class QueryResponse(BaseModel):
    """POST /api/query 응답 바디."""

    answer: str = Field(..., description="AI 답변")
    sources: list[SourceItem] = Field(default_factory=list, description="출처 목록")


# ─────────────────────────────────────────────────────────────────────────────
# 관리자
# ─────────────────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    """POST /api/admin/sync 요청 바디."""

    directory: Optional[str] = Field(
        default=None, description="동기화할 디렉터리 경로 (기본: SAMPLE_DIR)"
    )


class SyncResultItem(BaseModel):
    """파일 1개 동기화 결과."""

    source: str
    status: str  # indexed | skipped | error | empty
    chunks: int = 0
    document_id: Optional[int] = None
    error: Optional[str] = None


class SyncResponse(BaseModel):
    """POST /api/admin/sync 응답 바디."""

    total: int = Field(..., description="처리 파일 수")
    indexed: int
    skipped: int
    errors: int
    results: list[SyncResultItem]


class SyncStatusResponse(BaseModel):
    """GET /api/admin/sync/status 응답."""

    last_sync_at: Optional[str]
    is_running: bool
    last_sync_result: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# 헬스체크
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health 응답."""

    status: str
    db: dict[str, str]
    version: str = Field(default="v1")
