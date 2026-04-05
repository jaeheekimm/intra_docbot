"""
src/api/routes/admin.py
─────────────────────────────────────────────────────────────────────────────
관리자 API 엔드포인트.

    POST /api/admin/sync         → 수동 문서 동기화 트리거
    GET  /api/admin/sync/status  → 동기화 상태 조회
    GET  /api/admin/documents    → 등록된 문서 목록
    DELETE /api/admin/documents/{doc_id} → 문서 삭제
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_token
from src.api.schemas import (
    SyncRequest,
    SyncResponse,
    SyncResultItem,
    SyncStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ─────────────────────────────────────────────────────────────────────────────
# 동기화
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/sync",
    summary="문서 동기화 (수동 트리거)",
    response_model=SyncResponse,
)
async def manual_sync(
    request: SyncRequest,
    _token: str = Depends(verify_token),
) -> SyncResponse:
    """
    지정 디렉터리의 문서를 즉시 동기화한다.

    이미 동기화 중이면 409 반환.

    Args:
        request: SyncRequest (directory 옵션)

    Returns:
        SyncResponse: 처리 결과 요약
    """
    from src.pipeline.sync import run_sync, get_sync_status

    status_info = get_sync_status()
    if status_info["is_running"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="동기화가 이미 실행 중입니다.",
        )

    directory: Optional[Path] = Path(request.directory) if request.directory else None

    try:
        results = await asyncio.to_thread(run_sync, directory)
    except Exception as e:
        logger.error("수동 동기화 실패: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"동기화 실패: {e}",
        )

    items = [
        SyncResultItem(
            source=r.get("source", ""),
            status=r.get("status", "error"),
            chunks=r.get("chunks", 0),
            document_id=r.get("document_id"),
            error=r.get("error"),
        )
        for r in results
    ]

    return SyncResponse(
        total=len(items),
        indexed=sum(1 for i in items if i.status == "indexed"),
        skipped=sum(1 for i in items if i.status == "skipped"),
        errors=sum(1 for i in items if i.status == "error"),
        results=items,
    )


@router.get(
    "/sync/status",
    summary="동기화 상태 조회",
    response_model=SyncStatusResponse,
)
async def sync_status(
    _token: str = Depends(verify_token),
) -> SyncStatusResponse:
    """마지막 동기화 상태를 반환한다."""
    from src.pipeline.sync import get_sync_status

    info = get_sync_status()
    return SyncStatusResponse(
        last_sync_at=info.get("last_sync_at"),
        is_running=info.get("is_running", False),
        last_sync_result=info.get("last_sync_result", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 문서 관리
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    summary="등록된 문서 목록 조회",
    response_model=list[dict[str, Any]],
)
async def list_documents(
    is_latest: bool = True,
    limit: int = 100,
    offset: int = 0,
    _token: str = Depends(verify_token),
) -> list[dict[str, Any]]:
    """
    documents 테이블 조회.

    Args:
        is_latest: True → 최신 버전만 조회
        limit:     최대 반환 수 (기본 100)
        offset:    오프셋

    Returns:
        문서 딕셔너리 리스트
    """
    import psycopg2.extras
    from src.db import get_connection

    def _fetch():
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, source, doc_type, category, fingerprint,
                           pipeline_version, is_latest, created_at, updated_at
                    FROM documents
                    WHERE is_latest = %s
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (is_latest, limit, offset),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    try:
        docs = await asyncio.to_thread(_fetch)
        # datetime → string 변환
        for doc in docs:
            for key in ("created_at", "updated_at"):
                if doc.get(key):
                    doc[key] = doc[key].isoformat()
        return docs
    except Exception as e:
        logger.error("문서 목록 조회 실패: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete(
    "/documents/{doc_id}",
    summary="문서 삭제",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    doc_id: int,
    _token: str = Depends(verify_token),
) -> None:
    """
    문서 및 연관 청크·이미지를 삭제한다.
    ON DELETE CASCADE로 chunks, document_images 자동 삭제.

    Args:
        doc_id: 삭제할 문서 ID
    """
    from src.db import get_connection

    def _delete():
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,))
                deleted = cur.fetchone()
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    try:
        deleted = await asyncio.to_thread(_delete)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"문서 ID {doc_id}를 찾을 수 없습니다.",
            )
        logger.info("문서 삭제 완료: id=%d", doc_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("문서 삭제 실패: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
