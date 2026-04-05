"""
src/api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI 앱 진입점.

엔드포인트:
    POST /api/query/stream  → SSE 스트리밍 질의응답
    POST /api/query         → 일반 JSON 질의응답
    POST /api/admin/sync    → 수동 동기화
    GET  /api/admin/sync/status → 동기화 상태
    GET  /api/admin/documents   → 문서 목록
    DELETE /api/admin/documents/{id} → 문서 삭제
    GET  /health            → 헬스체크
    GET  /docs              → Swagger UI (자동 생성)

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ─── 로깅 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (시작/종료 훅) ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    앱 시작 시:
        1. DB 스키마 초기화
        2. 동기화 폴링 스레드 시작

    앱 종료 시:
        1. 폴링 스레드 종료 신호
    """
    logger.info("=== Samsung RAG API 시작 ===")

    # DB 초기화
    try:
        from src.db import init_db
        init_db()
    except Exception as e:
        logger.error("DB 초기화 실패: %s", e)
        raise

    # 동기화 폴링 스레드 (dev 환경에서는 자동 시작)
    env = os.getenv("ENV", "dev")
    sync_thread = None
    if env != "test":
        from src.pipeline.sync import start_sync_thread
        sync_thread = start_sync_thread()
        logger.info("동기화 폴링 스레드 시작됨")

    yield  # 앱 실행

    # 종료
    if sync_thread:
        from src.pipeline.sync import stop_sync_thread
        stop_sync_thread()
        sync_thread.join(timeout=5)
        logger.info("동기화 스레드 종료됨")

    logger.info("=== Samsung RAG API 종료 ===")


# ─── FastAPI 앱 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Samsung RAG API",
    description="""
사내 규정·제도 문서 기반 Hybrid RAG API.

## 인증
모든 엔드포인트는 **Bearer Token** 인증 필요:
```
Authorization: Bearer {token}
```

## 주요 기능
- **Hybrid 검색**: Dense(pgvector) + Sparse(BM25) + RRF + Reranker
- **SSE 스트리밍**: 실시간 답변 스트리밍
- **멀티턴 대화**: 이전 대화 컨텍스트 유지
- **문서 관리**: 수동/자동 동기화, 버전 관리
""",
    version=os.getenv("PIPELINE_VERSION", "v1"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 라우터 등록 ──────────────────────────────────────────────────────────────
from src.api.routes.query import router as query_router
from src.api.routes.admin import router as admin_router

app.include_router(query_router)
app.include_router(admin_router)


# ─── 헬스체크 ──────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Health"],
    summary="서버 상태 확인",
)
async def health_check() -> dict:
    """
    서버 및 DB 연결 상태를 반환한다.

    Returns:
        {"status": "ok", "db": {...}, "version": "v1"}
    """
    from src.db import health_check as db_health

    db_status = db_health()
    overall = "ok" if db_status.get("status") == "ok" else "degraded"

    return {
        "status": overall,
        "db": db_status,
        "version": os.getenv("PIPELINE_VERSION", "v1"),
        "env": os.getenv("ENV", "dev"),
    }
