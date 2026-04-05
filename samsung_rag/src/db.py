"""
src/db.py
─────────────────────────────────────────────────────────────────────────────
PostgreSQL 연결 관리 및 DDL 초기화.
pgvector 익스텐션과 필수 테이블을 자동 생성한다.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── 연결 설정 ────────────────────────────────────────────────────────────────
DB_CONFIG: dict[str, str | int] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "ragdb"),
    "user": os.getenv("DB_USER", "jaeheekim"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ─── DDL ──────────────────────────────────────────────────────────────────────
_DDL = """
-- pgvector 익스텐션
CREATE EXTENSION IF NOT EXISTS vector;

-- 문서 메타데이터
CREATE TABLE IF NOT EXISTS documents (
    id               SERIAL PRIMARY KEY,
    source           VARCHAR(512) NOT NULL,           -- 파일 경로 or ECM URL
    doc_type         VARCHAR(50),                     -- pdf, pptx, xlsx, docx, txt
    category         VARCHAR(200),                    -- 카테고리명
    fingerprint      VARCHAR(40) UNIQUE,              -- PIPELINE_VERSION + SHA1
    pipeline_version VARCHAR(20),
    is_latest        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- 청크 + 임베딩
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    VARCHAR(40) UNIQUE NOT NULL,           -- SHA1 해시
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT,                                  -- 원문 청크
    embed_text  TEXT,                                  -- prefix 포함 임베딩용 텍스트
    embedding   vector(1536),                          -- text-embedding-3-small
    -- 운영(Ollama bge-m3) 전환 시: vector(1024) 로 변경 + 마이그레이션 필요
    metadata    JSONB DEFAULT '{}',
    doc_type    VARCHAR(50),
    page        INTEGER,
    slide       INTEGER,
    sheet       VARCHAR(200),
    source      VARCHAR(512),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- chunks tsvector 검색용 컬럼
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS ts_content tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_ts_content
    ON chunks USING GIN (ts_content);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_documents_fingerprint
    ON documents (fingerprint);
CREATE INDEX IF NOT EXISTS idx_documents_is_latest
    ON documents (is_latest);

-- 이미지 메타데이터 (멀티모달 확장 대비)
CREATE TABLE IF NOT EXISTS document_images (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    image_path  VARCHAR(512),
    image_hash  VARCHAR(64),
    is_logo     BOOLEAN DEFAULT FALSE,     -- 동일 해시 3회 이상 시 TRUE
    page        INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_document_images_hash
    ON document_images (image_hash);
"""


def get_connection() -> psycopg2.extensions.connection:
    """새 psycopg2 연결을 반환한다."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        logger.error("DB 연결 실패: %s", e)
        raise


@contextmanager
def get_cursor(
    conn: psycopg2.extensions.connection | None = None,
    cursor_factory=psycopg2.extras.RealDictCursor,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    컨텍스트 매니저 커서.

    외부 conn을 전달하면 해당 커넥션 재사용,
    전달하지 않으면 새 커넥션을 열고 블록 종료 시 커밋/롤백 후 닫는다.
    """
    own_conn = conn is None
    _conn = get_connection() if own_conn else conn
    try:
        with _conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur
        if own_conn:
            _conn.commit()
    except Exception:
        if own_conn:
            _conn.rollback()
        raise
    finally:
        if own_conn:
            _conn.close()


def init_db() -> None:
    """
    pgvector 익스텐션과 필수 테이블을 생성한다.
    애플리케이션 시작 시 1회 호출.
    """
    logger.info("DB 스키마 초기화 시작")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        logger.info("DB 스키마 초기화 완료")
    except Exception as e:
        conn.rollback()
        logger.error("DB 스키마 초기화 실패: %s", e)
        raise
    finally:
        conn.close()


def health_check() -> dict[str, str]:
    """DB 연결 상태를 반환한다."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
        return {"status": "ok", "version": str(row["version"] if row else "unknown")}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
