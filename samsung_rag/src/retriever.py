"""
src/retriever.py
─────────────────────────────────────────────────────────────────────────────
Hybrid 검색 모듈.

검색 순서:
    1. Dense  : pgvector 코사인 유사도 → top 25
    2. Sparse : tsvector 키워드 검색   → top 60
    3. RRF    : Python에서 Reciprocal Rank Fusion 합산 → top 50
    4. Rerank : bge-reranker-v2-m3     → top_k (기본 5)

Reranker 운영 전환 주석:
    CPU(로컬) 실행 → 아래 주석 해제 시 GPU 서버 API 호출로 전환

유사도 필터:
    - dense_threshold: RRF 전 dense 결과에 최소 코사인 유사도 적용
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://gpu-server:8001/rerank")

# ─── Reranker 싱글턴 ──────────────────────────────────────────────────────────
_reranker = None


def _get_reranker():
    """
    CrossEncoder 싱글턴 반환 (지연 초기화).

    # ── GPU 서버 API 전환 시 이 함수 사용 안 함 ─────────────────────────────
    # def rerank_via_api(query: str, passages: list[str]) -> list[float]:
    #     import requests
    #     resp = requests.post(
    #         RERANKER_API_URL,
    #         json={"query": query, "passages": passages},
    #         timeout=30,
    #     )
    #     resp.raise_for_status()
    #     return resp.json()["scores"]
    # ────────────────────────────────────────────────────────────────────────
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Reranker 모델 로드 중: %s", RERANKER_MODEL)
            _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
            logger.info("Reranker 로드 완료")
        except Exception as e:
            logger.error("Reranker 로드 실패: %s", e)
            raise
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# 임베딩 (질의용)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    """
    질의 텍스트를 임베딩 벡터로 변환한다.

    개발: OpenAI text-embedding-3-small
    운영 전환:
    # import requests
    # OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # resp = requests.post(
    #     f"{OLLAMA_URL}/api/embeddings",
    #     json={"model": os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"), "prompt": query},
    # )
    # return resp.json()["embedding"]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# Dense 검색 (pgvector)
# ─────────────────────────────────────────────────────────────────────────────

def _dense_search(
    conn,
    query_vec: list[float],
    top_k: int = 25,
    threshold: float = 0.3,
) -> list[dict]:
    """
    pgvector 코사인 유사도 검색.

    Args:
        conn:      psycopg2 연결
        query_vec: 질의 임베딩 벡터
        top_k:     반환할 최대 결과 수
        threshold: 최소 코사인 유사도 (1 - distance)

    Returns:
        청크 딕셔너리 리스트 (chunk_id, content, source, metadata, score 포함)
    """
    import psycopg2.extras

    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    sql = """
        SELECT
            c.chunk_id,
            c.content,
            c.embed_text,
            c.source,
            c.doc_type,
            c.page,
            c.slide,
            c.sheet,
            c.metadata,
            1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.is_latest = TRUE
          AND 1 - (c.embedding <=> %s::vector) >= %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (vec_str, vec_str, threshold, vec_str, top_k))
        rows = cur.fetchall()

    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Sparse 검색 (tsvector BM25)
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_search(
    conn,
    query: str,
    top_k: int = 60,
) -> list[dict]:
    """
    PostgreSQL tsvector 키워드 검색.

    Kiwi 형태소 분석으로 명사·동사만 추출 후 to_tsquery('simple', ...)로 검색.

    Args:
        conn:   psycopg2 연결
        query:  원본 질의 텍스트
        top_k:  반환할 최대 결과 수

    Returns:
        청크 딕셔너리 리스트 (ts_rank 포함)
    """
    import psycopg2.extras
    from src.utils.preprocess import preprocess_for_bm25

    processed = preprocess_for_bm25(query)
    if not processed.strip():
        logger.warning("Sparse 검색: 형태소 추출 결과 없음 → 원본 사용")
        processed = query

    # 토큰을 AND 조건으로 결합
    tokens = [t.strip() for t in processed.split() if t.strip()]
    if not tokens:
        return []

    tsquery = " & ".join(tokens)

    sql = """
        SELECT
            c.chunk_id,
            c.content,
            c.embed_text,
            c.source,
            c.doc_type,
            c.page,
            c.slide,
            c.sheet,
            c.metadata,
            ts_rank(c.ts_content, to_tsquery('simple', %s)) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.is_latest = TRUE
          AND c.ts_content @@ to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (tsquery, tsquery, top_k))
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Sparse 검색 실패 (tsquery=%s): %s", tsquery, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# RRF (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_n: int = 50,
) -> list[dict]:
    """
    Dense + Sparse 결과를 RRF로 통합한다.

    RRF 점수: sum(1 / (k + rank))  (rank는 1-based)

    Args:
        dense_results:  Dense 검색 결과 (이미 rank 순)
        sparse_results: Sparse 검색 결과 (이미 rank 순)
        k:              RRF 상수 (기본 60)
        top_n:          최종 반환 수

    Returns:
        RRF 점수로 정렬된 청크 딕셔너리 리스트
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = item

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    result = []
    for cid in sorted_ids:
        item = chunk_map[cid].copy()
        item["rrf_score"] = rrf_scores[cid]
        result.append(item)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    bge-reranker-v2-m3로 후보 청크를 재순위화한다.

    CPU 실행 (로컬 개발).
    GPU 서버 API 전환 시:
    # scores = rerank_via_api(query, [c["content"] for c in candidates])

    Args:
        query:      원본 질의 텍스트
        candidates: RRF 결과 청크 리스트
        top_k:      최종 반환 수

    Returns:
        재순위화된 청크 리스트 (rerank_score 포함)
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["content"]) for c in candidates]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error("Reranker 예측 실패: %s", e)
        return candidates[:top_k]

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 통합 검색 인터페이스
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 5,
    dense_top: int = 25,
    sparse_top: int = 60,
    rrf_top: int = 50,
    dense_threshold: float = 0.3,
) -> list[dict]:
    """
    Hybrid 검색 메인 함수.

    Args:
        query:           검색 질의
        top_k:           최종 반환 청크 수 (Reranker 후)
        dense_top:       Dense 검색 반환 수
        sparse_top:      Sparse 검색 반환 수
        rrf_top:         RRF 합산 후 유지 수
        dense_threshold: Dense 최소 유사도 필터

    Returns:
        최종 정렬된 청크 딕셔너리 리스트
    """
    from src.db import get_connection
    from src.utils.preprocess import hanja_to_hangul

    # 질의 전처리
    query_clean = hanja_to_hangul(query)

    logger.info("Hybrid 검색 시작: query='%s...'", query_clean[:30])

    # 임베딩
    query_vec = _embed_query(query_clean)

    conn = get_connection()
    try:
        # Dense
        dense_results = _dense_search(conn, query_vec, top_k=dense_top, threshold=dense_threshold)
        logger.debug("Dense 결과: %d건", len(dense_results))

        # Sparse
        sparse_results = _sparse_search(conn, query_clean, top_k=sparse_top)
        logger.debug("Sparse 결과: %d건", len(sparse_results))
    finally:
        conn.close()

    # RRF
    rrf_results = _rrf_merge(dense_results, sparse_results, top_n=rrf_top)
    logger.debug("RRF 후: %d건", len(rrf_results))

    # Rerank
    final = _rerank(query_clean, rrf_results, top_k=top_k)
    logger.info("Hybrid 검색 완료: 최종 %d건 반환", len(final))

    return final
