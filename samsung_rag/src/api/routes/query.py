"""
src/api/routes/query.py
─────────────────────────────────────────────────────────────────────────────
질의응답 API 엔드포인트.

    POST /api/query/stream  → SSE 스트리밍
    POST /api/query         → 일반 JSON
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.api.auth import verify_token
from src.api.schemas import QueryRequest, QueryResponse, SourceItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])


# ─────────────────────────────────────────────────────────────────────────────
# SSE 스트리밍
# ─────────────────────────────────────────────────────────────────────────────

async def _sse_generator(request: QueryRequest) -> AsyncGenerator[dict, None]:
    """
    SSE 이벤트 제너레이터.

    이벤트 형식:
        - event: "token",  data: 텍스트 조각
        - event: "source", data: JSON (출처 정보)
        - event: "done",   data: "[DONE]"
        - event: "error",  data: 오류 메시지

    Args:
        request: QueryRequest

    Yields:
        SSE 이벤트 딕셔너리
    """
    import asyncio
    from src.rag_chain import run_rag
    from src.retriever import hybrid_search

    history = [m.model_dump() for m in request.history]

    try:
        # 검색 먼저 수행 (동기 → asyncio.to_thread로 비블로킹)
        chunks = await asyncio.to_thread(
            hybrid_search,
            request.question,
            request.top_k,
            25,
            60,
            50,
            request.dense_threshold,
        )

        # 출처 이벤트 먼저 전송
        seen: set[str] = set()
        for c in chunks:
            key = f"{c.get('source', '')}_{c.get('page', '')}_{c.get('slide', '')}_{c.get('sheet', '')}"
            if key not in seen:
                seen.add(key)
                source_item = SourceItem(
                    source=c.get("source", ""),
                    doc_type=c.get("doc_type", ""),
                    page=c.get("page"),
                    slide=c.get("slide"),
                    sheet=c.get("sheet"),
                    content_preview=c.get("content", "")[:200],
                    rerank_score=c.get("rerank_score"),
                )
                yield {
                    "event": "source",
                    "data": source_item.model_dump_json(),
                }

        # LLM 스트리밍 (동기 제너레이터를 thread pool에서 순회)
        from src.rag_chain import generate_stream

        stream_gen = generate_stream(
            question=request.question,
            context_chunks=chunks,
            history=history,
            temperature=request.temperature,
        )

        # 동기 제너레이터 → async 변환
        loop = asyncio.get_event_loop()
        for token in stream_gen:
            yield {"event": "token", "data": token}
            # 양보 (이벤트 루프 차단 방지)
            await asyncio.sleep(0)

        yield {"event": "done", "data": "[DONE]"}

    except Exception as e:
        logger.error("SSE 스트리밍 오류: %s", e)
        yield {"event": "error", "data": str(e)}


@router.post(
    "/stream",
    summary="질의응답 (SSE 스트리밍)",
    description="질문을 전송하면 Server-Sent Events로 답변을 실시간 스트리밍합니다.",
    response_class=EventSourceResponse,
)
async def query_stream(
    request: QueryRequest,
    _token: str = Depends(verify_token),
) -> EventSourceResponse:
    """
    SSE 스트리밍 질의응답 엔드포인트.

    이벤트 타입:
        - source: 출처 정보 (JSON)
        - token:  LLM 응답 텍스트 조각
        - done:   스트리밍 완료
        - error:  오류 발생
    """
    return EventSourceResponse(_sse_generator(request))


# ─────────────────────────────────────────────────────────────────────────────
# 일반 JSON
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="질의응답 (JSON)",
    description="질문을 전송하면 완성된 답변과 출처를 JSON으로 반환합니다.",
    response_model=QueryResponse,
)
async def query_json(
    request: QueryRequest,
    _token: str = Depends(verify_token),
) -> QueryResponse:
    """
    일반 JSON 질의응답 엔드포인트.

    Returns:
        QueryResponse: 답변 + 출처 목록
    """
    import asyncio
    from src.rag_chain import run_rag

    history = [m.model_dump() for m in request.history]

    result = await asyncio.to_thread(
        run_rag,
        request.question,
        history,
        request.top_k,
        request.temperature,
        False,
        request.dense_threshold,
    )

    sources = [
        SourceItem(
            source=s["source"],
            doc_type=s.get("doc_type", ""),
            page=s.get("page"),
            slide=s.get("slide"),
            sheet=s.get("sheet"),
            content_preview=s.get("content_preview", ""),
            rerank_score=s.get("rerank_score"),
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(answer=result["answer"], sources=sources)
