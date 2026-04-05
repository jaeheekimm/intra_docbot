"""
src/rag_chain.py
─────────────────────────────────────────────────────────────────────────────
RAG 파이프라인 모듈.

프롬프트 전략:
    - 현재 질문을 맨 앞 배치
    - 멀티턴: [현재질문] + [참고-이전대화] + [추가지침]
    - 시스템 프롬프트: 규정·제도 범위 외 질문 차단
    - 단일값 질문 → 간단 답변, 여러 항목 비교 → 마크다운 표

LLM:
    - 개발: OpenAI gpt-4o-mini
    - 운영 전환 시 (Ollama) 주석 해제

스트리밍:
    - generate_stream(): SSE 스트리밍용 제너레이터
    - generate(): 일반 JSON 응답
"""

from __future__ import annotations

import logging
import os
from typing import Generator, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── 시스템 프롬프트 ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """당신은 삼성 사내 규정·제도 전문 AI 어시스턴트입니다.
아래 지침을 반드시 준수하세요.

[역할]
- 사내 규정, 복리후생, 출장 규정, 영업 제도 등 업무 관련 문서만 답변합니다.
- 관련 문서가 없거나 질문이 규정·제도 범위 밖이면 "해당 내용은 제공된 문서에서 찾을 수 없습니다"라고 답변합니다.

[답변 형식]
- 단일 항목(금액, 날짜, 조건 등) 질문: 핵심만 간결하게 답변
- 여러 항목 비교·열거: 마크다운 표 형식 사용
- 조항 인용 시: "제X조(조항명)" 형식으로 명시
- 답변 마지막에 참고 문서 출처를 간략히 표시

[금지 사항]
- 개인 신상, 정치, 종교, 투자 조언 등 업무 무관 질문에는 답변하지 않습니다.
- 제공된 문서 외 내용을 추측하거나 임의로 생성하지 않습니다.
"""

_CONTEXT_TEMPLATE = """[참고 문서]
{context}

---
"""

_HISTORY_TEMPLATE = """[이전 대화 참고]
{history}

---
"""

_QUESTION_TEMPLATE = """[현재 질문]
{question}

위 참고 문서를 바탕으로 현재 질문에 답변해주세요."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM 클라이언트
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(
    question: str,
    context_chunks: list[dict],
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """
    LLM에 전달할 메시지 목록을 구성한다.

    순서: system → context → history(있을 때) → 현재 질문

    Args:
        question:       현재 질문
        context_chunks: 검색된 청크 리스트
        history:        이전 대화 [{"role": "user"|"assistant", "content": str}]

    Returns:
        OpenAI messages 형식 리스트
    """
    # 컨텍스트 구성
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "")
        filename = source.split("/")[-1].split("\\")[-1]
        page_info = ""
        if chunk.get("page"):
            page_info = f" (p.{chunk['page']})"
        elif chunk.get("slide"):
            page_info = f" (슬라이드 {chunk['slide']})"
        elif chunk.get("sheet"):
            page_info = f" (시트: {chunk['sheet']})"
        context_parts.append(
            f"[출처 {i}: {filename}{page_info}]\n{chunk.get('content', '')}"
        )

    context_str = "\n\n".join(context_parts)
    context_block = _CONTEXT_TEMPLATE.format(context=context_str)

    # 이전 대화 구성
    history_block = ""
    if history:
        history_lines: list[str] = []
        for msg in history[-6:]:  # 최근 6턴만 포함 (토큰 절약)
            role = "사용자" if msg.get("role") == "user" else "AI"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_block = _HISTORY_TEMPLATE.format(history="\n".join(history_lines))

    # 최종 유저 메시지
    user_content = context_block
    if history_block:
        user_content += history_block
    user_content += _QUESTION_TEMPLATE.format(question=question)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_stream(
    question: str,
    context_chunks: list[dict],
    history: Optional[list[dict]] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    LLM 응답을 SSE 스트리밍으로 생성한다.

    개발: OpenAI gpt-4o-mini
    운영 전환 시 (Ollama):
    # ── Ollama 스트리밍 전환 코드 ────────────────────────────────────────────
    # import requests, json
    # OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:32b")
    # messages = _build_messages(question, context_chunks, history)
    # payload = {"model": MODEL, "messages": messages, "stream": True,
    #            "options": {"temperature": temperature, "num_predict": max_tokens}}
    # with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
    #     for line in r.iter_lines():
    #         if line:
    #             data = json.loads(line)
    #             delta = data.get("message", {}).get("content", "")
    #             if delta:
    #                 yield delta
    # ────────────────────────────────────────────────────────────────────────

    Args:
        question:       현재 질문
        context_chunks: 검색된 청크 리스트
        history:        이전 대화
        temperature:    생성 온도
        max_tokens:     최대 토큰 수

    Yields:
        응답 텍스트 조각 (str)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = _build_messages(question, context_chunks, history)

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.error("LLM 스트리밍 실패: %s", e)
        yield f"\n\n[오류] LLM 응답 생성 실패: {e}"


def generate(
    question: str,
    context_chunks: list[dict],
    history: Optional[list[dict]] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    LLM 응답을 일반 JSON(비스트리밍)으로 생성한다.

    Args:
        question:       현재 질문
        context_chunks: 검색된 청크 리스트
        history:        이전 대화
        temperature:    생성 온도
        max_tokens:     최대 토큰 수

    Returns:
        완성된 응답 텍스트
    """
    # ── Ollama 비스트리밍 전환 코드 ─────────────────────────────────────────
    # import requests
    # OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:32b")
    # messages = _build_messages(question, context_chunks, history)
    # payload = {"model": MODEL, "messages": messages, "stream": False,
    #            "options": {"temperature": temperature, "num_predict": max_tokens}}
    # resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    # resp.raise_for_status()
    # return resp.json()["message"]["content"]
    # ────────────────────────────────────────────────────────────────────────

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = _build_messages(question, context_chunks, history)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM 응답 생성 실패: %s", e)
        raise


def run_rag(
    question: str,
    history: Optional[list[dict]] = None,
    top_k: int = 5,
    temperature: float = 0.3,
    stream: bool = False,
    dense_threshold: float = 0.3,
) -> dict | Generator[str, None, None]:
    """
    RAG 파이프라인 통합 실행 함수.

    Args:
        question:        사용자 질문
        history:         이전 대화 리스트
        top_k:           검색 청크 수
        temperature:     LLM 온도
        stream:          True → 스트리밍 제너레이터 반환
        dense_threshold: Dense 검색 최소 유사도

    Returns:
        stream=False: {"answer": str, "sources": list[dict]}
        stream=True:  응답 텍스트 제너레이터 (sources는 별도 반환 불가)
    """
    from src.retriever import hybrid_search

    # 검색
    chunks = hybrid_search(
        query=question,
        top_k=top_k,
        dense_threshold=dense_threshold,
    )

    if stream:
        return generate_stream(question, chunks, history, temperature)

    answer = generate(question, chunks, history, temperature)

    # 출처 정보 구성
    sources: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        key = f"{c.get('source', '')}_{c.get('page', '')}_{c.get('slide', '')}_{c.get('sheet', '')}"
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": c.get("source", ""),
                    "doc_type": c.get("doc_type", ""),
                    "page": c.get("page"),
                    "slide": c.get("slide"),
                    "sheet": c.get("sheet"),
                    "content_preview": c.get("content", "")[:200],
                    "rerank_score": c.get("rerank_score"),
                }
            )

    return {"answer": answer, "sources": sources, "chunks": chunks}
