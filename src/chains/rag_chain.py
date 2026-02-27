# src/chains/rag_chain.py

import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import HybridRetriever, format_source

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _build_context(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    hits: retriever.hybrid_search 결과 리스트
    """
    blocks = []
    total = 0
    for h in hits:
        md = h.get("metadata", {}) or {}
        src = format_source(md)
        text = (h.get("text") or "").strip()
        if not text:
            continue

        # (선택) title을 컨텍스트에 노출하고 싶으면 아래 2줄 활성화
        # title = (md.get("title") or "").strip()
        # header = f"[{src}]\n(제목) {title}" if title else f"[{src}]"

        header = f"[{src}]"
        block = f"{header}\n{text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


"""너는 사내 메뉴얼 문서 기반 AI 비서다.
제공된 문서 내용만 근거로 답하며, 문서에 없는 내용은 생성하지 않는다.

[규칙]
1) 문서에 없는 정보는 추측하지 않는다.
2) 문서에 근거가 없으면, 자연스럽게 안내하되 내부 구조(컨텍스트, 문서 등)는 언급하지 않는다.
3) 절차/방법/신청/설정/해결 요청은 단계적으로(1,2,3...) 작성한다.
4) 질문의 비공식 표현은 사용하지 말고, 문서의 공식 명칭으로 정규화하여 답한다.
5) 답변은 항목 또는 줄바꿈을 활용해 구조적으로 작성한다.
   - 나열 정보는 쉼표로 길게 쓰지 말고 줄 단위로 구분한다."""


def _make_prompt(question: str, context: str) -> str:
    return f"""당신은 사내 내부 문서 기반 AI 비서입니다.

[절대 규칙]
- [컨텍스트]에 질문에 대한 직접적인 답이 없으면, 알려줄 수 없음을 먼저 안내하십시오.
- 매번 다른 표현을 사용하되, "문서", "컨텍스트", "제공된 정보" 같은 내부 구조 표현은 절대 사용하지 마십시오.
- [컨텍스트]에 있더라도 질문과 직접 관련 없는 내용은 사용하지 마십시오.
- 사용자가 비공식 표현을 사용하더라도 답변에는 반드시 문서에 명시된 공식 명칭으로 바꿔서 작성하십시오.
- 질문에 대한 답을 먼저 하고, 추가 정보는 그 다음에 작성하십시오.

[답변 형식]
- 절차/방법/신청은 번호 단계(1, 2, 3...)로 작성하십시오.
- 나열 정보는 줄바꿈으로 구분하십시오.

[질문]
{question}

[컨텍스트]
{context}
"""


def get_rag_parts(*, top_k=5, dense_k=20, bm25_k=60, alpha=0.6):
    retriever = HybridRetriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)
    parser = StrOutputParser()

    def retrieve(question: str):
        hits = retriever.hybrid_search(
            question,
            top_k=top_k,
            dense_k=dense_k,
            bm25_k=bm25_k,
            alpha=alpha,
        )

        context = _build_context(hits)

        return {
            "question": question,
            "context": context,
            "hits": hits,
        }

    retrieve_r = RunnableLambda(retrieve)

    prompt_r = RunnableLambda(lambda s: _make_prompt(s["question"], s["context"]))

    answer_r = prompt_r | llm | parser

    return retrieve_r, answer_r


import numpy as np
from langchain_openai import OpenAIEmbeddings
import chromadb


def filter_sources_by_similarity(
    answer: str, hits: list, threshold: float = 0.35
) -> list:
    """
    답변 임베딩 1회 → Chroma에 저장된 chunk 벡터 직접 조회 → 코사인 유사도 계산
    threshold 이상인 chunk만 출처로 반환
    """
    if not hits or not answer.strip():
        return hits

    # 1) 답변 임베딩 (API 1회)
    embed_model = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    answer_vec = np.array(embed_model.embed_query(answer))

    # 2) Chroma에서 chunk_id로 저장된 벡터 직접 조회
    chroma_dir = os.getenv("CHROMA_DIR", "/tmp/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "intra_docs")
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_collection(collection_name)

    # hits에서 chunk_id 추출
    chunk_ids = []
    for h in hits:
        cid = (h.get("metadata") or {}).get("chunk_id")
        if cid:
            chunk_ids.append(cid)

    if not chunk_ids:
        # chunk_id 없으면 그냥 전체 반환
        return hits

    # Chroma에서 해당 chunk들의 벡터 조회
    result = col.get(ids=chunk_ids, include=["embeddings"])
    id_to_vec = {
        rid: np.array(vec) for rid, vec in zip(result["ids"], result["embeddings"])
    }

    # 3) 코사인 유사도 계산 & 필터링
    filtered = []
    all_scored = []  # ← 추가: 전체 hits에 score 붙이기
    for h in hits:
        cid = (h.get("metadata") or {}).get("chunk_id")
        chunk_vec = id_to_vec.get(cid)
        if chunk_vec is None:
            continue

        score = float(
            np.dot(answer_vec, chunk_vec)
            / (np.linalg.norm(answer_vec) * np.linalg.norm(chunk_vec) + 1e-9)
        )

        h = dict(h)  # 원본 수정 방지
        h["similarity_score"] = round(score, 3)  # ← 전부 score 붙임
        all_scored.append(h)  # ← 추가

        if score >= threshold:
            filtered.append(h)

    filtered.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    all_scored.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)  # ← 추가

    return (
        filtered,  # threshold 미달이면 그냥 빈 리스트
        all_scored,  # 원문탭은 score 붙은 전체
    )
