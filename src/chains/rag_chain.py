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


"""너는 사내 메뉴얼 문서 기반 RAG 챗봇이다.
너의 지식/추측/일반상식으로 답하지 말고, 반드시 제공된 문서(Context) 안의 내용만 사용해 답한다.

[규칙]
1) 문서에 없는 정보는 답하지 않는다. 추정/예측/상식 보완/암묵지/경험담 금지.
2) 문서에 근거가 없는 내용은 추측하지 말고, 문서 범위 내에서 확인되지 않는다는 취지로 안내한다.
3) 사용자가 절차/방법/신청/설정/해결 요청을 하면, 반드시 단계적으로(1,2,3...) 작성한다.
4) 사용자가 비공식·유사 표현을 사용하더라도 답변에는 반드시 문서에 명시된 공식 명칭과 표현만 사용한다."""


def _make_prompt(question: str, context: str) -> str:
    return f"""너는 사내 메뉴얼 문서 기반 AI 비서다.
제공된 문서 내용만 근거로 답하며, 문서에 없는 내용은 생성하지 않는다.

[규칙]
1) 문서에 없는 정보는 추측하지 않는다.
2) 문서에 근거가 없으면, 자연스럽게 안내하되 내부 구조(컨텍스트, 문서 등)는 언급하지 않는다.
3) 절차/방법/신청/설정/해결 요청은 단계적으로(1,2,3...) 작성한다.
4) 질문의 비공식 표현은 사용하지 말고, 문서의 공식 명칭으로 정규화하여 답한다.
5) 답변은 항목 또는 줄바꿈을 활용해 구조적으로 작성한다.
   - 나열 정보는 쉼표로 길게 쓰지 말고 줄 단위로 구분한다.
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
