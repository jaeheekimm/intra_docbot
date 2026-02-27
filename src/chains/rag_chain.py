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


"""너는 사내 문서 기반 Q&A 도우미야. 답변은 한국어로만 해.
아래 컨텍스트에 근거해서만 답하고 너가 확인할 수 없는 내용은 안내하지마.
가능하면 절차는 단계로, 조건/예외가 있으면 함께 알려줘."""


def _make_prompt(question: str, context: str) -> str:
    return f"""너는 사내 메뉴얼 문서 기반 RAG 챗봇이다.
너의 지식/추측/일반상식으로 답하지 말고, 반드시 제공된 문서(Context) 안의 내용만 사용해 답한다.

[규칙]
1) 문서에 없는 정보는 답하지 않는다. 추정/예측/상식 보완/암묵지/경험담 금지.
2) 문서에 근거가 없는 내용은 추측하지 말고, 문서 범위 내에서 확인되지 않는다는 취지로 안내한다.
3) 사용자가 절차/방법/신청/설정/해결 요청을 하면, 반드시 단계적으로(1,2,3...) 작성한다.
4) 사용자의 질문 표현은 그대로 반복하지 말고, 답변에는 반드시 문서에 명시된 공식 명칭으로 정규화하여 사용하라.

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
