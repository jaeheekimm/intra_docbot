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
    return f"""당신은 **제공된 컨텍스트를 기반으로만** 질문에 답변해야 합니다.
                절차는 단계로, 조건 혹은 예외가 있으면 함께 안내해야 합니다.

[질문]
{question}

[컨텍스트]
{context}
"""


def get_rag_chain(
    *,
    top_k: int = 5,
    dense_k: int = 20,
    bm25_k: int = 60,
    alpha: float = 0.6,
):
    """
    ✅ streaming 지원 + LangSmith 단계 구분(run_name) 버전

    - invoke(question:str) -> {"answer": str, "sources": [...], "hits": [...], ...}
    - stream(question:str) -> dict chunk들이 스트리밍으로 나옴
      (Streamlit에서는 chunk["answer"]를 누적/갱신해서 출력)
    """
    retriever = HybridRetriever()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)
    parser = StrOutputParser()

    def retrieve(question: str) -> Dict[str, Any]:
        hits = retriever.hybrid_search(
            question,
            top_k=top_k,
            dense_k=dense_k,
            bm25_k=bm25_k,
            alpha=alpha,
        )

        sources = [format_source(h.get("metadata", {}) or {}) for h in hits]

        # 중복 제거(순서 유지)
        uniq: List[str] = []
        for s in sources:
            if s not in uniq:
                uniq.append(s)

        context = _build_context(hits)
        return {
            "question": question,
            "context": context,
            "sources": uniq,
            "hits": hits,
        }

    # 01) Retrieve
    retrieve_r = RunnableLambda(retrieve).with_config(run_name="01.Retrieve(Hybrid)")

    # 02) Build Prompt
    prompt_r = RunnableLambda(
        lambda s: _make_prompt(s["question"], s["context"])
    ).with_config(run_name="02.BuildPrompt")

    # 03) LLM (Stream)
    llm_r = llm.with_config(run_name="03.LLM(Stream)")

    # 04) Parse Answer
    answer_r = (prompt_r | llm_r | parser).with_config(run_name="04.ParseAnswer")

    # Final Chain
    chain = (retrieve_r | RunnablePassthrough.assign(answer=answer_r)).with_config(
        run_name="RAG-Chain"
    )

    return chain
