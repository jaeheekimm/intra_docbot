# src/chains/rag_chain.py

import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import HybridRetriever, format_source

load_dotenv()

from src.utils.paths import LLM_MODEL, EMBED_MODEL, CHROMA_DIR, CHROMA_COLLECTION


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
제공된 문서 내용만 근거로 답한다.

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
- "문서", "컨텍스트", "제공된 정보" 같은 내부 구조 표현은 절대 사용하지 마십시오.
- 질문에서 명시적으로 요청한 내용만 답변하십시오. 묻지 않은 절차·방법·주의사항은 포함하지 마십시오.
- 사용자가 비공식 표현을 사용하더라도 답변에는 반드시 문서에 명시된 공식 명칭으로 바꿔서 작성하십시오.

[답변 형식]
- 금액·날짜·대상 등 단순 사실 질문은 관련된 핵심 값을 모두 포함하여 간결하게 답하십시오.
- 절차·방법·신청 방법을 물었을 때만 번호 단계(1, 2, 3...)로 작성하십시오.
- 나열 정보는 줄바꿈으로 구분하십시오.

[질문]
{question}

[컨텍스트]
{context}
"""


REFUSAL_MSG = (
    "저는 사내 문서 기반 검색 봇입니다. "
    "복리후생·그룹웨어·시설 안내 등 사내 관련 내용을 질문해 주세요."
)

# 명백히 문서 검색과 무관한 요청 패턴 (이것만 차단, 나머지는 RAG로 전달)
_OFFTOPIC_PATTERNS = [
    r"(이메일|메일|보고서|기획서|제안서|자기소개서|커버레터).{0,10}(써|작성|만들|써줘|작성해|만들어)",
    r"(번역|translate)\s*(해줘|해|줘|해주|해주세요)",
    r"(코드|함수|프로그램|스크립트).{0,5}(짜|작성|만들|써)",
    r"(시|노래|소설|이야기|에세이).{0,5}(써|지어|만들|작성)",
]


def is_document_query(question: str) -> bool:
    """명백히 문서 외 요청(작성·번역·창작·코드)만 차단. 나머지는 RAG로 전달."""
    for pat in _OFFTOPIC_PATTERNS:
        if re.search(pat, question):
            return False
    return True


def rewrite_query(question: str, history_txt: str) -> str:
    """대화 맥락을 반영해 독립적인 검색 쿼리로 재작성."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""다음 대화 히스토리를 참고하여, 현재 질문을 문서 검색에 적합한 독립적인 한 문장으로 재작성하세요.
재작성된 질문만 출력하세요. 설명 없이.

[대화 히스토리]
{history_txt}

[현재 질문]
{question}

[재작성된 질문]"""
    return llm.invoke(prompt).content.strip()


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
        print(f"[DEBUG] top_k={top_k}, hits 개수={len(hits)}")

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
) -> tuple:
    if not hits or not answer.strip():
        return [], hits

    embed_model = OpenAIEmbeddings(model=EMBED_MODEL)
    answer_vec = np.array(embed_model.embed_query(answer))

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(CHROMA_COLLECTION)

    chunk_ids = [
        (h.get("metadata") or {}).get("chunk_id")
        for h in hits
        if (h.get("metadata") or {}).get("chunk_id")
    ]

    result = col.get(ids=chunk_ids, include=["embeddings"])
    id_to_vec = {
        rid: np.array(vec) for rid, vec in zip(result["ids"], result["embeddings"])
    }

    filtered = []
    all_scored = []

    for h in hits:
        cid = (h.get("metadata") or {}).get("chunk_id")
        chunk_vec = id_to_vec.get(cid)

        # Chroma에서 못 찾으면 텍스트로 직접 임베딩 계산 (폴백)
        if chunk_vec is None:
            chunk_text = (h.get("text") or "").strip()
            if not chunk_text:
                continue
            chunk_vec = np.array(embed_model.embed_query(chunk_text))

        score = float(
            np.dot(answer_vec, chunk_vec)
            / (np.linalg.norm(answer_vec) * np.linalg.norm(chunk_vec) + 1e-9)
        )

        h = dict(h)
        h["similarity_score"] = round(score, 3)
        all_scored.append(h)

    all_scored.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    if all_scored:
        filtered = [h for h in all_scored if h["similarity_score"] >= threshold]
    else:
        filtered = []

    return filtered, all_scored
