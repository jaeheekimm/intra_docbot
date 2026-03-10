import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import HybridRetriever, format_source

load_dotenv()

from src.utils.paths import LLM_MODEL


def _build_context(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """hybrid_search 결과를 "[출처]\n텍스트" 형태로 이어붙인 컨텍스트 문자열 생성

    max_chars 초과하면 그 시점에서 중단
    """
    blocks = []
    total = 0
    for h in hits:
        md = h.get("metadata", {}) or {}
        src = format_source(md)
        text = (h.get("text") or "").strip()
        if not text:
            continue

        header = f"[{src}]"
        block = f"{header}\n{text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


def _make_prompt(question: str, context: str) -> str:
    return f"""당신은 사내 내부 문서 기반 AI 비서입니다.

[절대 규칙]
- [컨텍스트]에 질문에 대한 직접적인 답이 없으면, 관련처럼 보이는 다른 정보로 대체하지 말고 알려줄 수 없음을 안내하십시오.
- "문서", "컨텍스트", "제공된 정보" 같은 내부 구조 표현은 절대 사용하지 마십시오.
- 질문에서 요청하지 않은 주의사항·부가 정보는 포함하지 마십시오. 단, 답변의 자연스러운 도입·마무리 문장은 포함하십시오.
- 사용자가 비공식 표현을 사용하더라도 답변에는 반드시 [컨텍스트]에 명시된 공식 명칭으로 바꿔서 작성하십시오.
- 사용자 질문이 특정 답을 암시하거나 포함하더라도, [컨텍스트]에 해당 사실이 명확히 적혀 있지 않으면 절대 확인하거나 동의하지 마십시오. 추측하거나 동조하지 마십시오.

[답변 형식]
- 수치나 조건이 여러 개인 경우(월/연/대상/기간 등) 명시된 값을 빠짐없이 포함하십시오.
- 절차·방법·신청을 물었을 때만 번호 단계(1, 2, 3...)로 작성하십시오.
- 여러 항목을 나열할 때는 각 항목을 줄바꿈으로 구분하고,
  항목명이 있으면 "항목명: 값" 형식으로 작성하십시오.


[질문]
{question}

[컨텍스트]
{context}
"""


REFUSAL_MSG = (
    "저는 사내 문서 기반 검색 봇입니다. "
    "복리후생·그룹웨어·시설 안내 등 사내 관련 내용을 질문해 주세요."
)

# 명백히 RAG와 무관한 요청만 차단 (작성/번역/창작/코드). 나머지는 RAG로 전달
_OFFTOPIC_PATTERNS = [
    r"(이메일|메일|보고서|기획서|제안서|자기소개서|커버레터).{0,10}(써|작성|만들|써줘|작성해|만들어)",
    r"(번역|translate)\s*(해줘|해|줘|해주|해주세요)",
    r"(코드|함수|프로그램|스크립트).{0,5}(짜|작성|만들|써)",
    r"(시|노래|소설|이야기|에세이).{0,5}(써|지어|만들|작성)",
]


def is_document_query(question: str) -> bool:
    """명백히 문서 외 요청(작성·번역·창작·코드)이면 False, 나머지는 True"""
    for pat in _OFFTOPIC_PATTERNS:
        if re.search(pat, question):
            return False
    return True


def get_rag_parts(*, top_k=5, dense_k=20, bm25_k=60, alpha=0.6):
    """retrieve_r (검색 runnable)과 answer_r (LLM 답변 runnable)을 분리해서 반환

    스트리밍을 위해 두 단계를 분리. app.py에서 retrieve_r.invoke() → answer_r.stream() 순으로 호출
    """
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


def rewrite_query(user_input: str, history_txt: str) -> str:
    """이전 대화 맥락을 바탕으로 질문을 독립적인 검색 쿼리로 재작성"""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = f"""다음은 사내 문서 검색 챗봇과의 대화 이력과 현재 질문입니다.
현재 질문이 이전 대화 맥락에 의존하거나 모호한 경우, 독립적으로 이해 가능한 검색 쿼리로 재작성하세요.
명확한 질문이라면 그대로 반환하세요. 재작성된 쿼리 텍스트만 출력하세요.

[이전 대화]
{history_txt}

[현재 질문]
{user_input}

[재작성된 검색 쿼리]"""
    return llm.invoke(prompt).content.strip()


def filter_sources_by_similarity(
    _answer: str, hits: list, threshold: float = 0.4
) -> tuple:
    """출처 표시에 쓸 hits를 필터링해서 반환. 현재는 dense score 기준 1등만 표시.

    출처 필터링 방식 변천사:
    1. LLM한테 출처 추출 시켜보기 → 미시도
    2. reranker 모델로 재순위 → 써봤는데 성능 더 떨어짐
    3. 답변 임베딩과 청크 임베딩 코사인 유사도로 필터링 → 관련 없는 출처가 뜨는 경우 있었음
    → 현재: 그냥 dense score 1등 1개만 출처로 표시. threshold 파라미터는 현재 미사용.

    TODO: LLM한테 출처 뽑게 하는 것도 고민해볼 것. reranker는 더 좋은 모델로 재시도 여지 있음.
    반환: (filtered_hits, all_scored_hits)
    """
    all_scored = []
    for h in hits:
        h = dict(h)
        h["similarity_score"] = round(h.get("dense", 0), 3)
        all_scored.append(h)

    all_scored.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    filtered = all_scored[:1] if all_scored else []

    return filtered, all_scored
