import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.retriever import HybridRetriever, format_source

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Intra DocBot (PoC)", layout="wide")
st.title("Intra DocBot (PoC)")

with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("최종 top-k", min_value=3, max_value=10, value=5, step=1)
    dense_k = st.slider("dense 후보 k", min_value=5, max_value=50, value=20, step=1)
    bm25_k = st.slider("bm25 후보 k", min_value=10, max_value=200, value=50, step=5)
    alpha = st.slider(
        "alpha (dense 비중)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

    st.caption("출처 표시는 파일명 + 페이지/슬라이드만 표시합니다.")


@st.cache_resource
def get_retriever():
    return HybridRetriever()


@st.cache_resource
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


retriever = get_retriever()
llm = get_llm()

q = st.text_input("질문", placeholder="예: 휴가 신청은 어디서 어떻게 해?")


def build_context(hits):
    # 컨텍스트 너무 길어지지 않게 상위 몇 개만 사용
    blocks = []
    for h in hits:
        md = h["metadata"]
        src = format_source(md)
        text = (h["text"] or "").strip()
        if not text:
            continue
        blocks.append(f"[{src}]\n{text}")
    return "\n\n".join(blocks)


if st.button("검색/답변"):
    if not q.strip():
        st.warning("질문을 입력해.")
        st.stop()

    hits = retriever.hybrid_search(
        q,
        top_k=top_k,
        dense_k=dense_k,
        bm25_k=bm25_k,
        alpha=alpha,
    )

    context = build_context(hits)

    prompt = f"""너는 사내 문서 기반 Q&A 도우미야.
아래 컨텍스트에 근거해서만 답해.
컨텍스트에 없으면 '문서에서 확인되지 않음'이라고 말해.

[질문]
{q}

[컨텍스트]
{context}
"""

    with st.spinner("답변 생성 중..."):
        answer = llm.invoke(prompt).content

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("답변")
        st.write(answer)

    with col2:
        st.subheader("출처")
        # 파일명+페이지/슬라이드만
        sources = []
        for h in hits:
            sources.append(format_source(h["metadata"]))
        # 중복 제거(순서 유지)
        uniq = []
        for s in sources:
            if s not in uniq:
                uniq.append(s)
        for s in uniq:
            st.write(f"- {s}")

    with st.expander("검색된 컨텍스트(상위)"):
        for i, h in enumerate(hits, 1):
            md = h["metadata"]
            st.markdown(
                f"**#{i} {format_source(md)}**  (score={h['score']:.3f}, dense={h['dense']:.3f}, bm25={h['bm25']:.3f})"
            )
            st.write((h["text"] or "")[:800])
