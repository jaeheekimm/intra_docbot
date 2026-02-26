# src/app.py
import os
import sys
from pathlib import Path
import subprocess

# ⭐ 제일 먼저 ENV부터 세팅
os.environ["DATA_DIR"] = "./data"
os.environ["JSONL_PATH"] = "/tmp/parsed_documents.jsonl"
os.environ["OUT_IMG_DIR"] = "/tmp/extracted_images"
os.environ["OUT_IMG_MANIFEST"] = "/tmp/image_manifest.json"
os.environ["CHROMA_DIR"] = "/tmp/chroma_db"
os.environ["BM25_PATH"] = "/tmp/bm25_index.pkl"
os.environ["CHROMA_COLLECTION"] = "intra_docs"

# 그 다음에 나머지 import
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.chains.rag_chain import get_rag_chain


def ensure_indexes():
    chroma_dir = Path(os.environ["CHROMA_DIR"])
    bm25_path = Path(os.environ["BM25_PATH"])

    if chroma_dir.exists() and bm25_path.exists():
        return

    # ⭐ Streamlit이 실행 중인 동일한 Python(venv)로 실행해야 requirements가 적용됨
    python_exec = sys.executable

    subprocess.run([python_exec, "-m", "src.pipeline.extract"], check=True)
    subprocess.run([python_exec, "-m", "src.pipeline.ingest_chroma"], check=True)
    subprocess.run([python_exec, "-m", "src.pipeline.bm25_index"], check=True)


ensure_indexes()

st.set_page_config(page_title="Intra DocBot (PoC)", layout="wide")
st.title("Intra DocBot (PoC)")


# -----------------------------
# Session state 초기화
# -----------------------------
if "messages" not in st.session_state:
    # 각 원소: {"role": "user"|"assistant", "content": str, "sources": [...], "hits": [...]}
    st.session_state["messages"] = []

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = "답변은 친절하게, 사실만. 줄바꿈 깔끔하게"

if "history_pairs" not in st.session_state:
    st.session_state["history_pairs"] = 3  # 최근 3쌍(=6개 메시지)


# -----------------------------
# Sidebar (설정 UI)
# -----------------------------
with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("최종 top-k", 3, 10, 5, 1)
    dense_k = st.slider("dense 후보 k", 5, 50, 20, 1)
    bm25_k = st.slider("bm25 후보 k", 10, 200, 60, 5)
    alpha = st.slider("alpha (dense 비중)", 0.0, 1.0, 0.6, 0.05)

    st.divider()
    st.header("대화 설정")
    st.session_state["history_pairs"] = st.slider(
        "기억할 최근 대화(쌍)", 0, 5, st.session_state["history_pairs"], 1
    )

    st.divider()
    st.header("프롬프트")
    st.session_state["user_prompt"] = st.text_area(
        "추가 지침(선택)",
        value=st.session_state["user_prompt"],
        height=120,
        help="이 지침을 질문에 함께 붙여서 모델이 따르도록 유도합니다.",
    )

    st.divider()
    if st.button("대화내용 초기화"):
        st.session_state["messages"].clear()
        st.success("초기화 완료")


@st.cache_resource
def get_chain_cached(_top_k: int, _dense_k: int, _bm25_k: int, _alpha: float):
    return get_rag_chain(top_k=_top_k, dense_k=_dense_k, bm25_k=_bm25_k, alpha=_alpha)


# -----------------------------
# Util: 최근 대화 -> 문자열
# -----------------------------
def _build_recent_history_text(messages: List[Dict[str, Any]], pairs: int) -> str:
    if pairs <= 0:
        return ""
    recent = messages[-2 * pairs :]
    lines: List[str] = []
    for m in recent:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"사용자: {content}")
        else:
            lines.append(f"AI: {content}")
    return "\n".join(lines).strip()


def _compose_query(user_input: str) -> str:
    user_input = (user_input or "").strip()

    history_txt = _build_recent_history_text(
        st.session_state["messages"], st.session_state["history_pairs"]
    )

    prompt_txt = (st.session_state.get("user_prompt") or "").strip()

    parts: List[str] = []
    if prompt_txt:
        parts.append(f"[추가지침]\n{prompt_txt}")
    if history_txt:
        parts.append(f"[최근대화]\n{history_txt}")
    parts.append(f"[현재질문]\n{user_input}")

    return "\n\n".join(parts).strip()


# -----------------------------
# 기존 대화 출력 (채팅 UI)
# -----------------------------
def _render_sources_and_hits(msg: Dict[str, Any]):
    sources = msg.get("sources") or []
    hits = msg.get("hits") or []

    if sources:
        st.markdown("**출처**")
        for s in sources:
            st.write(f"- {s}")

    if hits:
        with st.expander("검색된 컨텍스트(상위)"):
            for i, h in enumerate(hits, 1):
                md = h.get("metadata", {}) or {}
                st.markdown(
                    f"**#{i} {md.get('file_name','')}** "
                    f"(score={float(h.get('score',0)):.3f}, dense={float(h.get('dense',0)):.3f}, bm25={float(h.get('bm25',0)):.3f})"
                )
                st.write((h.get("text") or "")[:800])


for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_sources_and_hits(msg)


# -----------------------------
# 입력 -> 스트리밍 응답
# -----------------------------
user_input = st.chat_input("질문을 입력해 (예: 휴가 신청은 어디서 어떻게 해?)")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    chain = get_chain_cached(top_k, dense_k, bm25_k, alpha)
    query = _compose_query(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_accum = ""
        last_chunk = None

        try:
            for chunk in chain.stream(query):
                if not isinstance(chunk, dict):
                    continue

                last_chunk = chunk

                if "answer" in chunk and chunk["answer"] is not None:
                    part = chunk["answer"]

                    if isinstance(part, str):
                        if len(part) >= len(answer_accum):
                            answer_accum = part
                        else:
                            answer_accum += part

                        placeholder.markdown(answer_accum)

        except Exception as e:
            placeholder.error(f"에러: {repr(e)}")
            st.stop()

        sources = (last_chunk or {}).get("sources") or []
        hits = (last_chunk or {}).get("hits") or []

        if sources:
            st.markdown("**출처**")
            for s in sources:
                st.write(f"- {s}")

        if hits:
            with st.expander("검색된 컨텍스트(상위)"):
                for i, h in enumerate(hits, 1):
                    md = h.get("metadata", {}) or {}
                    st.markdown(
                        f"**#{i} {md.get('file_name','')}** "
                        f"(score={float(h.get('score',0)):.3f}, dense={float(h.get('dense',0)):.3f}, bm25={float(h.get('bm25',0)):.3f})"
                    )
                    st.write((h.get("text") or "")[:800])

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_accum.strip(),
            "sources": sources,
            "hits": hits,
        }
    )
