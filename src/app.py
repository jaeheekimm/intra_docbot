import os, sys
from pathlib import Path
import subprocess
from typing import Any, Dict, List

# 1) 경로 먼저
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2) ENV 먼저
os.environ["DATA_DIR"] = "./data"
os.environ["JSONL_PATH"] = "/tmp/parsed_documents.jsonl"
os.environ["OUT_IMG_DIR"] = "/tmp/extracted_images"
os.environ["OUT_IMG_MANIFEST"] = "/tmp/image_manifest.json"
os.environ["CHROMA_DIR"] = "/tmp/chroma_db"
os.environ["BM25_PATH"] = "/tmp/bm25_index.pkl"
os.environ["CHROMA_COLLECTION"] = "intra_docs"

# 3) 라이브러리 import
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.chains.rag_chain import get_rag_parts


def ensure_indexes():
    chroma_dir = Path(os.environ["CHROMA_DIR"])
    bm25_path = Path(os.environ["BM25_PATH"])

    if chroma_dir.exists() and bm25_path.exists():
        return

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
    st.session_state["messages"] = []

if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = "답변은 친절하게, 사실만. 줄바꿈 깔끔하게"

if "history_pairs" not in st.session_state:
    st.session_state["history_pairs"] = 3


# -----------------------------
# Sidebar
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
    )

    st.divider()
    if st.button("대화내용 초기화"):
        st.session_state["messages"].clear()
        st.success("초기화 완료")


# -----------------------------
# Util
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


def _render_sources_and_hits(msg: Dict[str, Any]):
    hits = msg.get("hits") or []
    if not hits:
        return

    # 1) 출처(파일/페이지) 만들기: 파일별 페이지 모으기
    grouped: Dict[str, set] = {}
    for h in hits:
        md = h.get("metadata", {}) or {}
        file_name = md.get("file_name") or md.get("source") or "unknown"
        page = md.get("page")
        grouped.setdefault(file_name, set())
        if page is not None and page != "":
            grouped[file_name].add(page)

    # 보기 좋게 정렬
    sources_view = []
    for file_name in sorted(grouped.keys()):
        pages = sorted(list(grouped[file_name]))
        if pages:
            sources_view.append((file_name, pages))
        else:
            sources_view.append((file_name, []))

    tabs = st.tabs(["출처(파일/페이지)", "Top-K 원문"])

    with tabs[0]:
        with st.expander("출처 열기", expanded=False):
            for file_name, pages in sources_view:
                if pages:
                    st.write(f"- {file_name} / p. {', '.join(map(str, pages))}")
                else:
                    st.write(f"- {file_name}")

    with tabs[1]:
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            file_name = md.get("file_name") or md.get("source") or "unknown"
            page = md.get("page")
            title = f"#{i} {file_name}" + (
                f" / p.{page}" if page not in (None, "") else ""
            )
            with st.expander(title, expanded=False):
                st.write((h.get("text") or "").strip())


# -----------------------------
# 기존 대화 출력
# -----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])
        # if msg["role"] == "assistant":
        #     _render_sources_and_hits(msg)


# -----------------------------
# 입력 → 검색 1번 → LLM Stream → 출처 출력
# -----------------------------
user_input = st.chat_input("질문을 입력해")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    retrieve_r, answer_r = get_rag_parts(
        top_k=top_k,
        dense_k=dense_k,
        bm25_k=bm25_k,
        alpha=alpha,
    )

    query = _compose_query(user_input)

    # ⭐ 1️⃣ 검색 한 번만
    state = retrieve_r.invoke(query)
    hits = state.get("hits", [])

    # ⭐ 출처 생성 (파일명_페이지)
    sources = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        file_name = md.get("file_name", "unknown")
        page = md.get("page", "")
        sources.append(f"{file_name}_{page}")

    sources = list(dict.fromkeys(sources))

    # ⭐ 2️⃣ LLM만 stream
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_accum = ""

        for part in answer_r.stream(state):
            if isinstance(part, str) and part:
                answer_accum += part
                placeholder.markdown(answer_accum)

if hits:
    tabs = st.tabs(["출처", "Top-K 원문"])

    # 출처 탭
    with tabs[0]:
        st.markdown("**출처**")
        seen = set()
        for h in hits:
            md = h.get("metadata", {}) or {}
            file_name = md.get("file_name") or md.get("source") or "unknown"
            page = md.get("page")
            key = (file_name, page)
            if key in seen:
                continue
            seen.add(key)

            if page not in (None, ""):
                st.write(f"- {file_name} / p.{page}")
            else:
                st.write(f"- {file_name}")

    # Top-K 탭 (expander 없이 안전하게)
    with tabs[1]:
        for i, h in enumerate(hits, 1):
            md = h.get("metadata", {}) or {}
            file_name = md.get("file_name") or md.get("source") or "unknown"
            page = md.get("page")
            title = f"#{i} {file_name}" + (
                f" / p.{page}" if page not in (None, "") else ""
            )

            st.markdown(f"**{title}**")
            st.text_area(
                label=f"hit_{i}",
                value=(h.get("text") or "").strip(),
                height=180,
                key=f"hit_text_{i}",
            )
            st.divider()

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_accum.strip(),
            "hits": hits,
        }
    )
