import os, sys, html as html_lib, re
from pathlib import Path
import subprocess
from typing import Any, Dict, List
import streamlit as st

# 1) 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2) ENV
os.environ["DATA_DIR"] = "./data"
os.environ["JSONL_PATH"] = "/tmp/parsed_documents.jsonl"
os.environ["OUT_IMG_DIR"] = "/tmp/extracted_images"
os.environ["OUT_IMG_MANIFEST"] = "/tmp/image_manifest.json"
os.environ["CHROMA_DIR"] = "/tmp/chroma_db"
os.environ["BM25_PATH"] = "/tmp/bm25_index.pkl"
os.environ["CHROMA_COLLECTION"] = "intra_docs"

from dotenv import load_dotenv

load_dotenv()

from src.chains.rag_chain import get_rag_parts
import base64


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

st.set_page_config(page_title="PentAssistant", layout="wide")


def img_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


logo_b64 = img_to_base64("src/Aviator.png")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton,
.stSlider, label, p, div, textarea {{
    font-family: 'Noto Sans KR', sans-serif !important;
}}

.title-container {{
    display: flex; align-items: center; gap: 18px;
    padding: 32px 0 24px 0;
    border-bottom: 3px solid #e4effb;
    margin-bottom: 8px;
}}
.title-container img {{ height: 56px; width: auto; }}
.title-container h1 {{
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 2.4rem !important; font-weight: 900 !important;
    color: #1B3A6B !important; letter-spacing: -0.5px;
    margin: 0 !important; line-height: 1.1 !important;
}}
.title-sub {{ font-size: 0.85rem; color: #5A7A9F; margin-top: 4px; font-weight: 400; }}

.chat-set {{ margin-bottom: 5px; }}

/* 사용자 말풍선 */
.user-bubble-wrap {{
    display: flex; justify-content: flex-end;
    align-items: flex-end; gap: 10px; margin: 8px 0 24px 0;
}}
.user-bubble {{
    background-color: #1B3A6B; color: #FFFFFF;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; max-width: 70%;
    font-size: 0.95rem; line-height: 1.6;
    word-break: break-word; white-space: pre-wrap;
}}
.user-avatar {{
    width: 36px; height: 36px; border-radius: 50%;
    background-color: #1B3A6B;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}}

/* AI 말풍선 (히스토리 - 로고 이미지) */
.ai-bubble-wrap {{
    display: flex; align-items: flex-start; gap: 10px; margin: 8px 0 4px 0;
}}
.ai-avatar {{
    width: 36px; height: 36px; flex-shrink: 0; margin-top: 2px;
}}
.ai-avatar img {{ width: 36px; height: 36px; object-fit: contain; }}
.ai-bubble {{
    max-width: 75%; font-size: 0.95rem;
    line-height: 1.7; color: #0D0D0D; padding-top: 4px;
}}

/* 스트리밍 중 텍스트 아이콘 */
.ai-avatar-text {{
    width: 36px; height: 36px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    margin-top: 2px;
    font-size: 0.6rem; font-weight: 900;
    color: #1B3A6B; line-height: 1.15;
    text-align: center; letter-spacing: -0.3px;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="title-container">
    <img src="data:image/png;base64,{logo_b64}" />
    <div>
        <h1>PentAssistant</h1>
        <div class="title-sub">내부 문서 기반 AI 검색 시스템</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "history_pairs" not in st.session_state:
    st.session_state["history_pairs"] = 3

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("출처 원문 개수", 3, 10, 5, 1)
    dense_k = st.slider("의미기반 검색", 5, 50, 15, 1)
    bm25_k = st.slider("키워드 검색", 10, 200, 30, 5)
    alpha = st.slider("의미기반 검색의 비중", 0.0, 1.0, 0.6, 0.05)
    st.divider()
    st.header("대화 설정")
    st.session_state["history_pairs"] = st.slider(
        "기억할 최근 대화 세트", 0, 5, st.session_state["history_pairs"], 1
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


# ── Util ──────────────────────────────────────────────────
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
        lines.append(f"사용자: {content}" if role == "user" else f"AI: {content}")
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


def _render_user_bubble(content: str):
    escaped = html_lib.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""<div class="user-bubble-wrap">
            <div class="user-bubble">{escaped}</div>
            <div class="user-avatar">나</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_ai_bubble(content: str):
    col_icon, col_text = st.columns([0.06, 0.93])
    with col_icon:
        st.image("src/Aviator_bot.png", width=60)
    with col_text:
        st.markdown(content)


def _stream_and_render(answer_r, state) -> str:
    col_icon, col_text = st.columns([0.06, 0.93])
    with col_icon:
        st.image("src/Aviator_bot.png", width=60)
    with col_text:
        placeholder = st.empty()
        answer_accum = ""
        for part in answer_r.stream(state):
            if isinstance(part, str) and part:
                answer_accum += part
                placeholder.markdown(answer_accum)
    return answer_accum


def _render_hits_expander(hits: list, key_prefix: str):
    if not hits:
        return
    with st.expander("출처 및 원문", expanded=False):
        tabs = st.tabs(["출처", "원문"])
        with tabs[0]:
            seen = set()
            for h in hits:
                md = h.get("metadata", {}) or {}
                file_name = md.get("file_name") or md.get("source") or "unknown"
                page = md.get("page")
                key = (file_name, page)
                if key in seen:
                    continue
                seen.add(key)
                st.write(
                    f"- {file_name}"
                    + (f" / p.{page}" if page not in (None, "") else "")
                )
        with tabs[1]:
            for j, h in enumerate(hits, 1):
                md = h.get("metadata", {}) or {}
                file_name = md.get("file_name") or md.get("source") or "unknown"
                page = md.get("page")
                title = f"#{j} {file_name}" + (
                    f" / p.{page}" if page not in (None, "") else ""
                )
                st.markdown(f"**{title}**")
                st.text_area(
                    label=f"hit_{j}",
                    value=(h.get("text") or "").strip(),
                    height=180,
                    key=f"{key_prefix}_{j}",
                )


# ── 기존 대화 출력 ────────────────────────────────────────
messages = st.session_state["messages"]
i = 0
while i < len(messages):
    msg = messages[i]
    if msg["role"] == "user":
        assistant_msg = None
        if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            assistant_msg = messages[i + 1]

        st.markdown('<div class="chat-set">', unsafe_allow_html=True)
        _render_user_bubble(msg["content"])

        if assistant_msg:
            _render_ai_bubble(assistant_msg["content"])
            _render_hits_expander(
                assistant_msg.get("hits") or [], key_prefix=f"history_{i}"
            )

        st.markdown("</div>", unsafe_allow_html=True)
        i += 2 if assistant_msg else 1
    else:
        i += 1


# ── 신규 입력 ─────────────────────────────────────────────
user_input = st.chat_input("질문을 입력해 주세요.")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    st.markdown('<div class="chat-set">', unsafe_allow_html=True)
    _render_user_bubble(user_input)

    retrieve_r, answer_r = get_rag_parts(
        top_k=top_k, dense_k=dense_k, bm25_k=bm25_k, alpha=alpha
    )
    state = retrieve_r.invoke(_compose_query(user_input))
    hits = state.get("hits", [])

    answer_accum = _stream_and_render(answer_r, state)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_accum.strip(), "hits": hits}
    )
    _render_hits_expander(hits, key_prefix="new_hits")
    st.markdown("</div>", unsafe_allow_html=True)
    # st.rerun() 없음
