"""
main.py
─────────────────────────────────────────────────────────────────────────────
Streamlit 앱 진입점.

실행:
    streamlit run main.py

구조:
    - FastAPI를 HTTP로 호출하는 클라이언트 구조
      (직접 함수 호출 X → React 교체 시 FastAPI 엔드포인트 그대로 유지)
    - 사이드바: ENV=dev일 때만 파라미터 슬라이더 노출
    - 채팅 UI: 사용자(우측, 네이비) / AI(좌측) 말풍선
    - 출처 탭 + 원문 탭 (expander)
    - SSE 스트리밍 실시간 출력
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Generator, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── 설정 ─────────────────────────────────────────────────────────────────────
FASTAPI_BASE = (
    f"http://{os.getenv('FASTAPI_HOST', 'localhost')}:{os.getenv('FASTAPI_PORT', '8000')}"
)
API_TOKEN = os.getenv("FASTAPI_TOKEN", "dev-secret-token")
ENV = os.getenv("ENV", "dev")

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Samsung DocBot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 커스텀 CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* 사용자 말풍선 (우측, 네이비) */
.user-bubble {
    background-color: #1B2A4A;
    color: #FFFFFF;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    text-align: left;
    font-size: 14px;
    line-height: 1.6;
}
/* AI 말풍선 (좌측, 회색) */
.ai-bubble {
    background-color: #F0F2F6;
    color: #1A1A1A;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    font-size: 14px;
    line-height: 1.6;
}
/* 출처 뱃지 */
.source-badge {
    display: inline-block;
    background-color: #E8F0FE;
    color: #1967D2;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin: 2px;
}
/* 헤더 */
.chat-header {
    font-size: 24px;
    font-weight: 700;
    color: #1B2A4A;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── 세션 초기화 ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "sources": []}]
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "dense_threshold" not in st.session_state:
    st.session_state.dense_threshold = 0.3


# ─── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/24/Samsung_Logo.svg", width=120)
    st.markdown("## Samsung DocBot")
    st.markdown("사내 규정·제도 AI 어시스턴트")
    st.divider()

    # API 상태 확인
    try:
        resp = requests.get(f"{FASTAPI_BASE}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                st.success("API 연결됨")
            else:
                st.warning(f"API 경고: {data.get('status')}")
        else:
            st.error(f"API 오류: {resp.status_code}")
    except Exception:
        st.error("API 서버 연결 실패\n`uvicorn src.api.main:app` 실행 필요")

    st.divider()

    # Dev 환경에서만 파라미터 슬라이더 노출
    if ENV == "dev":
        st.markdown("### ⚙️ 검색 파라미터 (Dev)")
        st.session_state.top_k = st.slider(
            "검색 청크 수 (top_k)", min_value=1, max_value=20, value=st.session_state.top_k
        )
        st.session_state.temperature = st.slider(
            "LLM 온도 (temperature)", min_value=0.0, max_value=1.0,
            value=st.session_state.temperature, step=0.05
        )
        st.session_state.dense_threshold = st.slider(
            "Dense 유사도 임계값", min_value=0.0, max_value=1.0,
            value=st.session_state.dense_threshold, step=0.05
        )
        st.divider()

    # 동기화 버튼
    st.markdown("### 문서 관리")
    if st.button("문서 동기화", use_container_width=True):
        with st.spinner("동기화 중..."):
            try:
                resp = requests.post(
                    f"{FASTAPI_BASE}/api/admin/sync",
                    headers=HEADERS,
                    json={},
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"동기화 완료\n"
                        f"인덱싱: {data['indexed']} | 스킵: {data['skipped']} | 오류: {data['errors']}"
                    )
                elif resp.status_code == 409:
                    st.warning("동기화 이미 실행 중")
                else:
                    st.error(f"동기화 실패: {resp.status_code}")
            except Exception as e:
                st.error(f"요청 실패: {e}")

    # 대화 초기화
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── 메인 화면 ────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-header">📋 사내 규정·제도 AI 어시스턴트</div>', unsafe_allow_html=True)
st.caption("출장비, 복리후생, 영업 규정 등 사내 문서 기반으로 답변합니다.")
st.divider()

# ─── 대화 이력 렌더링 ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ai-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # 출처 탭 + 원문 탭
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"출처 {len(sources)}건 보기"):
                tab_names = ["출처 목록"] + [f"원문 {i+1}" for i in range(len(sources))]
                tabs = st.tabs(tab_names)
                with tabs[0]:
                    for i, src in enumerate(sources, start=1):
                        filename = src["source"].split("/")[-1].split("\\")[-1]
                        loc = ""
                        if src.get("page"):
                            loc = f" p.{src['page']}"
                        elif src.get("slide"):
                            loc = f" 슬라이드 {src['slide']}"
                        elif src.get("sheet"):
                            loc = f" {src['sheet']}"
                        score_str = ""
                        if src.get("rerank_score") is not None:
                            score_str = f" (점수: {src['rerank_score']:.3f})"
                        st.markdown(
                            f'<span class="source-badge">{i}. {filename}{loc}{score_str}</span>',
                            unsafe_allow_html=True,
                        )
                for i, src in enumerate(sources):
                    with tabs[i + 1]:
                        st.text(src.get("content_preview", ""))


# ─── 입력창 ───────────────────────────────────────────────────────────────────
user_input = st.chat_input("규정·제도에 대해 질문하세요...")

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    st.markdown(
        f'<div class="user-bubble">{user_input}</div>',
        unsafe_allow_html=True,
    )

    # SSE 스트리밍 요청
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # 현재 질문 제외
        if m["role"] in ("user", "assistant")
    ]

    payload = {
        "question": user_input,
        "history": history_payload,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "dense_threshold": st.session_state.dense_threshold,
    }

    ai_placeholder = st.empty()
    full_answer = ""
    collected_sources: list[dict] = []
    error_occurred = False

    try:
        with requests.post(
            f"{FASTAPI_BASE}/api/query/stream",
            headers={**HEADERS, "Accept": "text/event-stream"},
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    # 빈 줄: SSE 이벤트 구분자
                    if buffer:
                        # 버퍼 처리
                        buffer = ""
                    continue

                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                    buffer = event_type
                    continue

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    event_type = buffer if buffer else "token"

                    if event_type == "token":
                        full_answer += data_str
                        ai_placeholder.markdown(
                            f'<div class="ai-bubble">{full_answer}</div>',
                            unsafe_allow_html=True,
                        )

                    elif event_type == "source":
                        try:
                            src = json.loads(data_str)
                            collected_sources.append(src)
                        except json.JSONDecodeError:
                            pass

                    elif event_type == "done":
                        break

                    elif event_type == "error":
                        st.error(f"오류: {data_str}")
                        error_occurred = True
                        break

                    buffer = ""

    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다. `uvicorn src.api.main:app` 실행 필요")
        error_occurred = True
    except requests.exceptions.Timeout:
        st.error("응답 시간 초과")
        error_occurred = True
    except Exception as e:
        st.error(f"요청 실패: {e}")
        error_occurred = True

    if not error_occurred and full_answer:
        # 대화 이력에 AI 응답 추가
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer,
                "sources": collected_sources,
            }
        )

        # 출처 바로 표시
        if collected_sources:
            with st.expander(f"출처 {len(collected_sources)}건 보기"):
                tab_names = ["출처 목록"] + [f"원문 {i+1}" for i in range(len(collected_sources))]
                tabs = st.tabs(tab_names)
                with tabs[0]:
                    for i, src in enumerate(collected_sources, start=1):
                        filename = src["source"].split("/")[-1].split("\\")[-1]
                        loc = ""
                        if src.get("page"):
                            loc = f" p.{src['page']}"
                        elif src.get("slide"):
                            loc = f" 슬라이드 {src['slide']}"
                        elif src.get("sheet"):
                            loc = f" {src['sheet']}"
                        score_str = ""
                        if src.get("rerank_score") is not None:
                            score_str = f" (점수: {src['rerank_score']:.3f})"
                        st.markdown(
                            f'<span class="source-badge">{i}. {filename}{loc}{score_str}</span>',
                            unsafe_allow_html=True,
                        )
                for i, src in enumerate(collected_sources):
                    with tabs[i + 1]:
                        st.text(src.get("content_preview", ""))
