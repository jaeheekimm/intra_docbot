# PentAssistant

사내 내부 문서(PDF, PPTX, XLSX) 기반 한국어 AI 검색 챗봇입니다.

---

## 실행 방법

```bash
# 가상환경 활성화
.venv\Scripts\activate

# 앱 실행
streamlit run src/app.py
```

최초 실행 시 인덱스가 없으면 파싱 → 임베딩 → BM25 인덱싱을 자동으로 수행합니다.

---

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들어주세요.

```env
OPENAI_API_KEY=your_api_key_here

# 선택사항
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

경로 기본값 (`./data`, `./indexes/chroma_db`, `./indexes/bm25_index.pkl`)은 변경이 필요한 경우만 `.env`에 추가합니다.

---

## 파이프라인 수동 실행

```bash
python -m src.pipeline.extract        # 문서 파싱
python -m src.pipeline.ingest_chroma  # ChromaDB 임베딩
python -m src.pipeline.bm25_index     # BM25 인덱스 생성
```

---

## 동작 방식

사용자 질문이 들어오면 두 가지 검색을 병렬로 수행합니다.

- **Dense 검색** — OpenAI 임베딩 + ChromaDB 코사인 유사도
- **BM25 검색** — Kiwi 형태소 분석(명사/동사) 기반 키워드 검색

두 결과를 `score = alpha * dense + (1 - alpha) * bm25` 공식으로 합산한 뒤 상위 청크를 LLM 컨텍스트로 넘겨 답변을 생성합니다.

질문이 짧거나 대명사(그거, 이거 등)를 포함하면 이전 대화를 바탕으로 검색 쿼리를 자동으로 재작성합니다.

---

## 사이드바 설정

- **출처 원문 개수** — 최종 반환할 청크 수 (기본 5)
- **의미기반 검색** — dense 후보 수 (기본 15)
- **키워드 검색** — BM25 후보 수 (기본 30)
- **의미기반 검색의 비중** — alpha 값, 높을수록 의미 검색 우선 (기본 0.6)
- **기억할 최근 대화 세트** — 대화 맥락 유지 범위 (기본 3)
- **출처 유사도 기준** — 출처 표시 임계값 (기본 0.40)

---

## 기술 스택

- **LLM / 임베딩** — OpenAI (gpt-4o-mini, text-embedding-3-small)
- **벡터 DB** — ChromaDB
- **키워드 검색** — rank-bm25
- **한국어 형태소** — kiwipiepy
- **프레임워크** — LangChain, Streamlit
- **문서 파싱** — PyMuPDF, python-pptx, openpyxl
