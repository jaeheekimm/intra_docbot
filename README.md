# PentAssistant

사내 내부 문서 기반 AI 검색 챗봇입니다.
PDF, PPTX, XLSX 문서를 인덱싱하고 직원 질문에 한국어로 답변합니다.

---

## 주요 기능

- **하이브리드 검색**: 의미 기반 벡터 검색(ChromaDB) + 키워드 검색(BM25) 결합
- **한국어 형태소 분석**: Kiwi를 활용한 명사/동사 기반 토크나이징
- **다양한 문서 형식 지원**: PDF, PowerPoint, Excel
- **실시간 스트리밍 답변**: LangChain + OpenAI GPT 기반
- **출처 표시**: 답변에 활용된 원문 문서 및 페이지 표시
- **대화 맥락 유지**: 이전 대화를 참고한 질문 재작성(Query Rewriting)

---

## 데이터 흐름

1. `./data/` 의 PDF·PPTX·XLSX 파일을 **`extract.py`** 가 파싱해 JSONL로 저장
2. **`ingest_chroma.py`** 가 청크를 임베딩해 ChromaDB에 저장
3. **`bm25_index.py`** 가 동일 청크로 BM25 키워드 인덱스 생성
4. **`HybridRetriever`** 가 두 인덱스를 병합해 가중 점수로 순위 결정
5. **`rag_chain.py`** 가 상위 청크를 컨텍스트로 GPT에 전달해 답변 생성
6. **`app.py`** (Streamlit) 가 답변을 스트리밍으로 화면에 출력

---

## 설치 및 실행

### 1. 가상환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini              # 선택사항 (기본값: gpt-4o-mini)
EMBEDDING_MODEL=text-embedding-3-small  # 선택사항

# LangSmith 추적 (선택사항)
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

경로 재정의 (기본값으로 동작하므로 보통 생략 가능):

```env
DATA_DIR=./data
JSONL_PATH=./data/parsed_documents.jsonl
CHROMA_DIR=./indexes/chroma_db
BM25_PATH=./indexes/bm25_index.pkl
CHROMA_COLLECTION=intra_docs
```

### 3. 문서 준비

`./data/` 폴더에 PDF, PPTX, XLSX 파일을 넣습니다.

### 4. 앱 실행

```bash
streamlit run src/app.py
```

최초 실행 시 인덱스가 없으면 자동으로 파이프라인 전체(파싱 → 임베딩 → BM25 인덱싱)를 실행합니다.

---

## 파이프라인 수동 실행

```bash
# Step 1: 문서 파싱 → JSONL + 이미지 추출
python -m src.pipeline.extract

# Step 2: 청크 임베딩 → ChromaDB 저장
python -m src.pipeline.ingest_chroma

# Step 3: BM25 키워드 인덱스 생성
python -m src.pipeline.bm25_index
```

---

## 사이드바 검색 설정

| 설정 | 설명 | 기본값 |
|---|---|---|
| 출처 원문 개수 | 최종 반환할 청크 수 (top_k) | 5 |
| 의미기반 검색 | dense 검색 후보 수 | 15 |
| 키워드 검색 | BM25 검색 후보 수 | 30 |
| 의미기반 검색의 비중 | dense vs BM25 가중치 (alpha) | 0.6 |
| 기억할 최근 대화 세트 | 대화 맥락 유지 범위 | 3 |
| 출처 유사도 기준 | 출처 표시 threshold | 0.40 |

---

## 프로젝트 구조

```
intra_docbot/
├── src/
│   ├── app.py                  # Streamlit UI 진입점
│   ├── retriever.py            # HybridRetriever (dense + BM25 융합)
│   ├── chains/
│   │   └── rag_chain.py        # LangChain RAG 체인, 프롬프트, 출처 필터링
│   ├── parsers/
│   │   ├── pdf_parser.py       # PyMuPDF + PyPDFLoader
│   │   ├── pptx_parser.py      # python-pptx (슬라이드 단위)
│   │   └── xlsx_parser.py      # openpyxl (행 단위 / 시트 모드)
│   ├── pipeline/
│   │   ├── extract.py          # 문서 파싱 실행
│   │   ├── ingest_chroma.py    # ChromaDB 임베딩 (증분 업데이트)
│   │   └── bm25_index.py       # BM25 인덱스 빌드
│   └── utils/
│       ├── paths.py            # 경로 및 환경변수 관리
│       ├── hashing.py          # 파일 SHA1 해시 (증분 업데이트용)
│       └── files.py            # 파일 유틸리티
├── data/                       # 원본 문서 저장 위치
├── indexes/                    # ChromaDB, BM25 인덱스 저장 위치
├── requirements.txt
└── .env
```

---

## 기술 스택

- **LLM**: OpenAI GPT (gpt-4o-mini 기본)
- **임베딩**: OpenAI text-embedding-3-small
- **벡터 DB**: ChromaDB
- **키워드 검색**: rank-bm25
- **한국어 형태소**: kiwipiepy
- **프레임워크**: LangChain, Streamlit
- **문서 파싱**: PyMuPDF, python-pptx, openpyxl
