"""
전체 프로젝트의 경로·모델 설정을 한 곳에서 관리.

로컬 개발: 프로젝트 루트의 .env 파일로 경로 지정
배포 환경: 플랫폼 환경변수로 지정 (Streamlit Cloud 등)

경로 환경변수 예시 (.env):
    DATA_DIR=./data
    CHROMA_DIR=./indexes/chroma_db
    BM25_PATH=./indexes/bm25_index.pkl
    EMBEDDING_MODEL=text-embedding-3-small
    LLM_MODEL=gpt-4o
"""

import os

# ── 데이터 경로 ────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")

OUT_JSONL = os.getenv("JSONL_PATH", os.path.join(DATA_DIR, "parsed_documents.jsonl"))
OUT_IMG_DIR = os.getenv("OUT_IMG_DIR", os.path.join(DATA_DIR, "extracted_images"))
OUT_IMG_MANIFEST = os.getenv(
    "OUT_IMG_MANIFEST", os.path.join(DATA_DIR, "image_manifest.json")
)

# ── 인덱스 경로 ────────────────────────────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", "./indexes/chroma_db")
BM25_PATH = os.getenv("BM25_PATH", "./indexes/bm25_index.pkl")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "intra_docs")

# ── 모델 설정 ──────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
