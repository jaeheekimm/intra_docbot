from dotenv import load_dotenv

load_dotenv()


import os
import json
import pickle
import hashlib
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

kiwi = Kiwi()


from src.utils.paths import OUT_JSONL as JSONL_PATH, BM25_PATH

# ingest_chroma.py와 청킹 설정 반드시 동일하게 유지
PDF_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""],
)

PPT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60,
    separators=["\n\n", "\n", " ", ""],
)
PPT_SPLIT_THRESHOLD = 800

# 청킹 로직 바뀌면 올려야 함 (ingest_chroma.py도 같이)
PIPELINE_VERSION = "v5"


def load_docs_from_jsonl(path: str) -> List[Document]:
    """JSONL 파일에서 Document 리스트 로드"""
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj.get("page_content", ""),
                    metadata=obj.get("metadata", {}),
                )
            )
    return docs


def make_chunk_id(meta, text, local_idx):
    """청크 고유 ID 생성. ingest_chroma.py와 반드시 동일해야 HybridRetriever에서 병합 가능

    키: source | doc_type | page | slide | sheet | row | len(text) | text[:120]
    """
    base = (
        f"{meta.get('source','')}|{meta.get('doc_type','')}|"
        f"{meta.get('page','')}|{meta.get('slide','')}|{meta.get('sheet','')}|{meta.get('row','')}|"
        f"{len(text)}|{text[:120]}"
    )
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


def tokenize(text):
    """Kiwi 형태소 분석으로 명사(N)/동사(V)만 추출. retriever.py와 동일한 로직 유지"""
    tokens = []
    for tok in kiwi.tokenize(text):
        if tok.tag.startswith("N") or tok.tag.startswith("V"):
            tokens.append(tok.form)
    return tokens


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """pickle 저장용으로 metadata를 단순 타입으로 정리

    None 제거, list/dict는 JSON 문자열로 변환
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean


def split_by_type(docs: List[Document]) -> List[Document]:
    """ingest_chroma.py와 동일한 청킹 로직 (chunk_id 일치 필수)"""
    out: List[Document] = []
    for d in docs:
        doc_type = (d.metadata.get("doc_type") or "").lower()
        text = (d.page_content or "").strip()

        if len(text) < 20:
            continue

        if doc_type == "pdf":
            out.extend(PDF_SPLITTER.split_documents([d]))
        elif doc_type == "pptx":
            if len(text) >= PPT_SPLIT_THRESHOLD:
                out.extend(PPT_SPLITTER.split_documents([d]))
            else:
                out.append(d)
        elif doc_type == "xlsx":
            out.append(d)
        else:
            out.extend(PDF_SPLITTER.split_documents([d]))
    return out


def build_source_fingerprints(chunks):
    """source 단위로 fingerprint 계산. ingest_chroma.py와 동일 로직"""
    by_source = defaultdict(list)
    for c in chunks:
        src = c.metadata.get("source") or ""
        if not src:
            continue
        by_source[src].append((c.page_content or "").strip())

    fps = {}
    for src, texts in by_source.items():
        joined = "\n".join(texts)
        fps[src] = hashlib.sha1(
            (PIPELINE_VERSION + joined).encode("utf-8", errors="ignore")
        ).hexdigest()
    return fps


def main() -> None:
    """JSONL 로드 → 청킹 → fingerprint 비교 → BM25 인덱스 생성/갱신"""
    if not os.path.exists(JSONL_PATH):
        raise RuntimeError(f"{JSONL_PATH} 없음. 먼저 extract 실행해서 jsonl 생성해.")

    docs = load_docs_from_jsonl(JSONL_PATH)
    chunks = split_by_type(docs)

    source_fp = build_source_fingerprints(chunks)

    # 기존 pkl이 있고 fingerprint가 같으면 재생성 생략
    if os.path.exists(BM25_PATH):
        try:
            with open(BM25_PATH, "rb") as f:
                old_payload = pickle.load(f)
            if old_payload.get("source_fp_map") == source_fp:
                print("변경 없음. BM25 재생성 생략.")
                return
        except Exception:
            pass

    chunk_ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, c in enumerate(chunks):
        text = (c.page_content or "").strip()

        title = c.metadata.get("title")
        doc_type = (c.metadata.get("doc_type") or "").lower()
        file_name = c.metadata.get("file_name") or ""

        # ingest_chroma.py와 동일하게 embed_text 구성 (chunk_id 일치 필수)
        if title and isinstance(title, str):
            text = f"[TITLE] {title}\n{text}".strip()
        elif doc_type == "pdf" and file_name:
            text = f"[문서] {file_name}\n{text}".strip()

        if len(text) < 20:
            continue

        cid = make_chunk_id(c.metadata, text, i)
        md = sanitize_metadata(c.metadata)
        md["chunk_id"] = cid

        chunk_ids.append(cid)
        texts.append(text)
        metadatas.append(md)

    if not texts:
        raise RuntimeError(
            f"BM25용 텍스트가 0개입니다. JSONL={JSONL_PATH} / docs={len(docs)} / chunks={len(chunks)}"
        )

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "chunk_ids": chunk_ids,
        "texts": texts,
        "metadatas": metadatas,
        "bm25": bm25,
        "source_fp_map": source_fp,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"OK bm25_index={BM25_PATH}")
    print(f"- docs={len(docs)} chunks_for_bm25={len(texts)}")


if __name__ == "__main__":
    main()
