from dotenv import load_dotenv

load_dotenv()

import os
import json
import hashlib
from collections import defaultdict
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

from src.utils.paths import OUT_JSONL as JSONL_PATH, CHROMA_DIR, CHROMA_COLLECTION as COLLECTION, EMBED_MODEL

PIPELINE_VERSION = "v5"

# PDF 기본 청킹
PDF_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""],
)

# PPT는 보통 슬라이드 단위가 짧아서 split 안 하는게 자연스러움.
# 다만 슬라이드에 텍스트가 너무 길면만 안전장치로 split.
PPT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60,
    separators=["\n\n", "\n", " ", ""],
)

PPT_SPLIT_THRESHOLD = 800  # 이 길이 넘으면 split


def make_chunk_id(meta, text, local_idx):
    base = (
        f"{meta.get('source','')}|{meta.get('doc_type','')}|"
        f"{meta.get('page','')}|{meta.get('slide','')}|{meta.get('sheet','')}|{meta.get('row','')}|"
        f"{len(text)}|{text[:120]}"  # ← local_idx 대신 len(text)
    )
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


def load_docs_from_jsonl(path: str) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj.get("page_content", "") or "",
                    metadata=obj.get("metadata", {}) or {},
                )
            )
    return docs


def split_by_type(docs: List[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        doc_type = (d.metadata.get("doc_type") or "").lower()
        text = (d.page_content or "").strip()

        # 텍스트 없는 건 벡터화 의미 없어서 skip
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
            # 이미 행 단위라 split 안 함
            out.append(d)

        else:
            out.extend(PDF_SPLITTER.split_documents([d]))

    return out


def sanitize_metas(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chroma metadata는 str/int/float/bool만 허용.
    None 제거, dict/list 등 복잡 타입은 str로 변환.
    """
    cleaned: List[Dict[str, Any]] = []
    for m in metas:
        m = m or {}
        out: Dict[str, Any] = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        cleaned.append(out)
    return cleaned


def build_source_fingerprints(chunks):
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


# def build_source_fingerprints(chunks: List[Document]) -> Dict[str, str]:
#     """
#     source(파일) 단위로 전체 텍스트를 합쳐 fingerprint(sha1)을 만든다.
#     - 내용이 바뀌면 fingerprint가 바뀌므로, 해당 source만 delete+readd 가능.
#     """
#     by_source: Dict[str, List[str]] = defaultdict(list)

#     for c in chunks:
#         src = c.metadata.get("source") or ""
#         if not src:
#             continue
#         by_source[src].append((c.page_content or "").strip())

#     fps: Dict[str, str] = {}
#     for src, texts in by_source.items():
#         joined = "\n".join(texts)
#         fps[src] = hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()

#     return fps


def main():
    if not os.path.exists(JSONL_PATH):
        raise RuntimeError(
            f"{JSONL_PATH} 없음. 먼저 python -m src.pipeline.extract 실행"
        )

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 1) 로드 & 타입별 청킹
    docs = load_docs_from_jsonl(JSONL_PATH)
    chunks = split_by_type(docs)

    # 2) source 단위 fingerprint 생성(변경 감지용)
    source_fp = build_source_fingerprints(chunks)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # 3) chunk_id/source_fp 포함해서 적재 준비
    for i, c in enumerate(chunks):
        src = c.metadata.get("source")
        if src:
            c.metadata["source_fp"] = source_fp.get(src)

        title = c.metadata.get("title")
        content = (c.page_content or "").strip()
        doc_type = (c.metadata.get("doc_type") or "").lower()
        file_name = c.metadata.get("file_name") or ""

        if title and isinstance(title, str):
            embed_text = f"[TITLE] {title}\n{content}".strip()
        elif doc_type == "pdf" and file_name:
            embed_text = f"[문서] {file_name}\n{content}".strip()
        else:
            embed_text = content

        if not embed_text:
            continue

        # ← embed_text 만든 후에 chunk_id 계산 (bm25와 동일)
        cid = make_chunk_id(c.metadata, embed_text, i)
        c.metadata["chunk_id"] = cid

        texts.append(embed_text)
        metas.append(c.metadata)
        ids.append(cid)

    # (추가) image_paths는 Chroma metadata에 넣지 말고 image_count로만 유지
    for m in metas:
        if "image_paths" in m:
            ips = m.get("image_paths") or []
            m["image_count"] = len(ips) if isinstance(ips, list) else 0
            del m["image_paths"]

    metas = sanitize_metas(metas)

    # 4) 임베딩/DB 연결
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    # 5) 파일(source) 단위 업데이트:
    #    - 기존 source_fp와 같으면 스킵
    #    - 다르면 해당 source 문서들 delete 후 재적재
    col = vectordb._collection

    sources = sorted({m.get("source") for m in metas if m.get("source")})

    added_total = 0
    updated_sources = 0
    skipped_sources = 0

    for src in sources:
        # 이번 실행 fingerprint
        new_fp = None
        for m in metas:
            if m.get("source") == src:
                new_fp = m.get("source_fp")
                break

        # DB fingerprint(1개만)
        old_fp = None
        try:
            got = col.get(where={"source": src}, include=["metadatas"], limit=1)
            ms = got.get("metadatas") or []
            if ms:
                old_fp = ms[0].get("source_fp")
        except Exception:
            old_fp = None

        # 변경 없으면 스킵
        if old_fp is not None and new_fp is not None and old_fp == new_fp:
            skipped_sources += 1
            continue

        # 변경/신규면 기존 source 삭제 후 재적재
        try:
            col.delete(where={"source": src})
        except Exception:
            pass

        src_texts, src_metas, src_ids = [], [], []
        for t, m, i in zip(texts, metas, ids):
            if m.get("source") == src:
                src_texts.append(t)
                src_metas.append(m)
                src_ids.append(i)

        if src_ids:
            vectordb.add_texts(texts=src_texts, metadatas=src_metas, ids=src_ids)
            added_total += len(src_ids)
            updated_sources += 1

    print(
        f"OK raw_docs={len(docs)} indexed_chunks={len(chunks)} "
        f"sources={len(sources)} updated_or_new_sources={updated_sources} "
        f"skipped_unchanged_sources={skipped_sources} added_chunks={added_total} "
        f"chroma_dir={CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()
