"""
src/pipeline/ingest.py
─────────────────────────────────────────────────────────────────────────────
문서 수집 → 파싱 → 청킹 → 임베딩 → PostgreSQL 저장 파이프라인.

흐름:
    1. 파일 경로 목록 수집 (로컬 data/sample/ 폴더)
       [주석] ECM 엑셀 다운로드 방식
    2. fingerprint 계산 → 변경 없으면 스킵
    3. Docling 파싱
    4. hanja 전처리 후 청킹
    5. OpenAI 임베딩
       [주석] Ollama bge-m3 전환 코드
    6. PostgreSQL 저장 (document + chunks)
    7. 이미지 메타데이터 저장
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PIPELINE_VERSION: str = os.getenv("PIPELINE_VERSION", "v1")
SAMPLE_DIR: Path = Path(os.getenv("SAMPLE_DIR", "data/sample"))


# ─────────────────────────────────────────────────────────────────────────────
# 임베딩 클라이언트
# ─────────────────────────────────────────────────────────────────────────────

def _get_openai_embed_client():
    """OpenAI 임베딩 클라이언트 반환."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    텍스트 리스트를 임베딩 벡터 리스트로 변환한다.

    개발: OpenAI text-embedding-3-small (dim=1536)
    운영 전환 시 아래 주석 해제:

    # ── Ollama bge-m3 전환 코드 ──────────────────────────────────────────────
    # import requests
    # OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    # vectors = []
    # for text in texts:
    #     resp = requests.post(
    #         f"{OLLAMA_URL}/api/embeddings",
    #         json={"model": MODEL, "prompt": text},
    #         timeout=30,
    #     )
    #     resp.raise_for_status()
    #     vectors.append(resp.json()["embedding"])
    # return vectors
    # ────────────────────────────────────────────────────────────────────────

    Args:
        texts: 임베딩할 텍스트 목록

    Returns:
        float 벡터 리스트
    """
    if not texts:
        return []

    client = _get_openai_embed_client()
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error("임베딩 실패: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 청킹
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_markdown(
    markdown: str,
    source: str,
    doc_type: str,
    category: str = "",
    page: int = 0,
) -> list[dict]:
    """
    PDF/DOCX 마크다운을 헤딩 단위로 청킹한다.

    조항 구조(제1조, 1안 등) → MarkdownHeaderTextSplitter
    너무 짧은 청크(<100자) → RecursiveCharacterTextSplitter 후처리 (400자, overlap 80자)

    Args:
        markdown: Docling 변환 마크다운
        source:   파일 소스 경로
        doc_type: 문서 유형
        category: 카테고리명
        page:     시작 페이지 (메타데이터용)

    Returns:
        청크 딕셔너리 리스트
    """
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from src.utils.preprocess import build_embed_text, preprocess_for_embed

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    try:
        header_chunks = md_splitter.split_text(markdown)
    except Exception as e:
        logger.warning("MarkdownHeaderSplitter 실패 → 단순 분할: %s", e)
        header_chunks = []

    # 너무 짧거나 header 분할 실패 시 RecursiveCharacterTextSplitter 사용
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""],
    )

    raw_chunks: list[str] = []
    if header_chunks:
        for hc in header_chunks:
            text = hc.page_content.strip()
            if not text:
                continue
            if len(text) < 100:
                # 너무 짧은 청크는 더 잘게 나누지 않고 그대로 유지
                raw_chunks.append(text)
            elif len(text) > 400:
                sub_chunks = char_splitter.split_text(text)
                raw_chunks.extend(sub_chunks)
            else:
                raw_chunks.append(text)
    else:
        raw_chunks = char_splitter.split_text(markdown)

    results: list[dict] = []
    for content in raw_chunks:
        content = content.strip()
        if not content:
            continue
        content_clean = preprocess_for_embed(content)
        embed_text = build_embed_text(content_clean, source, doc_type, category)
        chunk_id = _make_chunk_id(source, doc_type, page, None, None, embed_text)
        results.append(
            {
                "chunk_id": chunk_id,
                "content": content_clean,
                "embed_text": embed_text,
                "doc_type": doc_type,
                "page": page,
                "slide": None,
                "sheet": None,
                "source": source,
                "metadata": {"category": category},
            }
        )
    return results


def _chunk_slides(slides, source: str) -> list[dict]:
    """
    PPTX 슬라이드 단위 청크 생성.

    Args:
        slides: ParsedSlide 리스트
        source: 파일 소스

    Returns:
        청크 딕셔너리 리스트
    """
    from src.utils.preprocess import build_embed_text, preprocess_for_embed

    results: list[dict] = []
    for slide in slides:
        content = f"{slide.title}\n{slide.content}".strip()
        if not content:
            continue
        content_clean = preprocess_for_embed(content)
        embed_text = build_embed_text(
            content_clean,
            source,
            "pptx",
            slide_title=slide.title,
        )
        chunk_id = _make_chunk_id(source, "pptx", None, slide.slide_num, None, embed_text)
        results.append(
            {
                "chunk_id": chunk_id,
                "content": content_clean,
                "embed_text": embed_text,
                "doc_type": "pptx",
                "page": None,
                "slide": slide.slide_num,
                "sheet": None,
                "source": source,
                "metadata": {"slide_title": slide.title},
            }
        )
    return results


def _chunk_sheets(
    sheets,
    source: str,
    filename: str,
) -> list[dict]:
    """
    XLSX 시트 단위 or 행 단위 청크 생성.

    조건:
        - 행 수 50 미만: 시트 단위
        - 파일명에 '신청서' or '양식' 포함: 시트 단위
        - 나머지: RecursiveCharacterTextSplitter (400자, overlap 80자) 분할

    Args:
        sheets:   ParsedSheet 리스트
        source:   파일 소스
        filename: 원본 파일명

    Returns:
        청크 딕셔너리 리스트
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from src.utils.preprocess import build_embed_text, preprocess_for_embed

    results: list[dict] = []
    is_form = any(kw in filename for kw in ("신청서", "양식"))
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80
    )

    for sheet in sheets:
        if sheet.row_count < 50 or is_form:
            # 시트 단위
            content_clean = preprocess_for_embed(sheet.content)
            embed_text = build_embed_text(content_clean, source, "xlsx")
            chunk_id = _make_chunk_id(source, "xlsx", None, None, sheet.sheet_name, embed_text)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "content": content_clean,
                    "embed_text": embed_text,
                    "doc_type": "xlsx",
                    "page": None,
                    "slide": None,
                    "sheet": sheet.sheet_name,
                    "source": source,
                    "metadata": {"sheet_name": sheet.sheet_name},
                }
            )
        else:
            sub_chunks = char_splitter.split_text(sheet.content)
            for sub in sub_chunks:
                sub_clean = preprocess_for_embed(sub)
                if not sub_clean:
                    continue
                embed_text = build_embed_text(sub_clean, source, "xlsx")
                chunk_id = _make_chunk_id(source, "xlsx", None, None, sheet.sheet_name, embed_text)
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "content": sub_clean,
                        "embed_text": embed_text,
                        "doc_type": "xlsx",
                        "page": None,
                        "slide": None,
                        "sheet": sheet.sheet_name,
                        "source": source,
                        "metadata": {"sheet_name": sheet.sheet_name},
                    }
                )
    return results


def _make_chunk_id(
    source: str,
    doc_type: str,
    page: Optional[int],
    slide: Optional[int],
    sheet: Optional[str],
    embed_text: str,
) -> str:
    """
    청크 고유 ID (SHA1) 생성.

    포맷: sha1("{source}|{doc_type}|{page}|{slide}|{sheet}|{len}|{prefix120}")
    """
    raw = f"{source}|{doc_type}|{page}|{slide}|{sheet}|{len(embed_text)}|{embed_text[:120]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fingerprint(file_path: Path) -> str:
    """
    PIPELINE_VERSION + 파일 내용 SHA1로 fingerprint 생성.

    Args:
        file_path: 대상 파일 경로

    Returns:
        40자 hex 문자열
    """
    h = hashlib.sha1()
    h.update(PIPELINE_VERSION.encode())
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 파일 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_local_files(directory: Path | None = None) -> list[Path]:
    """
    로컬 디렉터리에서 처리 가능한 파일 목록을 수집한다.

    지원: pdf, pptx, ppt, xlsx, xls, docx, doc, txt, csv

    # ── ECM 엑셀 다운로드 방식 (회사 실제 환경) ─────────────────────────────
    # import pandas as pd
    # EXCEL_PATH = Path("data/ecm_list.xlsx")
    # DOWNLOAD_URL = "http://kone.samsungcorp.com:8022/jsp/download.jsp?r_object_id={r_object_id}"
    # TARGET_CATEGORY = "50. 영업관련"
    #
    # def collect_from_ecm(excel_path: Path) -> list[Path]:
    #     df = pd.read_excel(excel_path)
    #     target = df[df["CATEGORY_ID"].str.contains(TARGET_CATEGORY, na=False)]
    #     saved: list[Path] = []
    #     for _, row in target.iterrows():
    #         r_obj_id = row["R_OBJECT_ID"]
    #         org_file = row["ORG_FILE"]
    #         url = DOWNLOAD_URL.format(r_object_id=r_obj_id)
    #         dest = Path("data/ecm") / org_file
    #         dest.parent.mkdir(parents=True, exist_ok=True)
    #         import requests
    #         resp = requests.get(url, timeout=60)
    #         resp.raise_for_status()
    #         dest.write_bytes(resp.content)
    #         saved.append(dest)
    #     return saved
    # ────────────────────────────────────────────────────────────────────────

    Args:
        directory: 검색 디렉터리 (기본: SAMPLE_DIR)

    Returns:
        파일 경로 리스트
    """
    search_dir = directory or SAMPLE_DIR
    supported_exts = {".pdf", ".pptx", ".ppt", ".xlsx", ".xls", ".docx", ".doc", ".txt", ".csv"}
    files = [
        p for p in sorted(search_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in supported_exts
    ]
    logger.info("수집된 파일 수: %d (from %s)", len(files), search_dir)
    return files


def _extract_metadata_from_filename(file_path: Path) -> dict:
    """
    파일명에서 메타데이터를 추출한다.

    예: "50_영업관련_출장비규정_v2.pdf" → category="영업관련", doc_type="pdf"

    Args:
        file_path: 파일 경로

    Returns:
        메타데이터 딕셔너리
    """
    stem = file_path.stem
    parts = stem.split("_")
    category = parts[1] if len(parts) >= 2 else ""
    return {
        "filename": file_path.name,
        "category": category,
        "doc_type": file_path.suffix.lower().lstrip("."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DB 저장
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_document(
    conn,
    source: str,
    doc_type: str,
    category: str,
    fingerprint: str,
) -> int:
    """
    documents 테이블에 문서를 등록한다.

    - 동일 fingerprint 존재 시 스킵 (document_id 반환)
    - source 동일 + fingerprint 다른 경우: 기존 is_latest=FALSE → 신규 삽입

    Args:
        conn:        psycopg2 연결
        source:      파일 소스 경로
        doc_type:    문서 유형
        category:    카테고리
        fingerprint: fingerprint 해시

    Returns:
        document_id (int)
    """
    import psycopg2.extras

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 동일 fingerprint 이미 존재 → 스킵
        cur.execute(
            "SELECT id FROM documents WHERE fingerprint = %s",
            (fingerprint,),
        )
        row = cur.fetchone()
        if row:
            logger.debug("fingerprint 동일 → 스킵: %s", source)
            return row["id"]

        # 기존 버전 is_latest = FALSE
        cur.execute(
            "UPDATE documents SET is_latest = FALSE, updated_at = NOW() WHERE source = %s AND is_latest = TRUE",
            (source,),
        )

        # 신규 삽입
        cur.execute(
            """
            INSERT INTO documents (source, doc_type, category, fingerprint, pipeline_version, is_latest)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            RETURNING id
            """,
            (source, doc_type, category, fingerprint, PIPELINE_VERSION),
        )
        new_row = cur.fetchone()
        doc_id: int = new_row["id"]
        logger.info("문서 등록: id=%d source=%s", doc_id, source)
        return doc_id


def _insert_chunks(conn, document_id: int, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    청크와 임베딩을 chunks 테이블에 삽입한다.
    chunk_id 중복 시 ON CONFLICT DO NOTHING으로 스킵.

    Args:
        conn:        psycopg2 연결
        document_id: 대상 문서 ID
        chunks:      청크 딕셔너리 리스트
        embeddings:  벡터 리스트 (chunks와 동일 순서)
    """
    import json
    import psycopg2.extras

    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append(
            (
                chunk["chunk_id"],
                document_id,
                chunk["content"],
                chunk["embed_text"],
                str(emb),           # pgvector는 문자열 "[0.1, 0.2, ...]" 허용
                json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                chunk["doc_type"],
                chunk.get("page"),
                chunk.get("slide"),
                chunk.get("sheet"),
                chunk["source"],
            )
        )

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks
                (chunk_id, document_id, content, embed_text, embedding,
                 metadata, doc_type, page, slide, sheet, source)
            VALUES %s
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            records,
            template="""(
                %s, %s, %s, %s, %s::vector,
                %s::jsonb, %s, %s, %s, %s, %s
            )""",
        )
    logger.info("청크 %d개 저장 완료 (document_id=%d)", len(records), document_id)


def _insert_images(conn, document_id: int, image_paths: list[str]) -> None:
    """
    이미지 메타데이터를 document_images 테이블에 저장한다.

    Args:
        conn:        psycopg2 연결
        document_id: 대상 문서 ID
        image_paths: 이미지 파일 경로 리스트
    """
    import hashlib
    from src.parsers.docling_parser import is_logo as check_is_logo

    if not image_paths:
        return

    records = []
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            img_hash = ""
        logo = check_is_logo(img_hash)
        records.append((document_id, img_path, img_hash, logo, None))

    with conn.cursor() as cur:
        import psycopg2.extras
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO document_images (document_id, image_path, image_hash, is_logo, page)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            records,
        )
    logger.info("이미지 %d개 저장 완료 (document_id=%d)", len(records), document_id)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(file_path: str | Path, category: str = "") -> dict:
    """
    단일 파일을 수집·파싱·청킹·임베딩·저장한다.

    Args:
        file_path: 처리할 파일 경로
        category:  카테고리명 (파일명에서 추출 or 직접 지정)

    Returns:
        결과 요약 딕셔너리
        {
            "source": str,
            "status": "skipped" | "indexed",
            "chunks": int,
            "document_id": int
        }
    """
    from src.db import get_connection
    from src.parsers.docling_parser import parse_file

    path = Path(file_path)
    source = str(path)
    meta = _extract_metadata_from_filename(path)
    if not category:
        category = meta["category"]
    doc_type = meta["doc_type"]

    # fingerprint 계산
    fingerprint = _compute_fingerprint(path)

    # DB에서 중복 확인
    conn = get_connection()
    try:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM documents WHERE fingerprint = %s",
                (fingerprint,),
            )
            existing = cur.fetchone()
        if existing:
            logger.info("변경 없음 → 스킵: %s", path.name)
            return {
                "source": source,
                "status": "skipped",
                "chunks": 0,
                "document_id": existing["id"],
            }
    finally:
        conn.close()

    # 파싱
    logger.info("파싱 시작: %s", path.name)
    parsed = parse_file(path, source=source)

    # 청킹
    chunks: list[dict] = []
    if parsed.slides:
        chunks = _chunk_slides(parsed.slides, source)
    elif parsed.sheets:
        chunks = _chunk_sheets(parsed.sheets, source, path.name)
    elif parsed.markdown:
        chunks = _chunk_markdown(
            parsed.markdown, source, doc_type, category
        )
    else:
        logger.warning("파싱 결과 없음: %s", path.name)
        return {"source": source, "status": "empty", "chunks": 0, "document_id": -1}

    if not chunks:
        logger.warning("청크 생성 결과 없음: %s", path.name)
        return {"source": source, "status": "empty", "chunks": 0, "document_id": -1}

    # 임베딩
    logger.info("임베딩 생성: %d 청크", len(chunks))
    embed_texts_list = [c["embed_text"] for c in chunks]

    # 배치 처리 (OpenAI 최대 2048 inputs)
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(embed_texts_list), BATCH_SIZE):
        batch = embed_texts_list[i : i + BATCH_SIZE]
        all_embeddings.extend(embed_texts(batch))

    # DB 저장
    conn = get_connection()
    try:
        doc_id = _upsert_document(conn, source, doc_type, category, fingerprint)
        _insert_chunks(conn, doc_id, chunks, all_embeddings)
        _insert_images(conn, doc_id, parsed.image_paths)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("DB 저장 실패 (%s): %s", path.name, e)
        raise
    finally:
        conn.close()

    logger.info("인덱싱 완료: %s (chunks=%d)", path.name, len(chunks))
    return {
        "source": source,
        "status": "indexed",
        "chunks": len(chunks),
        "document_id": doc_id,
    }


def ingest_directory(directory: Path | None = None) -> list[dict]:
    """
    디렉터리 내 모든 지원 파일을 순서대로 인덱싱한다.

    Args:
        directory: 검색 디렉터리 (기본: SAMPLE_DIR)

    Returns:
        각 파일의 ingest_file 결과 리스트
    """
    files = collect_local_files(directory)
    results: list[dict] = []
    for f in files:
        try:
            result = ingest_file(f)
            results.append(result)
        except Exception as e:
            logger.error("파일 인덱싱 실패 (%s): %s", f.name, e)
            results.append({"source": str(f), "status": "error", "error": str(e)})
    return results
