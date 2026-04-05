"""
src/parsers/docling_parser.py
─────────────────────────────────────────────────────────────────────────────
Docling 기반 문서 파싱 모듈.

지원 형식:
    - PDF, PPTX, XLSX, DOCX: Docling 직접 파싱
    - doc, ppt, xls (레거시): LibreOffice로 변환 후 Docling 파싱

이미지 처리:
    - Docling 파싱 시 이미지 추출 → data/extracted_images/ 저장
    - 동일 해시 3회 이상 → is_logo=TRUE (로고 감지)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "data/extracted_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 이미지 해시 → 발생 횟수 추적 (로고 감지용)
_image_hash_counter: dict[str, int] = {}

# Docling 지원 확장자
_DOCLING_NATIVE = {".pdf", ".pptx", ".xlsx", ".docx"}
# LibreOffice 변환이 필요한 레거시 확장자
_LEGACY_EXTS = {".doc", ".ppt", ".xls"}


@dataclass
class ParsedSlide:
    """PPTX 슬라이드 단위 파싱 결과."""

    slide_num: int
    title: str
    content: str


@dataclass
class ParsedSheet:
    """XLSX 시트 단위 파싱 결과."""

    sheet_name: str
    content: str
    row_count: int


@dataclass
class ParsedDocument:
    """파싱 완료 문서 컨테이너."""

    source: str
    doc_type: str
    markdown: str = ""              # PDF/DOCX 마크다운 전체
    slides: list[ParsedSlide] = field(default_factory=list)   # PPTX
    sheets: list[ParsedSheet] = field(default_factory=list)   # XLSX
    image_paths: list[str] = field(default_factory=list)      # 추출된 이미지 경로
    raw_text: str = ""              # 폴백용 순수 텍스트


def _hash_file(path: Path) -> str:
    """파일 SHA1 해시를 반환한다."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _convert_legacy_with_libreoffice(src: Path, tmp_dir: Path) -> Optional[Path]:
    """
    레거시 Office 파일을 LibreOffice로 변환한다.

    Args:
        src:     원본 파일 경로 (.doc / .ppt / .xls)
        tmp_dir: 변환 결과를 저장할 임시 디렉터리

    Returns:
        변환된 파일 경로. 실패 시 None.
    """
    ext_map = {".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx"}
    target_ext = ext_map.get(src.suffix.lower())
    if not target_ext:
        return None

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        logger.error("LibreOffice(soffice)가 설치되어 있지 않습니다.")
        return None

    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", target_ext[1:], str(src),
             "--outdir", str(tmp_dir)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        converted = tmp_dir / (src.stem + target_ext)
        if converted.exists():
            logger.info("LibreOffice 변환 성공: %s → %s", src.name, converted.name)
            return converted
        logger.error("LibreOffice 변환 후 파일 없음: %s", converted)
        return None
    except subprocess.TimeoutExpired:
        logger.error("LibreOffice 변환 타임아웃: %s", src.name)
        return None
    except subprocess.CalledProcessError as e:
        logger.error("LibreOffice 변환 오류: %s\n%s", src.name, e.stderr.decode())
        return None


def _extract_images_from_docling(doc_result, source_name: str) -> list[str]:
    """
    Docling 파싱 결과에서 이미지를 추출하고 저장한다.

    Args:
        doc_result:  Docling DocumentConverter 결과 객체
        source_name: 문서 식별자 (파일명)

    Returns:
        저장된 이미지 파일 경로 리스트
    """
    saved_paths: list[str] = []
    try:
        # Docling 이미지 추출 API (버전에 따라 다를 수 있음)
        if not hasattr(doc_result, "document"):
            return saved_paths

        pictures = getattr(doc_result.document, "pictures", [])
        for idx, pic in enumerate(pictures):
            try:
                img_data = pic.image.pil_image if hasattr(pic, "image") else None
                if img_data is None:
                    continue

                # 이미지 해시
                import io
                buf = io.BytesIO()
                img_data.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                img_hash = hashlib.md5(img_bytes).hexdigest()

                # 로고 감지 카운터
                _image_hash_counter[img_hash] = _image_hash_counter.get(img_hash, 0) + 1

                # 저장 경로
                safe_name = source_name.replace("/", "_").replace("\\", "_")
                img_path = IMAGES_DIR / f"{safe_name}_{idx}_{img_hash[:8]}.png"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(str(img_path))
                logger.debug("이미지 저장: %s (hash=%s)", img_path.name, img_hash)
            except Exception as e:
                logger.warning("이미지 추출 실패 (index=%d): %s", idx, e)
    except Exception as e:
        logger.warning("이미지 전체 추출 실패: %s", e)
    return saved_paths


def is_logo(image_hash: str) -> bool:
    """동일 해시가 3회 이상 등장하면 로고로 판단한다."""
    return _image_hash_counter.get(image_hash, 0) >= 3


def _parse_with_docling(file_path: Path, source: str) -> ParsedDocument:
    """
    Docling DocumentConverter로 파일을 파싱한다.

    Args:
        file_path: 실제 파일 경로
        source:    원본 소스 식별자

    Returns:
        ParsedDocument
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        logger.error("docling 미설치. pip install docling 실행 필요.")
        raise

    ext = file_path.suffix.lower()
    doc_type = ext.lstrip(".")

    converter = DocumentConverter()
    try:
        result = converter.convert(str(file_path))
    except Exception as e:
        logger.error("Docling 변환 실패 (%s): %s", file_path.name, e)
        raise

    image_paths = _extract_images_from_docling(result, file_path.stem)

    doc = ParsedDocument(source=source, doc_type=doc_type, image_paths=image_paths)

    if ext in (".pdf", ".docx", ".doc"):
        # 마크다운으로 내보내기
        try:
            doc.markdown = result.document.export_to_markdown()
        except Exception:
            doc.markdown = result.document.export_to_text()
        doc.raw_text = result.document.export_to_text()

    elif ext == ".pptx":
        # 슬라이드 단위 추출
        try:
            pages = result.document.pages
            for page_num, page in enumerate(pages, start=1):
                # Docling 페이지에서 텍스트 추출
                title = ""
                texts: list[str] = []
                for item in page.cells if hasattr(page, "cells") else []:
                    text = getattr(item, "text", "") or ""
                    if not text.strip():
                        continue
                    if not title:
                        title = text.strip()
                    else:
                        texts.append(text.strip())
                doc.slides.append(
                    ParsedSlide(
                        slide_num=page_num,
                        title=title,
                        content="\n".join(texts),
                    )
                )
            if not doc.slides:
                # 폴백: 전체 마크다운
                doc.markdown = result.document.export_to_markdown()
        except Exception as e:
            logger.warning("PPTX 슬라이드 추출 실패, 마크다운 폴백: %s", e)
            doc.markdown = result.document.export_to_markdown()

    elif ext in (".xlsx", ".xls", ".csv"):
        # 시트 단위 추출
        try:
            for table in result.document.tables if hasattr(result.document, "tables") else []:
                sheet_name = getattr(table, "label", "Sheet")
                rows = getattr(table, "data", [])
                row_count = len(rows)
                # 표 → 마크다운 변환
                md_rows: list[str] = []
                for i, row in enumerate(rows):
                    cells = [str(c) for c in row]
                    md_rows.append("| " + " | ".join(cells) + " |")
                    if i == 0:
                        md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
                doc.sheets.append(
                    ParsedSheet(
                        sheet_name=str(sheet_name),
                        content="\n".join(md_rows),
                        row_count=row_count,
                    )
                )
            if not doc.sheets:
                doc.markdown = result.document.export_to_markdown()
        except Exception as e:
            logger.warning("XLSX 시트 추출 실패, 마크다운 폴백: %s", e)
            doc.markdown = result.document.export_to_markdown()

    else:
        doc.markdown = result.document.export_to_markdown()
        doc.raw_text = result.document.export_to_text()

    return doc


def _parse_txt_fallback(file_path: Path, source: str) -> ParsedDocument:
    """
    .txt 파일을 직접 읽어 ParsedDocument로 반환한다.
    Docling 파싱 흐름을 유지하기 위해 markdown 필드에 내용을 담는다.

    Args:
        file_path: txt 파일 경로
        source:    소스 식별자

    Returns:
        ParsedDocument (doc_type='txt')
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp949") as f:
            text = f.read()

    return ParsedDocument(
        source=source,
        doc_type="txt",
        markdown=text,
        raw_text=text,
    )


def parse_file(file_path: str | Path, source: Optional[str] = None) -> ParsedDocument:
    """
    파일을 파싱하여 ParsedDocument를 반환하는 메인 진입점.

    처리 흐름:
        1. 레거시 파일 → LibreOffice 변환 → Docling
        2. 네이티브 파일 → Docling 직접 파싱
        3. .txt → 직접 읽기

    Args:
        file_path: 파싱할 파일 경로
        source:    메타데이터용 소스 식별자 (기본: 파일 경로 문자열)

    Returns:
        ParsedDocument
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    _source = source or str(path)
    ext = path.suffix.lower()

    logger.info("파일 파싱 시작: %s (type=%s)", path.name, ext)

    # txt 폴백
    if ext == ".txt":
        doc = _parse_txt_fallback(path, _source)
        logger.info("TXT 파싱 완료: %s", path.name)
        return doc

    # 레거시 변환
    if ext in _LEGACY_EXTS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            converted = _convert_legacy_with_libreoffice(path, tmp_dir)
            if converted is None:
                raise RuntimeError(f"레거시 파일 변환 실패: {path.name}")
            doc = _parse_with_docling(converted, _source)
        logger.info("레거시 변환+파싱 완료: %s", path.name)
        return doc

    # Docling 네이티브
    if ext in _DOCLING_NATIVE:
        doc = _parse_with_docling(path, _source)
        logger.info("Docling 파싱 완료: %s", path.name)
        return doc

    # 알 수 없는 확장자 → 텍스트 폴백
    logger.warning("미지원 확장자 %s → 텍스트 폴백 시도", ext)
    try:
        doc = _parse_txt_fallback(path, _source)
        return doc
    except Exception as e:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}") from e
