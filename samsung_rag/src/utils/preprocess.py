"""
src/utils/preprocess.py
─────────────────────────────────────────────────────────────────────────────
텍스트 전처리 유틸리티.

처리 순서:
    원본 텍스트
    → hanja 라이브러리로 한자 → 한글 변환
    → kiwipiepy 형태소 분석 (BM25/tsvector용 명사·동사 추출)
    → 정규화 (공백·특수문자)
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ─── hanja 로드 ───────────────────────────────────────────────────────────────
try:
    import hanja  # pip install hanja

    _HANJA_AVAILABLE = True
except ImportError:
    logger.warning("hanja 패키지 미설치 → 한자 변환 건너뜀")
    _HANJA_AVAILABLE = False

# ─── Kiwi 로드 ────────────────────────────────────────────────────────────────
try:
    from kiwipiepy import Kiwi  # pip install kiwipiepy

    _KIWI: Optional[Kiwi] = None  # 지연 초기화 (무거운 모델)
    _KIWI_AVAILABLE = True
except ImportError:
    logger.warning("kiwipiepy 미설치 → 형태소 분석 건너뜀")
    _KIWI_AVAILABLE = False
    _KIWI = None


def _get_kiwi() -> "Kiwi | None":
    """Kiwi 인스턴스 싱글턴 반환 (지연 초기화)."""
    global _KIWI
    if not _KIWI_AVAILABLE:
        return None
    if _KIWI is None:
        logger.info("Kiwi 형태소 분석기 초기화 중...")
        from kiwipiepy import Kiwi

        _KIWI = Kiwi()
        logger.info("Kiwi 초기화 완료")
    return _KIWI


# ─── BM25 검색 대상 품사 ───────────────────────────────────────────────────────
# NNG: 일반명사, NNP: 고유명사, VV: 동사, VA: 형용사, SL: 외국어
_TARGET_POS = {"NNG", "NNP", "VV", "VA", "SL"}


def hanja_to_hangul(text: str) -> str:
    """
    hanja 라이브러리를 사용해 한자를 한글로 변환한다.

    Args:
        text: 원본 텍스트 (한자 포함 가능)

    Returns:
        한자가 한글로 치환된 텍스트.
        hanja 미설치 시 원본 반환.
    """
    if not _HANJA_AVAILABLE or not text:
        return text
    try:
        return hanja.translate(text, "substitution")
    except Exception as e:
        logger.warning("hanja 변환 실패: %s", e)
        return text


def extract_morphemes(text: str) -> list[str]:
    """
    kiwipiepy로 형태소 분석 후 명사·동사 원형 리스트를 반환한다.
    tsvector 생성 및 BM25 인덱싱에 사용.

    Args:
        text: 분석할 텍스트

    Returns:
        명사·동사 원형 토큰 리스트.
        kiwipiepy 미설치 시 공백 분리 토큰 반환.
    """
    if not text:
        return []

    kiwi = _get_kiwi()
    if kiwi is None:
        return text.split()

    try:
        result = kiwi.analyze(text)
        if not result:
            return text.split()
        tokens = result[0][0]  # 최고 점수 분석 결과
        return [
            token.form
            for token in tokens
            if token.tag in _TARGET_POS and len(token.form) > 1
        ]
    except Exception as e:
        logger.warning("Kiwi 형태소 분석 실패: %s", e)
        return text.split()


def normalize_whitespace(text: str) -> str:
    """연속 공백·탭·개행을 단일 공백으로 정규화한다."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    제어 문자, 불필요한 특수문자를 제거하고 공백을 정규화한다.

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    # 제어 문자 제거 (탭·개행 제외)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 중복 구두점 정리
    text = re.sub(r"\.{4,}", "...", text)
    return normalize_whitespace(text)


def preprocess_for_embed(text: str) -> str:
    """
    임베딩 전 전체 전처리 파이프라인.

    순서:
        1. hanja → 한글
        2. 정제 (clean_text)

    Args:
        text: 원본 청크 텍스트

    Returns:
        임베딩용 정제 텍스트
    """
    text = hanja_to_hangul(text)
    text = clean_text(text)
    return text


def preprocess_for_bm25(text: str) -> str:
    """
    BM25 / tsvector 검색용 전처리.

    순서:
        1. hanja → 한글
        2. 형태소 분석 → 명사·동사 추출
        3. 공백 결합

    Args:
        text: 원본 텍스트

    Returns:
        형태소 추출 후 공백 결합 문자열
    """
    text = hanja_to_hangul(text)
    tokens = extract_morphemes(text)
    return " ".join(tokens) if tokens else text


def build_embed_text(
    content: str,
    source: str,
    doc_type: str,
    category: str = "",
    slide_title: str = "",
) -> str:
    """
    embed_text prefix 포맷을 doc_type에 따라 생성한다.

    - PDF/DOCX: "[문서] {파일명} | {카테고리}\n{본문}"
    - PPTX:     "[TITLE] {슬라이드제목}\n{본문}"
    - XLSX/CSV: "[문서] {파일명}\n{본문}"

    Args:
        content:     청크 본문
        source:      파일 경로 or 파일명
        doc_type:    pdf, docx, pptx, xlsx, csv, txt
        category:    카테고리명 (옵션)
        slide_title: PPTX 슬라이드 제목 (옵션)

    Returns:
        prefix가 붙은 임베딩용 텍스트
    """
    filename = source.split("/")[-1].split("\\")[-1]
    doc_type_lower = doc_type.lower()

    if doc_type_lower == "pptx":
        title = slide_title or filename
        prefix = f"[TITLE] {title}"
    elif doc_type_lower in ("xlsx", "csv", "xls"):
        prefix = f"[문서] {filename}"
    else:
        # pdf, docx, txt, doc 등
        cat_part = f" | {category}" if category else ""
        prefix = f"[문서] {filename}{cat_part}"

    return f"{prefix}\n{content}"
