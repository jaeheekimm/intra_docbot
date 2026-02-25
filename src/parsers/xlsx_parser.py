import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from langchain_core.documents import Document

from src.utils.files import safe_filename
from src.utils.hashing import sha1_short

logger = logging.getLogger(__name__)


def _guess_image_ext(img) -> str:
    """
    openpyxl Image 객체에서 확장자 추정.
    케이스가 다양해서 '추정'만 하고, 실패하면 bin.
    """
    # 1) format / _format 같은 속성이 있는 경우
    fmt = getattr(img, "format", None) or getattr(img, "_format", None)
    if isinstance(fmt, str) and fmt.strip():
        return fmt.strip().lower().lstrip(".")

    # 2) path에 확장자가 들어있는 경우
    path = getattr(img, "path", None)
    if isinstance(path, str) and "." in os.path.basename(path):
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext:
            return ext

    # 3) 기본값
    return "bin"


def extract_xlsx_images(xlsx_path: str, out_dir: str) -> Dict[str, List[str]]:
    """
    return: {sheet_name: [image_path, ...]}
    """
    mapping: Dict[str, List[str]] = {}
    wb = load_workbook(xlsx_path, data_only=True)
    base = safe_filename(os.path.basename(xlsx_path))

    for ws in wb.worksheets:
        imgs = getattr(ws, "_images", []) or []
        if not imgs:
            continue

        for i, img in enumerate(imgs, start=1):
            try:
                blob = img._data()
            except Exception as e:
                logger.warning(
                    "Failed to extract image: file=%s sheet=%s idx=%s err=%s",
                    xlsx_path,
                    ws.title,
                    i,
                    repr(e),
                )
                continue

            if not blob:
                logger.warning(
                    "Empty image blob: file=%s sheet=%s idx=%s",
                    xlsx_path,
                    ws.title,
                    i,
                )
                continue

            hid = sha1_short(blob)
            ext = _guess_image_ext(img)
            fname = f"{base}__sheet{safe_filename(ws.title)}__img{i:03d}__{hid}.{ext}"
            save_path = os.path.join(out_dir, fname)

            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(blob)

            mapping.setdefault(ws.title, []).append(save_path)

    wb.close()
    return mapping


def _pick_header_row(ws, max_scan_rows: int = 20) -> Tuple[int, List[str]]:
    """
    앞쪽에서 '헤더로 쓸만한 첫 행'을 찾는다.
    - 완전 공백 행은 스킵
    - 헤더 셀이 공백이면 A,B,C... 컬럼레터로 대체
    return: (header_row_index(1-based), headers)
    """
    for r_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if r_idx > max_scan_rows:
            break

        vals = ["" if v is None else str(v).strip() for v in row]
        if all(v == "" for v in vals):
            continue

        headers: List[str] = []
        for c_idx, v in enumerate(vals, start=1):
            h = v if v != "" else get_column_letter(c_idx)
            headers.append(h)
        return r_idx, headers

    return 0, []


def _row_to_text_with_headers(headers: List[str], values: Tuple[Any, ...]) -> str:
    """
    헤더 기반: "헤더: 값 | 헤더: 값" 형태
    """
    parts: List[str] = []
    for i, v in enumerate(values):
        if i >= len(headers):
            break
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f"{headers[i]}: {s}")
    return " | ".join(parts).strip()


def load_xlsx_docs_rows(
    xlsx_path: str,
    xlsx_img_map: Dict[str, List[str]],
    *,
    attach_sheet_images: bool = False,  # ★ 기본 False 권장
    max_rows_per_sheet: Optional[int] = None,
    header_scan_rows: int = 20,
) -> List[Document]:
    """
    XLSX는 '행 단위 Document'로 생성.
    - 기본은 헤더 기반 텍스트("컬럼: 값")로 변환해서 검색 품질 개선
    - 이미지: 시트 단위 매핑만 가능하므로 기본은 attach_sheet_images=False 권장
    """
    wb = load_workbook(xlsx_path, data_only=True)
    file_name = os.path.basename(xlsx_path)

    docs: List[Document] = []
    for ws in wb.worksheets:
        sheet_images = xlsx_img_map.get(ws.title, [])
        image_count = len(sheet_images)

        header_row_idx, headers = _pick_header_row(ws, max_scan_rows=header_scan_rows)

        row_count = 0
        for r_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            # 헤더 행은 본문에서 제외
            if header_row_idx and r_idx == header_row_idx:
                continue

            if headers:
                text = _row_to_text_with_headers(headers, row)
            else:
                # 헤더를 못 찾았으면 기존 방식 fallback
                vals = ["" if v is None else str(v).strip() for v in row]
                if all(v == "" for v in vals):
                    continue
                text = " | ".join([v for v in vals if v != ""]).strip()

            if not text:
                continue

            md = {
                "source": xlsx_path,
                "file_name": file_name,
                "doc_type": "xlsx",
                "page": None,
                "slide": None,
                "sheet": ws.title,
                "row": r_idx,
                "image_count": image_count,  # ★ 판단용
                "image_paths": (sheet_images if attach_sheet_images else []),
            }
            docs.append(Document(page_content=text, metadata=md))

            row_count += 1
            if max_rows_per_sheet and row_count >= max_rows_per_sheet:
                break

        # 시트에 텍스트가 하나도 없고 이미지가 있으면 "시트 요약 Doc" 하나 남김
        if row_count == 0 and sheet_images:
            md = {
                "source": xlsx_path,
                "file_name": file_name,
                "doc_type": "xlsx",
                "page": None,
                "slide": None,
                "sheet": ws.title,
                "row": None,
                "image_count": image_count,
                "image_paths": sheet_images,  # 요약 doc에는 이미지 붙이는 게 유용
            }
            docs.append(Document(page_content="", metadata=md))

    wb.close()
    return docs


def xlsx_image_manifest(
    xlsx_path: str, xlsx_img_map: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for sheet, paths in xlsx_img_map.items():
        for ip in paths:
            items.append(
                {
                    "source_file": xlsx_path,
                    "source_type": "xlsx",
                    "page": None,
                    "slide": None,
                    "sheet": sheet,
                    "row": None,
                    "image_path": ip,
                }
            )
    return items
