import os
import re
from typing import Dict, List, Any, Optional, Tuple

from pptx import Presentation
from langchain_core.documents import Document

from src.utils.files import safe_filename
from src.utils.hashing import sha1_short


def _normalize_pptx_text(t: str) -> str:
    # PPTX 텍스트는 줄바꿈/공백이 어색한 경우가 있어 가볍게 정리
    s = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def _extract_sorted_text_blocks(slide) -> List[Tuple[int, int, str]]:
    """
    슬라이드 내 텍스트 프레임들을 (top, left) 기준으로 정렬 가능한 형태로 추출.
    return: [(top, left, text), ...]
    """
    blocks: List[Tuple[int, int, str]] = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.text_frame:
            raw = shape.text_frame.text
            text = _normalize_pptx_text(raw)
            if not text:
                continue
            top = int(getattr(shape, "top", 0) or 0)
            left = int(getattr(shape, "left", 0) or 0)
            blocks.append((top, left, text))

    blocks.sort(key=lambda x: (x[0], x[1]))  # 위->아래, 왼->오
    return blocks


def _pick_title_from_blocks(
    blocks: List[Tuple[int, int, str]],
) -> Tuple[Optional[str], List[str]]:
    """
    매우 보수적 title 추정:
    - 정렬된 첫 텍스트 중 "짧은" 텍스트를 title로 (<= 80자)
    - title을 고르지 못하면 None
    return: (title or None, body_texts)
    """
    if not blocks:
        return None, []

    texts = [b[2] for b in blocks]

    # title 후보: 첫 몇 개 중 짧고(<=80) 너무 길게 줄바꿈 없는 것
    title: Optional[str] = None
    title_idx: Optional[int] = None
    for i, t in enumerate(texts[:5]):  # 상단 쪽 몇 개만 본다
        one_line = t.replace("\n", " ").strip()
        if 1 <= len(one_line) <= 80:
            title = one_line
            title_idx = i
            break

    if title_idx is None:
        return None, texts

    body = texts[:title_idx] + texts[title_idx + 1 :]
    return title, body


def extract_pptx_images(pptx_path: str, out_dir: str) -> Dict[int, List[str]]:
    """
    return: {slide_number(1-based): [image_path, ...]}
    """
    mapping: Dict[int, List[str]] = {}

    prs = Presentation(pptx_path)
    base = safe_filename(os.path.basename(pptx_path))

    # 슬라이드 크기(EMU)
    sw, sh = prs.slide_width, prs.slide_height

    # 오른쪽 위 코너 영역(비율) - 필요하면 숫자만 조절
    RU_X = int(sw * 0.82)
    RU_Y = int(sh * 0.18)

    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if getattr(shape, "shape_type", None) == 13 and getattr(
                shape, "image", None
            ):
                x, y = int(shape.left), int(shape.top)

                # 1) 1번 슬라이드: 이미지 전부 제거
                if slide_idx == 1:
                    continue

                # 2) 2번부터: 오른쪽 위 코너 이미지만 제거
                if slide_idx >= 2 and (x >= RU_X) and (y <= RU_Y):
                    continue

                img = shape.image
                blob = img.blob
                ext = img.ext

                hid = sha1_short(blob)
                fname = f"{base}__slide{slide_idx:03d}__{hid}.{ext}"
                save_path = os.path.join(out_dir, fname)

                if not os.path.exists(save_path):
                    with open(save_path, "wb") as f:
                        f.write(blob)

                mapping.setdefault(slide_idx, []).append(save_path)

    return mapping


def load_pptx_docs(pptx_path: str, ppt_img_map: Dict[int, List[str]]) -> List[Document]:
    """
    PPTX는 slide 단위 Document로 생성.
    ✅ (top,left) 좌표 기준 텍스트 정렬
    ✅ title을 metadata에 추가(원문 삭제 없이, title은 별도 필드로)
    """
    prs = Presentation(pptx_path)
    file_name = os.path.basename(pptx_path)

    docs: List[Document] = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        blocks = _extract_sorted_text_blocks(slide)
        title, body_texts = _pick_title_from_blocks(blocks)

        # 본문 content: body_texts를 줄바꿈으로 합치기
        content = "\n".join([t for t in body_texts if t]).strip()

        # 텍스트도 없고 이미지도 없으면 스킵
        if not title and not content and not ppt_img_map.get(slide_idx):
            continue

        md = {
            "source": pptx_path,
            "file_name": file_name,
            "doc_type": "pptx",
            "page": None,
            "slide": slide_idx,
            "sheet": None,
            "row": None,
            "title": title,  # 추가
            "image_paths": ppt_img_map.get(slide_idx, []),
        }

        # title만 있고 content가 비면 content는 빈 문자열로 두되 doc은 남김
        docs.append(Document(page_content=content, metadata=md))

    return docs


def pptx_image_manifest(
    pptx_path: str, ppt_img_map: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for slide, paths in ppt_img_map.items():
        for ip in paths:
            items.append(
                {
                    "source_file": pptx_path,
                    "source_type": "pptx",
                    "page": None,
                    "slide": slide,
                    "sheet": None,
                    "row": None,
                    "image_path": ip,
                }
            )
    return items
