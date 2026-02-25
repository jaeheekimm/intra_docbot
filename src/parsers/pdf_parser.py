import os
import re
from typing import Dict, List, Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.utils.files import safe_filename
from src.utils.hashing import sha1_short


def clean_pdf_text(text: str) -> str:
    """
    PDF 텍스트를 '보수적으로' 정리.
    - 하이픈 줄바꿈:  "하이-\n픈" -> "하이픈"
    - 단일 줄바꿈: 문장 중간에 끊긴 줄바꿈은 공백으로
    - 문단 구분(빈 줄)은 유지
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) 하이픈 줄바꿈(영문/한글 모두 어느 정도 대응)
    #    예) "inter-\nnational" -> "international"
    #    예) "하이-\n픈" -> "하이픈"
    t = re.sub(r"([A-Za-z가-힣])-\n([A-Za-z가-힣])", r"\1\2", t)

    # 2) 문단 구분을 임시 토큰으로 보호
    #    (빈 줄 2개 이상은 문단으로 보고 유지)
    PARA = "\n\n<<PARA>>\n\n"
    t = re.sub(r"\n{2,}", PARA, t)

    # 3) 남은 단일 줄바꿈은 공백으로(문장 중간 끊김 완화)
    t = t.replace("\n", " ")

    # 4) 문단 토큰 복원
    t = t.replace(PARA, "\n\n")

    # 5) 공백 정리
    t = re.sub(r"[ \t]{2,}", " ", t).strip()

    return t


def extract_pdf_images(pdf_path: str, out_dir: str) -> Dict[int, List[str]]:
    """
    return: {page_number(1-based): [image_path, ...]}
    """
    mapping: Dict[int, List[str]] = {}
    doc = fitz.open(pdf_path)
    base = safe_filename(os.path.basename(pdf_path))

    MIN_W, MIN_H = 150, 150
    MIN_AREA = 150 * 150

    # (옵션) 같은 xref 재사용 시 extract_image 중복 방지
    xref_cache: Dict[int, Dict[str, Any]] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images = page.get_images(full=True)
        if not images:
            continue

        page_no = page_idx + 1
        for img_i, img in enumerate(images, start=1):
            xref = img[0]

            # extract_image 1회
            if xref in xref_cache:
                extracted = xref_cache[xref]
            else:
                extracted = doc.extract_image(xref)
                xref_cache[xref] = extracted

            blob = extracted.get("image", b"")
            if not blob:
                continue

            ext = extracted.get("ext", "bin")
            w = extracted.get("width", 0)
            h = extracted.get("height", 0)

            # 작은 이미지 스킵(아이콘/장식 제거용)
            if (w and h) and (w < MIN_W or h < MIN_H or (w * h) < MIN_AREA):
                continue

            hid = sha1_short(blob)
            fname = f"{base}__page{page_no:03d}__img{img_i:03d}__{hid}.{ext}"
            save_path = os.path.join(out_dir, fname)

            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(blob)

            mapping.setdefault(page_no, []).append(save_path)

    doc.close()
    return mapping


def load_pdf_docs(pdf_path: str, pdf_img_map: Dict[int, List[str]]) -> List[Document]:
    """
    PDF는 page 단위 Document로 생성.
    + page_content에 보수적 텍스트 정리(clean_pdf_text) 적용
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    file_name = os.path.basename(pdf_path)
    for d in docs:
        # ✅ 1) 텍스트 정리(보수적)
        d.page_content = clean_pdf_text(d.page_content)

        page0 = d.metadata.get("page", None)
        page_1based = (page0 + 1) if isinstance(page0, int) else None

        d.metadata.update(
            {
                "source": pdf_path,
                "file_name": file_name,
                "doc_type": "pdf",
                "page": page_1based,
                "slide": None,
                "sheet": None,
                "row": None,
            }
        )
        d.metadata["image_paths"] = pdf_img_map.get(page_1based, [])

    return docs


def pdf_image_manifest(
    pdf_path: str, pdf_img_map: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for page, paths in pdf_img_map.items():
        for ip in paths:
            items.append(
                {
                    "source_file": pdf_path,
                    "source_type": "pdf",
                    "page": page,
                    "slide": None,
                    "sheet": None,
                    "row": None,
                    "image_path": ip,
                }
            )
    return items
