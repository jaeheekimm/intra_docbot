from dotenv import load_dotenv

load_dotenv()

import os
import json
from typing import List

from langchain_core.documents import Document

from src.utils.paths import DATA_DIR, OUT_IMG_DIR, OUT_JSONL, OUT_IMG_MANIFEST
from src.utils.files import walk_files

from src.parsers.pdf_parser import extract_pdf_images, load_pdf_docs, pdf_image_manifest
from src.parsers.pptx_parser import (
    extract_pptx_images,
    load_pptx_docs,
    pptx_image_manifest,
)
from src.parsers.xlsx_parser import (
    extract_xlsx_images,
    load_xlsx_docs_rows,
    xlsx_image_manifest,
)


def save_docs_jsonl(docs: List[Document], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            obj = {"page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    all_paths = walk_files(DATA_DIR)
    pdfs = [p for p in all_paths if p.lower().endswith(".pdf")]
    pptxs = [p for p in all_paths if p.lower().endswith(".pptx")]
    xlsxs = [p for p in all_paths if p.lower().endswith(".xlsx")]

    docs: List[Document] = []
    img_manifest = []

    # PDF
    for p in pdfs:
        try:
            pdf_img_map = extract_pdf_images(p, OUT_IMG_DIR)
            img_manifest.extend(pdf_image_manifest(p, pdf_img_map))
            docs.extend(load_pdf_docs(p, pdf_img_map))
        except Exception as e:
            print(f"[PDF FAIL] {p} err={e}")

    # PPTX
    for p in pptxs:
        try:
            ppt_img_map = extract_pptx_images(p, OUT_IMG_DIR)
            img_manifest.extend(pptx_image_manifest(p, ppt_img_map))
            docs.extend(load_pptx_docs(p, ppt_img_map))
        except Exception as e:
            print(f"[PPTX FAIL] {p} err={e}")

    # XLSX (행 단위)
    for p in xlsxs:
        try:
            xlsx_img_map = extract_xlsx_images(p, OUT_IMG_DIR)
            img_manifest.extend(xlsx_image_manifest(p, xlsx_img_map))
            docs.extend(load_xlsx_docs_rows(p, xlsx_img_map, attach_sheet_images=False))
        except Exception as e:
            print(f"[XLSX FAIL] {p} err={e}")

    save_docs_jsonl(docs, OUT_JSONL)
    with open(OUT_IMG_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(img_manifest, f, ensure_ascii=False, indent=2)

    print(f"OK docs={len(docs)}")
    print(f"- documents: {OUT_JSONL}")
    print(f"- images_dir: {OUT_IMG_DIR}")
    print(f"- image_manifest: {OUT_IMG_MANIFEST}")


if __name__ == "__main__":
    main()
