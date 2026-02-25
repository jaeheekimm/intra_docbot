"""
    문서 -> 파싱
-> parsed_documents.jsonl 저장
-> extracted_images 폴더에 이미지 저장
-> image_manifest.json 저장
"""

# src/utils/paths.py
import os

DATA_DIR = os.getenv("DATA_DIR", "./data")

OUT_JSONL = os.getenv("JSONL_PATH", os.path.join(DATA_DIR, "parsed_documents.jsonl"))
OUT_IMG_DIR = os.getenv("OUT_IMG_DIR", os.path.join(DATA_DIR, "extracted_images"))
OUT_IMG_MANIFEST = os.getenv(
    "OUT_IMG_MANIFEST", os.path.join(DATA_DIR, "image_manifest.json")
)
