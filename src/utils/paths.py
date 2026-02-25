"""
    문서 -> 파싱
-> parsed_documents.jsonl 저장
-> extracted_images 폴더에 이미지 저장
-> image_manifest.json 저장
"""

# extract.py 에서 사용됨

import os

DATA_DIR = "./data"
OUT_IMG_DIR = os.path.join(DATA_DIR, "extracted_images")  # 이미지 실제 파일 저장소
OUT_JSONL = os.path.join(DATA_DIR, "parsed_documents.jsonl")
OUT_IMG_MANIFEST = os.path.join(
    DATA_DIR, "image_manifest.json"
)  # 이미지가 어디서 왔는지(출처/위치/메타) 를 기록한 인덱스
