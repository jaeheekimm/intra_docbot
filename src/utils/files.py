import os
import re
from typing import List


def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name) # 윈도우에서 파일명에 못 쓰는 문자들 치환
    name = re.sub(r"\s+", " ", name).strip()    # 공백 제거
    return name

"""
    extract.py에서 입력 폴더 전체를 훑음
    모든 파일 경로를 리스트로 모음
    그걸 하나씩 parser에 넘김
"""
def walk_files(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            paths.append(os.path.join(dirpath, fn))
    return paths
