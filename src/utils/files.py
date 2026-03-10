import os
import re
from typing import List


def safe_filename(name: str) -> str:
    """윈도우 파일명에 못 쓰는 특수문자를 _로 치환하고 공백 정규화"""
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def walk_files(root: str) -> List[str]:
    """root 하위 전체 파일 경로를 재귀적으로 수집해서 리스트로 반환"""
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            paths.append(os.path.join(dirpath, fn))
    return paths
