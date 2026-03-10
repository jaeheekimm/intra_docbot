import hashlib


def sha1_hex(data: bytes) -> str:
    """바이트 데이터의 SHA1 해시를 16진수 문자열로 반환"""
    return hashlib.sha1(data).hexdigest()


def sha1_short(data: bytes, n: int = 12) -> str:
    """SHA1 해시 앞 n자리만 잘라서 반환 (파일명용 짧은 ID)"""
    return sha1_hex(data)[:n]
