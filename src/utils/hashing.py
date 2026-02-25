import hashlib


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_short(data: bytes, n: int = 12) -> str:
    return sha1_hex(data)[:n]
