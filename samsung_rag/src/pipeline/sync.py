"""
src/pipeline/sync.py
─────────────────────────────────────────────────────────────────────────────
문서 동기화 모듈 (Polling 방식).

- 주기적으로 소스 디렉터리를 스캔하여 신규/변경 파일을 인덱싱한다.
- 백그라운드 스레드로 실행 (FastAPI lifespan 또는 직접 호출).
- 수동 트리거: POST /api/admin/sync
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_POLL_INTERVAL: int = int(os.getenv("SYNC_INTERVAL_SECONDS", "300"))  # 기본 5분
_SAMPLE_DIR: Path = Path(os.getenv("SAMPLE_DIR", "data/sample"))

# 동기화 상태
_sync_lock = threading.Lock()
_last_sync_at: Optional[float] = None
_last_sync_result: list[dict] = []
_is_running = False
_stop_event = threading.Event()


def get_sync_status() -> dict:
    """
    마지막 동기화 상태를 반환한다.

    Returns:
        {
            "last_sync_at": ISO 타임스탬프 or None,
            "last_sync_result": 결과 리스트,
            "is_running": bool
        }
    """
    import datetime

    with _sync_lock:
        return {
            "last_sync_at": (
                datetime.datetime.fromtimestamp(_last_sync_at).isoformat()
                if _last_sync_at
                else None
            ),
            "last_sync_result": _last_sync_result,
            "is_running": _is_running,
        }


def run_sync(directory: Path | None = None) -> list[dict]:
    """
    동기화를 즉시 실행한다 (수동 트리거 및 폴링 루프 공용).

    Args:
        directory: 스캔할 디렉터리 (기본: SAMPLE_DIR)

    Returns:
        ingest_directory 결과 리스트
    """
    global _last_sync_at, _last_sync_result, _is_running

    with _sync_lock:
        if _is_running:
            logger.warning("동기화 이미 실행 중 → 건너뜀")
            return []
        _is_running = True

    try:
        from src.pipeline.ingest import ingest_directory

        scan_dir = directory or _SAMPLE_DIR
        logger.info("동기화 시작: %s", scan_dir)
        results = ingest_directory(scan_dir)

        indexed = sum(1 for r in results if r.get("status") == "indexed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        errors = sum(1 for r in results if r.get("status") == "error")
        logger.info(
            "동기화 완료: indexed=%d, skipped=%d, errors=%d",
            indexed, skipped, errors,
        )

        with _sync_lock:
            _last_sync_at = time.time()
            _last_sync_result = results

        return results
    except Exception as e:
        logger.error("동기화 실패: %s", e)
        raise
    finally:
        with _sync_lock:
            _is_running = False


def _polling_loop(directory: Path | None, interval: int) -> None:
    """
    폴링 루프 스레드 함수.

    Args:
        directory: 스캔 디렉터리
        interval:  폴링 주기 (초)
    """
    logger.info("동기화 폴링 시작 (interval=%ds)", interval)
    while not _stop_event.is_set():
        try:
            run_sync(directory)
        except Exception as e:
            logger.error("폴링 동기화 오류: %s", e)
        # interval 동안 stop_event 대기 (즉시 중단 가능)
        _stop_event.wait(timeout=interval)
    logger.info("동기화 폴링 종료")


def start_sync_thread(
    directory: Path | None = None,
    interval: int = _POLL_INTERVAL,
) -> threading.Thread:
    """
    백그라운드 폴링 스레드를 시작한다.

    Args:
        directory: 스캔 디렉터리
        interval:  폴링 주기 (초)

    Returns:
        실행 중인 Thread 객체
    """
    _stop_event.clear()
    thread = threading.Thread(
        target=_polling_loop,
        args=(directory, interval),
        daemon=True,
        name="sync-polling",
    )
    thread.start()
    logger.info("동기화 스레드 시작됨 (thread_id=%s)", thread.ident)
    return thread


def stop_sync_thread() -> None:
    """폴링 스레드 종료 신호를 보낸다."""
    _stop_event.set()
    logger.info("동기화 스레드 종료 신호 전송")
