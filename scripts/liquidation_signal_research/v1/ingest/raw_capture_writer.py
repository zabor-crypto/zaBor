"""Asynchronous raw event capture to immutable partition store.

Performance design:
  The writer loop runs in a dedicated ThreadPoolExecutor thread so zstd
  compression and OS page-cache writes never block the asyncio event loop.
  The asyncio side is limited to queue operations (put_nowait / get) and a
  periodic flush heartbeat — both are O(1) and GIL-friendly.

  queue_size default is 10 000.  At 100 events/sec that gives ~100 s of
  back-pressure before drops, comfortably absorbing Binance startup bursts
  (~5 000 events in the first few seconds).  put_nowait is used so on_raw_event
  never blocks the event loop even when the queue is full.

  A lightweight periodic flush task wakes every flush_interval_seconds and
  schedules store.flush_all() on the writer thread to ensure data is committed
  to the OS page cache during low-activity periods.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Union

from v1.contracts.schema_registry import DEFAULT_SCHEMA_REGISTRY
from v1.contracts.event_types import RawMarketEvent
from v1.store.jsonl_partition_store import JsonlPartitionStore

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 10_000
_DEFAULT_FLUSH_INTERVAL_SECONDS = 10.0

# Sentinel value sent to the writer thread to signal shutdown.
_STOP = object()
# Sentinel value sent to the writer thread to trigger an explicit flush.
_FLUSH = object()


@dataclass
class CaptureStats:
    enqueued: int = 0
    written: int = 0
    dropped: int = 0


class RawCaptureWriter:
    """Thread-isolated raw event capture writer.

    The public interface is fully async.  Internally, all blocking I/O
    (zstd encode + write + periodic FLUSH_BLOCK) runs on a private daemon
    thread, keeping the asyncio event loop completely free.
    """

    def __init__(
        self,
        *,
        store: JsonlPartitionStore,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        schema_name: str = "raw_market_event_v1",
        flush_interval_seconds: float = _DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.store = store
        self.schema_name = schema_name
        # Thread-safe queue shared between asyncio (put_nowait) and writer thread (get).
        self._q: queue.SimpleQueue[Union[RawMarketEvent, object]] = queue.SimpleQueue()
        self._queue_size = int(queue_size)
        self._queue_approx: int = 0  # approximate depth for overflow guard
        self.stats = CaptureStats()
        self._flush_interval_seconds = float(flush_interval_seconds)

        self._thread: Optional[threading.Thread] = None
        self._flush_task: Optional[asyncio.Task] = None

    # ── Public async API ───────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._writer_thread, daemon=True, name="raw-capture-writer"
            )
            self._thread.start()
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(
                self._periodic_flush(), name="raw-capture-flush"
            )

    async def stop(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        # Signal the writer thread to drain and exit.
        self._q.put(_STOP)
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
            self._thread = None

    async def enqueue(self, event: RawMarketEvent) -> None:
        if self._queue_approx >= self._queue_size:
            self.stats.dropped += 1
            if self.stats.dropped % 100 == 1:
                logger.warning(
                    "raw_capture_writer queue full (dropped=%d); "
                    "increase capture_queue_size or reduce write load",
                    self.stats.dropped,
                )
            return
        self._q.put(event)
        self._queue_approx += 1
        self.stats.enqueued += 1

    # ── Internal: asyncio side ─────────────────────────────────────────────────

    async def _periodic_flush(self) -> None:
        """Schedule a flush on the writer thread every flush_interval_seconds."""
        while True:
            await asyncio.sleep(self._flush_interval_seconds)
            self._q.put(_FLUSH)

    # ── Internal: writer thread ────────────────────────────────────────────────

    def _writer_thread(self) -> None:
        """Blocking I/O loop — runs entirely off the event loop."""
        while True:
            item = self._q.get()

            if item is _STOP:
                break

            if item is _FLUSH:
                try:
                    self.store.flush_all()
                except Exception:
                    logger.exception("periodic raw-capture flush failed")
                continue

            event: RawMarketEvent = item  # type: ignore[assignment]
            self._queue_approx -= 1
            try:
                payload = event.to_dict()
                DEFAULT_SCHEMA_REGISTRY.validate(self.schema_name, payload)
                self.store.append_record(
                    schema_name=self.schema_name,
                    schema_version=event.schema_version,
                    symbol=event.symbol,
                    stream=event.stream,
                    event_ts_ms=event.exchange_ts_ms,
                    record=payload,
                )
                self.stats.written += 1
            except Exception:
                self.stats.dropped += 1
                logger.exception("Failed to persist raw event: id=%s", event.event_id)

        self.store.close()
