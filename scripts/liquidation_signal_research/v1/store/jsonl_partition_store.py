"""Immutable append-only JSONL partition store with zstd compression.

Performance design:
  The original code called writer.flush(FLUSH_FRAME) + file_handle.flush() on
  every single appended record.  FLUSH_FRAME finalises a complete zstd frame
  (header + compressed blocks + checksum) per event, which means:
    1. Each record is compressed in isolation — the compressor cannot exploit
       repetition across records, tanking ratios from ~10:1 to ~1:1.
    2. Every write triggers an OS write(2) + fdatasync-like flush, causing the
       asyncio event loop to block on disk I/O at the raw-event rate (~100/s).
       On a shared VPS this stalls the loop long enough to miss WebSocket pings.

  Fix: buffer N records in memory before flushing.  Every flush_interval_events
  records we emit one FLUSH_BLOCK (keeps the compression context alive → good
  ratios, still readable) and one file_handle.flush() (OS buffer drain).
  On partition close / hour roll-over we emit FLUSH_FRAME to seal the file.
  This reduces OS writes from O(events) to O(events / flush_interval_events).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Tuple

import zstandard as zstd

from .manifest import StoreManifest

_DEFAULT_FLUSH_INTERVAL_EVENTS = 500  # flush after every N appended records


@dataclass
class _PartitionHandle:
    path: Path
    file_handle: object
    writer: object
    writes_since_flush: int = field(default=0, compare=False)


class JsonlPartitionStore:
    """
    Append-only partition store.

    Path pattern:
      root/date=YYYY-MM-DD/hour=HH/symbol=SYMBOL/stream=STREAM/<schema>.jsonl.zst
    """

    def __init__(
        self,
        root: Path,
        compression_level: int = 3,
        flush_interval_events: int = _DEFAULT_FLUSH_INTERVAL_EVENTS,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = StoreManifest(self.root)
        self._compression_level = int(compression_level)
        self._flush_interval_events = max(1, int(flush_interval_events))
        self._handles: Dict[Tuple[str, str, str, str, str], _PartitionHandle] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _partition_time(event_ts_ms: int) -> Tuple[str, str]:
        dt = datetime.fromtimestamp(event_ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H")

    def _partition_path(
        self,
        schema_name: str,
        symbol: str,
        stream: str,
        partition_date: str,
        partition_hour: str,
    ) -> Path:
        return (
            self.root
            / f"date={partition_date}"
            / f"hour={partition_hour}"
            / f"symbol={symbol}"
            / f"stream={stream}"
            / f"{schema_name}.jsonl.zst"
        )

    def _open_partition(
        self,
        *,
        schema_name: str,
        schema_version: str,
        symbol: str,
        stream: str,
        partition_date: str,
        partition_hour: str,
    ) -> _PartitionHandle:
        key = (schema_name, symbol, stream, partition_date, partition_hour)
        handle = self._handles.get(key)
        if handle is not None:
            return handle

        path = self._partition_path(schema_name, symbol, stream, partition_date, partition_hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()

        file_handle = path.open("ab")
        compressor = zstd.ZstdCompressor(level=self._compression_level, write_content_size=False)
        writer = compressor.stream_writer(file_handle, closefd=False)
        handle = _PartitionHandle(path=path, file_handle=file_handle, writer=writer)
        self._handles[key] = handle

        # Close handles for prior hours of the same stream to prevent file handle accumulation.
        stale_keys = [
            k for k in list(self._handles.keys())
            if k != key and k[0] == schema_name and k[1] == symbol and k[2] == stream
        ]
        for stale_key in stale_keys:
            stale = self._handles.pop(stale_key, None)
            if stale is not None:
                self._close_handle(stale)

        if created:
            self.manifest.append(
                schema_name=schema_name,
                schema_version=schema_version,
                symbol=symbol,
                stream=stream,
                partition_date=partition_date,
                partition_hour=partition_hour,
                file_path=path,
                metadata={"format": "jsonl.zst", "immutable": True},
            )
        return handle

    @staticmethod
    def _close_handle(handle: _PartitionHandle) -> None:
        """Seal and close a partition handle: flush remaining data, write frame end."""
        try:
            handle.writer.flush(zstd.FLUSH_FRAME)
            handle.writer.close()
        finally:
            try:
                handle.file_handle.flush()
            finally:
                handle.file_handle.close()

    def append_record(
        self,
        *,
        schema_name: str,
        schema_version: str,
        symbol: str,
        stream: str,
        event_ts_ms: int,
        record: Mapping[str, object],
    ) -> Path:
        partition_date, partition_hour = self._partition_time(event_ts_ms)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"

        with self._lock:
            handle = self._open_partition(
                schema_name=schema_name,
                schema_version=schema_version,
                symbol=symbol,
                stream=stream,
                partition_date=partition_date,
                partition_hour=partition_hour,
            )
            handle.writer.write(line.encode("utf-8"))
            handle.writes_since_flush += 1
            # Flush every flush_interval_events records using FLUSH_BLOCK.
            # This keeps the zstd compression context alive across records
            # (better compression ratios) while bounding data loss to at most
            # flush_interval_events records on an unclean shutdown.
            if handle.writes_since_flush >= self._flush_interval_events:
                handle.writer.flush(zstd.FLUSH_BLOCK)
                handle.file_handle.flush()
                handle.writes_since_flush = 0
            return handle.path

    def flush_all(self) -> None:
        """Flush all open partition handles to OS (called from background task)."""
        with self._lock:
            for handle in self._handles.values():
                if handle.writes_since_flush > 0:
                    handle.writer.flush(zstd.FLUSH_BLOCK)
                    handle.file_handle.flush()
                    handle.writes_since_flush = 0

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                self._close_handle(handle)
            self._handles.clear()

    def __enter__(self) -> "JsonlPartitionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
