"""Parquet writer with date/venue/coin/channel partitioning.

Each venue client pushes rows into a single ``ParquetWriter`` instance
via ``put(row)``. Rows are buffered per (channel, venue, coin) and
flushed to a parquet file when either:

  * the buffer reaches ``flush_rows``, or
  * ``flush_interval_s`` seconds have elapsed since the last flush.

File path layout::

  data/recorder/<channel>/dt=YYYY-MM-DD/venue=<v>/coin=<c>/<HHMMSSZ>_<n>.parquet

The writer is venue-agnostic. It accepts dicts and writes whatever
columns are present, with two required keys: ``timestamp_utc`` (event
time) and ``recv_ts_utc`` (local receipt time). The diff between the
two is the publication-latency proxy used by the engine.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..utils.io import data_dir
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.recorder.writer")


@dataclass
class WriterConfig:
    flush_rows: int = 5_000
    flush_interval_s: float = 30.0
    compression: str = "zstd"
    base_dir: Optional[Path] = None  # defaults to data/recorder


@dataclass
class _Buffer:
    rows: list[dict] = field(default_factory=list)
    last_flush_ts: float = field(default_factory=time.monotonic)
    seq: int = 0


class ParquetWriter:
    """Async-friendly buffered parquet writer.

    Thread-safety: not thread-safe. All ``put`` calls must come from the
    same event loop. The periodic flusher runs as one asyncio task.
    """

    def __init__(self, cfg: Optional[WriterConfig] = None) -> None:
        self.cfg = cfg or WriterConfig()
        self.base = self.cfg.base_dir or (data_dir() / "recorder")
        self.base.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[tuple[str, str, str], _Buffer] = defaultdict(_Buffer)
        self._stopped = False
        self._flusher_task: Optional[asyncio.Task] = None
        self.rows_written = 0
        self.files_written = 0

    async def start(self) -> None:
        self._stopped = False
        self._flusher_task = asyncio.create_task(self._flusher_loop(), name="recorder-flusher")

    async def stop(self) -> None:
        self._stopped = True
        if self._flusher_task is not None:
            self._flusher_task.cancel()
            try:
                await self._flusher_task
            except asyncio.CancelledError:
                pass
        self.flush_all()

    def put(self, channel: str, venue: str, coin: str, row: dict[str, Any]) -> None:
        if "timestamp_utc" not in row or "recv_ts_utc" not in row:
            raise ValueError(f"row missing required ts fields: {row.keys()}")
        key = (channel, venue, coin)
        buf = self._buffers[key]
        buf.rows.append(row)
        if len(buf.rows) >= self.cfg.flush_rows:
            self._flush_key(key)

    def flush_all(self) -> None:
        for key in list(self._buffers.keys()):
            self._flush_key(key)

    def _flush_key(self, key: tuple[str, str, str]) -> None:
        buf = self._buffers[key]
        if not buf.rows:
            buf.last_flush_ts = time.monotonic()
            return
        channel, venue, coin = key
        rows = buf.rows
        buf.rows = []
        buf.last_flush_ts = time.monotonic()
        try:
            df = pd.DataFrame(rows)
            for col in ("timestamp_utc", "recv_ts_utc"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            now = pd.Timestamp.now(tz="UTC")
            day = now.strftime("%Y-%m-%d")
            stamp = now.strftime("%H%M%SZ")
            buf.seq += 1
            out = (
                self.base
                / channel
                / f"dt={day}"
                / f"venue={venue}"
                / f"coin={coin}"
                / f"{stamp}_{buf.seq:04d}.parquet"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False, compression=self.cfg.compression)
            self.rows_written += len(df)
            self.files_written += 1
            _LOG.debug("flushed %d rows to %s", len(df), out)
        except Exception:  # noqa: BLE001 — must never kill the recorder
            _LOG.exception("flush failed for key=%s; dropped %d rows", key, len(rows))

    async def _flusher_loop(self) -> None:
        interval = self.cfg.flush_interval_s
        while not self._stopped:
            try:
                await asyncio.sleep(interval)
                now = time.monotonic()
                for key, buf in list(self._buffers.items()):
                    if buf.rows and (now - buf.last_flush_ts) >= interval:
                        self._flush_key(key)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                _LOG.exception("flusher iteration failed")
