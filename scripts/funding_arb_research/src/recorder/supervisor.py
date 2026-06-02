"""Recorder supervisor.

Owns the writer + the per-venue WS client tasks + the periodic
quota-enforcement and stats-log tick. Ensures clean shutdown on
SIGINT / SIGTERM so in-flight rows are flushed to parquet rather than
dropped.

The supervisor itself is venue-agnostic; venue clients are passed in
already constructed.
"""
from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path
from typing import Optional

from .quota import QuotaManager
from .venue_clients.base import VenueClient
from .writer import ParquetWriter, WriterConfig
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.recorder.supervisor")


class RecorderSupervisor:
    def __init__(
        self,
        clients: list[VenueClient],
        writer: ParquetWriter,
        quota_gb: float = 50.0,
        stats_interval_s: float = 60.0,
    ) -> None:
        self.clients = clients
        self.writer = writer
        self.quota = QuotaManager(writer.base, quota_gb)
        self.stats_interval_s = stats_interval_s
        self._stop = asyncio.Event()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                # Windows / restricted env; fall back to default behavior
                pass

    async def _stats_loop(self) -> None:
        started_mono = time.monotonic()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.stats_interval_s)
            except asyncio.TimeoutError:
                pass
            uptime = time.monotonic() - started_mono
            now_wall = time.time()  # client stamps with time.time()
            for c in self.clients:
                stale = now_wall - c.last_msg_ts if c.last_msg_ts else None
                _LOG.info(
                    "venue=%s msgs=%d last_msg_age=%s",
                    c.cfg.venue, c.msgs_received,
                    f"{stale:.1f}s" if stale is not None else "n/a",
                )
            _LOG.info(
                "writer rows=%d files=%d uptime=%.0fs",
                self.writer.rows_written, self.writer.files_written, uptime,
            )
            self.quota.tick()

    async def run(self) -> None:
        self._install_signal_handlers()
        await self.writer.start()
        client_tasks = [
            asyncio.create_task(c.run(), name=f"recorder-{c.cfg.venue}")
            for c in self.clients
        ]
        stats_task = asyncio.create_task(self._stats_loop(), name="recorder-stats")
        try:
            await self._stop.wait()
        finally:
            _LOG.info("shutdown: cancelling tasks and flushing")
            for t in client_tasks:
                t.cancel()
            stats_task.cancel()
            for t in [*client_tasks, stats_task]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await self.writer.stop()
            _LOG.info("shutdown complete; rows=%d files=%d",
                      self.writer.rows_written, self.writer.files_written)
