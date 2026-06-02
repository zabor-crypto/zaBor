"""Disk-quota rotation.

The recorder writes parquet files under ``data/recorder/``. Without
bounds the directory grows forever; on a research box that's fine for a
while but eventually the disk fills. ``QuotaManager`` deletes the
oldest day-partitions until total usage is back under
``max_gb``. It runs periodically from the supervisor.

Deletion granularity is one ``dt=YYYY-MM-DD`` directory at a time so we
never tear a partial day. The recorder coverage window in run reports
should reference the oldest surviving day, not the recorder start date.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable

from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.recorder.quota")


def _du_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _iter_day_partitions(base: Path) -> Iterable[Path]:
    """Yield every ``dt=YYYY-MM-DD`` directory under base, oldest first."""
    if not base.exists():
        return
    days: list[Path] = []
    for channel_dir in base.iterdir():
        if not channel_dir.is_dir():
            continue
        for day_dir in channel_dir.iterdir():
            if day_dir.is_dir() and day_dir.name.startswith("dt="):
                days.append(day_dir)
    days.sort(key=lambda p: p.name)  # dt=YYYY-MM-DD sorts chronologically
    yield from days


def enforce_quota(base: Path, max_gb: float) -> int:
    """Delete oldest day-partitions until usage ≤ ``max_gb``.

    Returns number of partitions deleted.
    """
    if max_gb <= 0:
        return 0
    cap = int(max_gb * 1024**3)
    used = _du_bytes(base)
    if used <= cap:
        return 0
    deleted = 0
    for day_dir in _iter_day_partitions(base):
        if used <= cap:
            break
        size = _du_bytes(day_dir)
        try:
            shutil.rmtree(day_dir)
            used -= size
            deleted += 1
            _LOG.warning("quota: removed %s (%.2f MB)", day_dir, size / 1024**2)
        except OSError:
            _LOG.exception("quota: failed to remove %s", day_dir)
    return deleted


class QuotaManager:
    """Periodic enforcer. Call ``tick()`` from any loop."""

    def __init__(self, base: Path, max_gb: float, check_interval_s: float = 300.0) -> None:
        self.base = base
        self.max_gb = max_gb
        self.check_interval_s = check_interval_s
        self._last = 0.0

    def tick(self) -> None:
        now = time.monotonic()
        if (now - self._last) < self.check_interval_s:
            return
        self._last = now
        enforce_quota(self.base, self.max_gb)
