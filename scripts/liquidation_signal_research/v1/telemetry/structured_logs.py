"""Structured JSONL telemetry sinks for live and replay diagnostics.

Performance design:
  JsonlSink keeps the file handle open for the process lifetime instead of
  opening and closing on every write.

  Previous version used buffering=1 (line-buffered), which flushed every line
  to the OS page cache via write(2).  Decision and feature logs fire at the
  decision cadence (~16 writes/sec across 4 symbols) plus every risk event.
  Each OS flush stalls the asyncio event loop briefly; on a slow VPS this added
  measurable latency.

  New version uses a fully-buffered handle (64 KB OS buffer) and explicitly
  flushes every flush_interval_writes writes.  Default is 50 writes ≈ 3 s at
  idle cadence, keeping telemetry lag well under any actionable window.
  feature_log and decision_log are instantiated with flush_interval_writes=50.
  order_lifecycle_log and risk_health_log use flush_interval_writes=1 (flush
  every write) to preserve fill/risk event durability on crash.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Mapping, Optional


class JsonlSink:
    def __init__(self, path: Path, flush_interval_writes: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._flush_interval = max(1, int(flush_interval_writes))
        self._writes_since_flush: int = 0
        # Use a 64 KB OS write buffer; explicit flush happens every
        # flush_interval_writes writes instead of on every newline.
        self._handle: TextIOWrapper = self.path.open("a", encoding="utf-8", buffering=65536)

    def write(self, payload: Mapping[str, object]) -> None:
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n"
        with self._lock:
            self._handle.write(line)
            self._writes_since_flush += 1
            if self._writes_since_flush >= self._flush_interval:
                self._handle.flush()
                self._writes_since_flush = 0

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()
            self._writes_since_flush = 0

    def close(self) -> None:
        with self._lock:
            try:
                self._handle.flush()
                self._handle.close()
            except Exception:
                pass


class TelemetryRouter:
    """Fan-out for mandatory v1 telemetry streams."""

    def __init__(
        self,
        root: Path,
        *,
        # High-frequency logs: flush every N writes to amortise OS overhead.
        feature_flush_interval: int = 50,
        decision_flush_interval: int = 50,
        # Low-frequency / high-durability logs: flush on every write.
        order_flush_interval: int = 1,
        risk_flush_interval: int = 1,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.raw_capture_log = JsonlSink(
            self.root / "raw_capture_log.jsonl",
            flush_interval_writes=risk_flush_interval,
        )
        self.feature_log = JsonlSink(
            self.root / "feature_log.jsonl",
            flush_interval_writes=feature_flush_interval,
        )
        self.decision_log = JsonlSink(
            self.root / "decision_log.jsonl",
            flush_interval_writes=decision_flush_interval,
        )
        self.order_lifecycle_log = JsonlSink(
            self.root / "order_lifecycle_log.jsonl",
            flush_interval_writes=order_flush_interval,
        )
        self.risk_health_log = JsonlSink(
            self.root / "risk_health_log.jsonl",
            flush_interval_writes=risk_flush_interval,
        )
        self.replay_diagnostics_log = JsonlSink(
            self.root / "replay_diagnostics_log.jsonl",
            flush_interval_writes=decision_flush_interval,
        )

    @staticmethod
    def _envelope(payload: Mapping[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        out.update(payload)
        return out

    def log_raw_capture(self, payload: Mapping[str, object]) -> None:
        self.raw_capture_log.write(self._envelope(payload))

    def log_feature(self, payload: Mapping[str, object]) -> None:
        self.feature_log.write(self._envelope(payload))

    def log_decision(self, payload: Mapping[str, object]) -> None:
        self.decision_log.write(self._envelope(payload))

    def log_order(self, payload: Mapping[str, object]) -> None:
        self.order_lifecycle_log.write(self._envelope(payload))

    def log_risk(self, payload: Mapping[str, object]) -> None:
        self.risk_health_log.write(self._envelope(payload))

    def log_replay(self, payload: Mapping[str, object]) -> None:
        self.replay_diagnostics_log.write(self._envelope(payload))

    def flush_all(self) -> None:
        for sink in (
            self.raw_capture_log,
            self.feature_log,
            self.decision_log,
            self.order_lifecycle_log,
            self.risk_health_log,
            self.replay_diagnostics_log,
        ):
            sink.flush()

    def close(self) -> None:
        for sink in (
            self.raw_capture_log,
            self.feature_log,
            self.decision_log,
            self.order_lifecycle_log,
            self.risk_health_log,
            self.replay_diagnostics_log,
        ):
            sink.close()
