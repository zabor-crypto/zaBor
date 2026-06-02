"""Event-time rolling windows for second-level feature computation.

Performance design:
  Timestamps are monotonically non-decreasing, so bisect_left finds the window
  boundary in O(log n) instead of the previous O(n) linear scan.

  Both EventWindow and MidPriceWindow use parallel flat lists (_ts, _vals) for
  cache-friendly iteration and fast list-slicing.  A logical _start pointer
  advances cheaply on each add(); the backing lists are compacted only when
  front-waste exceeds half their length (amortised O(1) prune).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TimedValue:
    """Returned by EventWindow.latest() for backward compatibility."""
    ts_ms: int
    value: float


class EventWindow:
    """Time-windowed accumulator with O(log n) range queries.

    All queries (sum, mean) use bisect_left to locate the window boundary,
    then sum/slice only the valid suffix — O(log n + k) instead of O(n).
    """

    __slots__ = ("max_window_ms", "_ts", "_vals", "_start")

    def __init__(self, max_window_ms: int) -> None:
        self.max_window_ms = int(max_window_ms)
        self._ts: List[int] = []
        self._vals: List[float] = []
        self._start: int = 0

    def add(self, ts_ms: int, value: float) -> None:
        ts_ms = int(ts_ms)
        self._ts.append(ts_ms)
        self._vals.append(float(value))
        # Advance start pointer past expired entries — O(log n), no data movement.
        cutoff = ts_ms - self.max_window_ms
        if self._ts[self._start] < cutoff:
            self._start = bisect.bisect_left(self._ts, cutoff, self._start)
        # Compact when front-waste exceeds half the list (amortised O(1)).
        if self._start > 64 and self._start >= (len(self._ts) >> 1):
            self._ts = self._ts[self._start:]
            self._vals = self._vals[self._start:]
            self._start = 0

    def sum(self, window_ms: int, now_ts_ms: Optional[int] = None) -> float:
        if self._start >= len(self._ts):
            return 0.0
        if now_ts_ms is None:
            now_ts_ms = self._ts[-1]
        cutoff = int(now_ts_ms) - int(window_ms)
        idx = bisect.bisect_left(self._ts, cutoff, self._start)
        return float(sum(self._vals[idx:]))

    def mean(self, window_ms: int, now_ts_ms: Optional[int] = None) -> float:
        if self._start >= len(self._ts):
            return 0.0
        if now_ts_ms is None:
            now_ts_ms = self._ts[-1]
        cutoff = int(now_ts_ms) - int(window_ms)
        idx = bisect.bisect_left(self._ts, cutoff, self._start)
        vals = self._vals[idx:]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def latest(self) -> Optional[TimedValue]:
        if self._start >= len(self._ts):
            return None
        return TimedValue(ts_ms=self._ts[-1], value=self._vals[-1])


class MidPriceWindow:
    """Compact mid-price trajectory for impact/recovery signals.

    Uses the same parallel-list + bisect design as EventWindow.
    return_bps() is O(log n) for start-point lookup.
    """

    __slots__ = ("max_window_ms", "_ts", "_vals", "_start")

    def __init__(self, max_window_ms: int) -> None:
        self.max_window_ms = int(max_window_ms)
        self._ts: List[int] = []
        self._vals: List[float] = []
        self._start: int = 0

    def add(self, ts_ms: int, mid_price: float) -> None:
        ts_ms = int(ts_ms)
        self._ts.append(ts_ms)
        self._vals.append(float(mid_price))
        cutoff = ts_ms - self.max_window_ms
        if self._ts[self._start] < cutoff:
            self._start = bisect.bisect_left(self._ts, cutoff, self._start)
        if self._start > 64 and self._start >= (len(self._ts) >> 1):
            self._ts = self._ts[self._start:]
            self._vals = self._vals[self._start:]
            self._start = 0

    def return_bps(self, lookback_ms: int, now_ts_ms: Optional[int] = None) -> float:
        n = len(self._ts)
        if n - self._start < 2:
            return 0.0
        if now_ts_ms is None:
            now_ts_ms = self._ts[-1]
        cutoff = int(now_ts_ms) - int(lookback_ms)
        idx = bisect.bisect_left(self._ts, cutoff, self._start)
        # Fall back to oldest available entry if nothing within window.
        if idx >= n:
            idx = self._start
        start_val = self._vals[idx]
        if start_val <= 0.0:
            return 0.0
        return ((self._vals[-1] - start_val) / start_val) * 10_000.0

    def latest(self) -> Optional[Tuple[int, float]]:
        if self._start >= len(self._ts):
            return None
        return (self._ts[-1], self._vals[-1])
