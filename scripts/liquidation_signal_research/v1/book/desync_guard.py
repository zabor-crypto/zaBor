"""Order book desync/staleness guard for no-trade gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DesyncState:
    symbol: str
    desynced: bool = True
    reason: str = "init"
    last_good_exchange_ts_ms: int = 0
    last_snapshot_ts_ms: int = 0


class DesyncGuard:
    def __init__(self, symbol: str, stale_after_ms: int = 2000) -> None:
        self.state = DesyncState(symbol=symbol)
        self.stale_after_ms = int(stale_after_ms)

    def mark_snapshot(self, exchange_ts_ms: int) -> None:
        self.state.desynced = False
        self.state.reason = "ok"
        self.state.last_snapshot_ts_ms = int(exchange_ts_ms)
        self.state.last_good_exchange_ts_ms = int(exchange_ts_ms)

    def mark_delta_ok(self, exchange_ts_ms: int) -> None:
        self.state.desynced = False
        self.state.reason = "ok"
        self.state.last_good_exchange_ts_ms = int(exchange_ts_ms)

    def mark_desync(self, reason: str) -> None:
        self.state.desynced = True
        self.state.reason = reason

    def refresh_staleness(self, now_exchange_ts_ms: int) -> None:
        if self.state.last_good_exchange_ts_ms <= 0:
            self.mark_desync("no_good_data")
            return
        age = int(now_exchange_ts_ms) - int(self.state.last_good_exchange_ts_ms)
        if age > self.stale_after_ms:
            self.mark_desync(f"stale_{age}ms")

    @property
    def healthy(self) -> bool:
        return not self.state.desynced

    @property
    def reason(self) -> str:
        return self.state.reason
