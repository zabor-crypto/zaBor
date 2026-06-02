"""Stream supervisor for health tracking and ingestion lifecycle."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from v1.contracts.event_types import MANDATORY_STREAMS, STREAM_FORCE_ORDER, RawMarketEvent
from v1.ingest.binance_ws_client import BinanceWsClient

logger = logging.getLogger(__name__)


@dataclass
class StreamHealth:
    last_recv_mono_ns: int = 0
    reconnects: int = 0
    parse_errors: int = 0


@dataclass
class SupervisorHealthSnapshot:
    healthy: bool
    unhealthy_reasons: List[str] = field(default_factory=list)


class StreamSupervisor:
    def __init__(
        self,
        *,
        symbols: Iterable[str],
        stale_after_ms: int = 2500,
        sparse_streams: Optional[Set[str]] = None,
        stale_overrides: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.stale_after_ms = int(stale_after_ms)
        self.sparse_streams = set(sparse_streams or {STREAM_FORCE_ORDER})
        self._stale_overrides: Dict[Tuple[str, str], int] = dict(stale_overrides or {})
        self._health: Dict[Tuple[str, str], StreamHealth] = {
            (symbol, stream): StreamHealth() for symbol in self.symbols for stream in MANDATORY_STREAMS
        }
        self._client: Optional[BinanceWsClient] = None
        self._task: Optional[asyncio.Task] = None

    def observe(self, event: RawMarketEvent) -> None:
        key = (event.symbol.upper(), event.stream)
        if key not in self._health:
            return
        health = self._health[key]
        health.last_recv_mono_ns = event.recv_mono_ts_ns
        if event.parse_status != "ok":
            health.parse_errors += 1

    async def start(self, client: BinanceWsClient) -> None:
        if self._task is not None:
            raise RuntimeError("StreamSupervisor already started")
        self._client = client
        self._task = asyncio.create_task(client.run_forever(), name="binance-ws-supervisor")

    async def stop(self) -> None:
        if self._client:
            self._client.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def health_snapshot(self) -> SupervisorHealthSnapshot:
        now = time.monotonic_ns()
        unhealthy: List[str] = []

        for (symbol, stream), health in self._health.items():
            if stream in self.sparse_streams:
                continue
            if health.last_recv_mono_ns <= 0:
                unhealthy.append(f"{symbol}:{stream}:no_data")
                continue
            stale_ms = self._stale_overrides.get((symbol, stream), self.stale_after_ms)
            age_ms = (now - health.last_recv_mono_ns) / 1_000_000.0
            if age_ms > stale_ms:
                unhealthy.append(f"{symbol}:{stream}:stale_{int(age_ms)}ms")

        return SupervisorHealthSnapshot(healthy=not unhealthy, unhealthy_reasons=unhealthy)
