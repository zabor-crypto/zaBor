"""Order-book snapshot synchronization helpers for Binance UM futures."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Optional, Sequence, Tuple

from v1._fapi import next_base, safe_urlopen

_DEPTH_PATH = "/fapi/v1/depth"


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    last_update_id: int
    bids: List[Tuple[Decimal, Decimal]]
    asks: List[Tuple[Decimal, Decimal]]
    exchange_ts_ms: int


class BinanceSnapshotClient:
    """Small REST snapshot client to recover from depth desyncs."""

    def __init__(
        self,
        *,
        endpoint: str = "https://fapi.binance.com/fapi/v1/depth",
        timeout_seconds: float = 5.0,
        http_get: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.http_get = http_get

    def _blocking_fetch(self, symbol: str, limit: int) -> OrderBookSnapshot:
        query = urllib.parse.urlencode({"symbol": symbol, "limit": limit})

        if self.http_get is not None:
            url = f"{self.endpoint}?{query}"
            body = self.http_get(url)
        else:
            url = f"{next_base()}{_DEPTH_PATH}?{query}"
            req = urllib.request.Request(url)
            with safe_urlopen(req, self.timeout_seconds) as response:
                body = response.read().decode("utf-8")

        payload = json.loads(body)

        bids = [
            (Decimal(str(px)), Decimal(str(qty)))
            for px, qty in payload.get("bids", [])
            if Decimal(str(qty)) > 0
        ]
        asks = [
            (Decimal(str(px)), Decimal(str(qty)))
            for px, qty in payload.get("asks", [])
            if Decimal(str(qty)) > 0
        ]

        return OrderBookSnapshot(
            symbol=symbol,
            last_update_id=int(payload["lastUpdateId"]),
            bids=bids,
            asks=asks,
            exchange_ts_ms=int(payload.get("E", 0) or 0),
        )

    async def fetch_snapshot(self, symbol: str, limit: int = 1000) -> OrderBookSnapshot:
        return await asyncio.to_thread(self._blocking_fetch, symbol, limit)
