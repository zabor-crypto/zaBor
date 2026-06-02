"""Binance Futures mark/funding REST poller.

The Binance Futures WS gateway from AWS Tokyo silently drops
``@markPrice`` (verified 2026-04-27 — connection accepts subscribe,
server pushes nothing). This poller covers the gap: hits
``GET /fapi/v1/premiumIndex`` (no symbol param → all symbols) every
``poll_interval_s`` and writes one row per (configured) coin to the
``mark`` channel under ``venue=binance``.

Schema matches the WS-mark output of ``venue_clients/binance.py`` so
downstream code can union the two sources transparently:

  ``timestamp_utc, recv_ts_utc, venue, symbol, mark_price, index_price,
    funding_rate, next_funding_time, est_settle_price``

Polling cadence: 10s by default. Funding rate updates only at the
exchange's ~5s premium-index cadence, so 10s is plenty. The endpoint
costs 1 weight per call regardless of whether ``symbol`` is set, so
all-symbols polling is the cheap option (one call covers our whole
universe).

References:
  - https://binance-docs.github.io/apidocs/futures/en/#mark-price
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pandas as pd

from ..writer import ParquetWriter
from ...collectors.base import CollectorContext, request_json
from ...utils.logging import get_logger

# Re-use the Binance Futures 1000-prefix overrides from the WS client.
from ..venue_clients.binance import _BASE_PREFIX_OVERRIDES, _binance_base


@dataclass
class BinanceMarkPollerConfig:
    coins: list[str]                                       # canonical bases ("BTC","PEPE",...)
    poll_interval_s: float = 10.0
    rest_base: str = "https://fapi.binance.com"
    rate_limit_rps: float = 10.0


class BinanceMarkPoller:
    """Long-lived async poller for Binance Futures mark/funding."""

    name = "binance_mark_rest"

    def __init__(self, cfg: BinanceMarkPollerConfig, writer: ParquetWriter) -> None:
        self.cfg = cfg
        self.writer = writer
        self.log = get_logger("funding_arb.recorder.binance_mark_rest")
        self.msgs_received = 0          # parity with VenueClient stat field
        self.last_msg_ts: float = 0.0
        # Pre-compute the set of Binance Futures symbols we care about
        # so we can filter the all-symbols payload cheaply.
        self._wanted_symbols = {
            f"{_binance_base(c)}USDT": c.upper() for c in cfg.coins
        }
        # Compatibility shim so RecorderSupervisor's stats loop can read .cfg.venue
        self.cfg.venue = "binance_mark_rest"  # type: ignore[attr-defined]

    async def run(self) -> None:
        backoff = 2.0
        while True:
            try:
                async with CollectorContext(
                    venue="binance",
                    rest_base=self.cfg.rest_base,
                    rate_limit_rps=self.cfg.rate_limit_rps,
                ) as ctx:
                    while True:
                        await self._tick(ctx)
                        await asyncio.sleep(self.cfg.poll_interval_s)
                backoff = 2.0
            except asyncio.CancelledError:
                self.log.info("cancelled, exiting")
                raise
            except Exception:  # noqa: BLE001
                self.log.exception("poll loop error; reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _tick(self, ctx: CollectorContext) -> None:
        payload = await request_json(ctx, "GET", "/fapi/v1/premiumIndex")
        if not isinstance(payload, list):
            return
        recv = time.time()
        recv_ts = pd.Timestamp(recv, unit="s", tz="UTC")
        n = 0
        for r in payload:
            sym = r.get("symbol")
            coin = self._wanted_symbols.get(sym)
            if coin is None:
                continue
            try:
                row = {
                    "timestamp_utc": pd.Timestamp(int(r["time"]), unit="ms", tz="UTC"),
                    "recv_ts_utc": recv_ts,
                    "venue": "binance",
                    "symbol": sym,
                    "mark_price": _f(r.get("markPrice")),
                    "index_price": _f(r.get("indexPrice")),
                    "est_settle_price": _f(r.get("estimatedSettlePrice")),
                    "funding_rate": _f(r.get("lastFundingRate")),
                    "next_funding_time": _to_ts(r.get("nextFundingTime")),
                    "interest_rate": _f(r.get("interestRate")),
                    "source": "binance/premiumIndex",
                }
            except (KeyError, ValueError, TypeError) as e:
                self.log.warning("schema issue for %s: %s", sym, e)
                continue
            self.writer.put("mark", "binance", coin, row)
            n += 1
        self.msgs_received += n
        self.last_msg_ts = recv


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_ts(v) -> pd.Timestamp | None:
    if v is None or v in ("", "0", 0):
        return None
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return None
