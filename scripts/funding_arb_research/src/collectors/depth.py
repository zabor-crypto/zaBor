"""Top-of-book + L2 depth snapshots for each CEX.

A depth snapshot is normalized to:

    DepthSnapshot(
        venue, symbol, timestamp_utc,
        bids=[(price, qty), ...],   # qty in *base* asset
        asks=[(price, qty), ...],
    )

The ``depth_notional_usd(side, depth)`` helper turns one side of a snapshot
into the cumulative notional liquidity available within a price band — the
cost model uses this to size slippage realistically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from ..utils.io import hash_payload
from ..utils.time import to_utc
from .base import CollectorContext, request_json


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DepthSnapshot:
    venue: str
    symbol: str
    timestamp_utc: pd.Timestamp
    bids: list[tuple[float, float]]    # [(price, qty)], qty in base
    asks: list[tuple[float, float]]
    raw_hash: str

    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2.0


def depth_notional_usd(
    levels: Iterable[tuple[float, float]],
    *,
    band_bps: float = 10.0,
    reference_price: Optional[float] = None,
) -> float:
    """Sum notional (USD) on one side of the book within ``band_bps`` of the
    inside price.

    ``band_bps = 10`` means we only count liquidity within 10 bps of the
    top-of-book — a reasonable proxy for what a marketable taker actually
    sweeps. ``reference_price`` defaults to the best price on this side.
    """
    levels_list = list(levels)
    if not levels_list:
        return 0.0
    ref = reference_price if reference_price is not None else levels_list[0][0]
    if ref <= 0:
        return 0.0
    band = ref * band_bps * 1e-4
    total = 0.0
    for px, qty in levels_list:
        if abs(px - ref) > band:
            break
        total += float(px) * float(qty)
    return total


# --------------------------------------------------------------------------- #
# Per-venue parsers
# --------------------------------------------------------------------------- #

def _parse_pairs(raw_pairs) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not raw_pairs:
        return out
    for r in raw_pairs:
        try:
            px = float(r[0])
            qty = float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if px <= 0 or qty <= 0:
            continue
        out.append((px, qty))
    return out


def parse_binance_depth(payload: dict, symbol: str) -> DepthSnapshot:
    """Binance ``/fapi/v1/depth`` returns ``{lastUpdateId, E, T, bids, asks}``."""
    ts_ms = payload.get("T") or payload.get("E")
    ts = to_utc(int(ts_ms)) if ts_ms else to_utc(pd.Timestamp.utcnow())
    return DepthSnapshot(
        venue="binance", symbol=symbol, timestamp_utc=ts,
        bids=_parse_pairs(payload.get("bids")),
        asks=_parse_pairs(payload.get("asks")),
        raw_hash=hash_payload(payload),
    )


def parse_bybit_depth(payload: dict, symbol: str) -> DepthSnapshot:
    result = (payload or {}).get("result") or {}
    ts_ms = result.get("ts") or result.get("u")
    ts = to_utc(int(ts_ms)) if ts_ms else to_utc(pd.Timestamp.utcnow())
    return DepthSnapshot(
        venue="bybit", symbol=symbol, timestamp_utc=ts,
        bids=_parse_pairs(result.get("b")),
        asks=_parse_pairs(result.get("a")),
        raw_hash=hash_payload(payload),
    )


def parse_okx_depth(payload: dict, symbol: str) -> DepthSnapshot:
    """OKX returns ``{code, data: [{ts, bids, asks}]}`` where each level is
    ``[price, size, _liq, _orderCount]``."""
    data = (payload or {}).get("data") or []
    if not data:
        return DepthSnapshot("okx", symbol, to_utc(pd.Timestamp.utcnow()),
                              [], [], hash_payload(payload))
    book = data[0]
    ts_ms = book.get("ts")
    ts = to_utc(int(ts_ms)) if ts_ms else to_utc(pd.Timestamp.utcnow())
    return DepthSnapshot(
        venue="okx", symbol=symbol, timestamp_utc=ts,
        bids=_parse_pairs(book.get("bids")),
        asks=_parse_pairs(book.get("asks")),
        raw_hash=hash_payload(payload),
    )


def parse_hyperliquid_depth(payload: dict, coin: str) -> DepthSnapshot:
    """HL ``l2Book`` returns ``{coin, time, levels: [bids, asks]}`` where
    each level is ``{px, sz, n}``."""
    if not isinstance(payload, dict):
        return DepthSnapshot("hyperliquid", coin, to_utc(pd.Timestamp.utcnow()),
                              [], [], hash_payload(payload))
    levels = payload.get("levels") or [[], []]
    bids_raw = levels[0] if len(levels) > 0 else []
    asks_raw = levels[1] if len(levels) > 1 else []

    def _hl_pairs(rows) -> list[tuple[float, float]]:
        out = []
        for r in rows or []:
            try:
                out.append((float(r["px"]), float(r["sz"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    ts_ms = payload.get("time")
    ts = to_utc(int(ts_ms)) if ts_ms else to_utc(pd.Timestamp.utcnow())
    return DepthSnapshot(
        venue="hyperliquid", symbol=coin, timestamp_utc=ts,
        bids=_hl_pairs(bids_raw), asks=_hl_pairs(asks_raw),
        raw_hash=hash_payload(payload),
    )


# --------------------------------------------------------------------------- #
# Collector facade
# --------------------------------------------------------------------------- #

class DepthCollector:
    """Multi-venue depth collector.

    Usage:
        async with DepthCollector() as dc:
            snap = await dc.fetch("binance", "BTCUSDT", limit=100)
    """

    DEFAULT_BASES: dict[str, str] = {
        "binance": "https://fapi.binance.com",
        "bybit": "https://api.bybit.com",
        "okx": "https://www.okx.com",
        "hyperliquid": "https://api.hyperliquid.xyz",
    }

    def __init__(self, bases: Optional[dict[str, str]] = None):
        self._bases = bases or self.DEFAULT_BASES
        self._ctxs: dict[str, CollectorContext] = {}

    async def __aenter__(self):
        for venue, base in self._bases.items():
            ctx = CollectorContext(
                venue=venue, rest_base=base,
                rate_limit_rps=5,
            )
            await ctx.__aenter__()
            self._ctxs[venue] = ctx
        return self

    async def __aexit__(self, *exc):
        for ctx in self._ctxs.values():
            await ctx.__aexit__(*exc)
        self._ctxs.clear()

    async def fetch(self, venue: str, symbol: str, *, limit: int = 50) -> DepthSnapshot:
        ctx = self._ctxs[venue]
        if venue == "binance":
            payload = await request_json(ctx, "GET", "/fapi/v1/depth",
                                         params={"symbol": symbol, "limit": limit})
            return parse_binance_depth(payload, symbol)
        if venue == "bybit":
            payload = await request_json(ctx, "GET", "/v5/market/orderbook",
                                         params={"category": "linear",
                                                 "symbol": symbol,
                                                 "limit": limit})
            return parse_bybit_depth(payload, symbol)
        if venue == "okx":
            payload = await request_json(ctx, "GET", "/api/v5/market/books",
                                         params={"instId": symbol, "sz": str(limit)})
            return parse_okx_depth(payload, symbol)
        if venue == "hyperliquid":
            payload = await request_json(ctx, "POST", "/info",
                                         json_body={"type": "l2Book", "coin": symbol})
            return parse_hyperliquid_depth(payload, symbol)
        raise ValueError(f"unknown venue: {venue}")

    async def fetch_many(
        self, requests: list[tuple[str, str]], *, limit: int = 50,
    ) -> list[DepthSnapshot]:
        out: list[DepthSnapshot] = []
        for venue, sym in requests:
            try:
                out.append(await self.fetch(venue, sym, limit=limit))
            except Exception as e:
                # Log + continue; depth is best-effort.
                ctx = self._ctxs.get(venue)
                if ctx is not None:
                    ctx.logger.warning("depth fetch failed %s/%s: %s", venue, sym, e)
        return out


def to_dataframe(snaps: list[DepthSnapshot], *, band_bps: float = 10.0) -> pd.DataFrame:
    """Flatten a list of snapshots into a parquet-friendly frame:
    one row per (venue, symbol, timestamp_utc) with notional aggregates."""
    rows = []
    for s in snaps:
        rows.append({
            "timestamp_utc": s.timestamp_utc,
            "venue": s.venue, "symbol": s.symbol,
            "best_bid": s.best_bid(), "best_ask": s.best_ask(),
            "mid_price": s.mid(),
            "bid_notional_usd": depth_notional_usd(s.bids, band_bps=band_bps),
            "ask_notional_usd": depth_notional_usd(s.asks, band_bps=band_bps),
            "n_bid_levels": len(s.bids),
            "n_ask_levels": len(s.asks),
            "raw_payload_hash": s.raw_hash,
        })
    return pd.DataFrame(rows)
