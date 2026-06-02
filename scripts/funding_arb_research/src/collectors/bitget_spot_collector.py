"""Bitget spot collector (public REST).

Bitget's perp Mode-2 (negative funding hedged with spot short) is
disabled per the v1 design — see ``event_funding_capture_v1_design.md``.
Spot data is therefore not load-bearing for the main strategy; we
collect it only as a sanity reference (basis sanity, perp ↔ spot price
divergence, depth/liquidity comparisons). Kept minimal:

  * ``/api/v2/spot/public/symbols``   — symbol metadata
  * ``/api/v2/spot/market/candles``    — OHLCV (1m/5m/15m/1h/4h/1d)
  * ``/api/v2/spot/market/tickers``    — last/bid/ask/24h volume per symbol
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..utils.time import to_ms
from .base import CollectorContext, request_json
from .bitget_collector import _check_v2, _to_float, _to_int, _to_ts_ms

_PATH_SYMBOLS = "/api/v2/spot/public/symbols"
_PATH_CANDLES = "/api/v2/spot/market/candles"
_PATH_TICKERS = "/api/v2/spot/market/tickers"


class BitgetSpotCollector:
    def __init__(self, ctx: Optional[CollectorContext] = None) -> None:
        self.ctx = ctx or CollectorContext(
            venue="bitget", rest_base="https://api.bitget.com", rate_limit_rps=10.0,
        )

    async def fetch_symbols(self) -> pd.DataFrame:
        payload = await request_json(self.ctx, "GET", _PATH_SYMBOLS)
        data = _check_v2(payload, self.ctx.logger, "spot/symbols") or []
        rows = []
        for s in data:
            rows.append({
                "venue": "bitget", "market": "spot",
                "symbol": s.get("symbol"),
                "base_asset": s.get("baseCoin"),
                "quote_asset": s.get("quoteCoin"),
                "min_trade_amount": _to_float(s.get("minTradeAmount")),
                "max_trade_amount": _to_float(s.get("maxTradeAmount")),
                "taker_fee_rate": _to_float(s.get("takerFeeRate")),
                "maker_fee_rate": _to_float(s.get("makerFeeRate")),
                "price_precision": _to_int(s.get("pricePrecision")),
                "quantity_precision": _to_int(s.get("quantityPrecision")),
                "status": s.get("status"),
                "source": "bitget/spot/symbols",
            })
        return pd.DataFrame(rows)

    async def fetch_candles(
        self, symbol: str, granularity: str = "1h",
        start: datetime | pd.Timestamp | None = None,
        end: datetime | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """OHLCV candles. ``granularity`` ∈ {1min,5min,15min,30min,1h,4h,6h,12h,1day,1week}."""
        params: dict[str, Any] = {"symbol": symbol, "granularity": granularity, "limit": limit}
        if start is not None:
            params["startTime"] = to_ms(start)
        if end is not None:
            params["endTime"] = to_ms(end)
        payload = await request_json(self.ctx, "GET", _PATH_CANDLES, params=params)
        data = _check_v2(payload, self.ctx.logger, "spot/candles") or []
        rows = []
        for r in data:
            # Bitget v2 spot candle: [ts, open, high, low, close, baseVol, quoteVol, usdtVol]
            try:
                rows.append({
                    "timestamp_utc": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                    "venue": "bitget", "market": "spot", "symbol": symbol,
                    "open": float(r[1]), "high": float(r[2]),
                    "low": float(r[3]), "close": float(r[4]),
                    "base_volume": float(r[5]),
                    "quote_volume": float(r[6]) if len(r) > 6 else None,
                    "usdt_volume": float(r[7]) if len(r) > 7 else None,
                    "source": "bitget/spot/candles",
                })
            except (IndexError, ValueError, TypeError):
                continue
        return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True) if rows else pd.DataFrame()

    async def fetch_tickers(self, symbol: Optional[str] = None) -> pd.DataFrame:
        params = {"symbol": symbol} if symbol else None
        payload = await request_json(self.ctx, "GET", _PATH_TICKERS, params=params)
        data = _check_v2(payload, self.ctx.logger, "spot/tickers") or []
        if isinstance(data, dict):
            data = [data]
        rows = []
        for t in data:
            rows.append({
                "venue": "bitget", "market": "spot",
                "symbol": t.get("symbol"),
                "last": _to_float(t.get("lastPr")),
                "best_bid": _to_float(t.get("bidPr")),
                "best_ask": _to_float(t.get("askPr")),
                "bid_size": _to_float(t.get("bidSz")),
                "ask_size": _to_float(t.get("askSz")),
                "high_24h": _to_float(t.get("high24h")),
                "low_24h": _to_float(t.get("low24h")),
                "base_volume_24h": _to_float(t.get("baseVolume")),
                "quote_volume_24h": _to_float(t.get("quoteVolume")),
                "ts": _to_ts_ms(t.get("ts")),
                "source": "bitget/spot/tickers",
            })
        return pd.DataFrame(rows)
