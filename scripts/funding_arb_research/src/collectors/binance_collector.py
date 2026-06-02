"""Binance USD-M futures collector.

Public REST endpoints only; no API key required for funding/price data.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..normalization.normalize_funding import (
    normalize_binance_funding,
    normalize_binance_premium_index,
)
from ..normalization.normalize_prices import normalize_binance_book_ticker
from ..utils.io import write_parquet
from ..utils.time import to_ms, to_utc
from .base import CollectorContext, request_json


BINANCE_FUNDING_PATH = "/fapi/v1/fundingRate"
BINANCE_BOOK_TICKER_PATH = "/fapi/v1/ticker/bookTicker"
BINANCE_PREMIUM_INDEX_PATH = "/fapi/v1/premiumIndex"
BINANCE_DEPTH_PATH = "/fapi/v1/depth"


class BinanceCollector:
    """Fetches funding history and book tickers for one or more symbols."""

    def __init__(
        self,
        ctx: Optional[CollectorContext] = None,
        funding_interval_hours: int = 8,
    ):
        self.ctx = ctx or CollectorContext(
            venue="binance",
            rest_base="https://fapi.binance.com",
            rate_limit_rps=10,
        )
        self.funding_interval_hours = funding_interval_hours

    async def fetch_funding_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        page_size: int = 1000,
    ) -> pd.DataFrame:
        """Page through funding history for ``symbol`` between ``start`` and ``end``."""
        start_ms = to_ms(start)
        end_ms = to_ms(end)
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": page_size,
            }
            payload = await request_json(self.ctx, "GET", BINANCE_FUNDING_PATH, params=params)
            if not payload:
                break
            rows.extend(payload)
            last_ts = int(payload[-1]["fundingTime"])
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            if len(payload) < page_size:
                break

        df = normalize_binance_funding(
            rows,
            issues=self.ctx.issues,
            funding_interval_hours=self.funding_interval_hours,
        )
        self.ctx.logger.info(
            "binance funding %s: %d rows (%s -> %s)",
            symbol, len(df),
            to_utc(start).isoformat(), to_utc(end).isoformat(),
        )
        return df

    async def fetch_book_ticker(self, symbol: Optional[str] = None) -> pd.DataFrame:
        params = {"symbol": symbol} if symbol else None
        payload = await request_json(self.ctx, "GET", BINANCE_BOOK_TICKER_PATH, params=params)
        return normalize_binance_book_ticker(payload, issues=self.ctx.issues)

    async def fetch_premium_index(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Snapshot of premiumIndex (mark, index, lastFundingRate, ...).

        ``lastFundingRate`` here is the live in-progress funding accrual,
        which is the best predicted-funding proxy Binance exposes.
        """
        params = {"symbol": symbol} if symbol else None
        payload = await request_json(self.ctx, "GET", BINANCE_PREMIUM_INDEX_PATH, params=params)
        return normalize_binance_premium_index(
            payload, issues=self.ctx.issues, funding_interval_hours=self.funding_interval_hours,
        )

    async def fetch_depth(
        self, symbol: str, *, limit: int = 100,
    ) -> dict:
        """Raw L2 depth payload: ``{lastUpdateId, bids:[[px,qty]...], asks:[...]}``."""
        return await request_json(
            self.ctx, "GET", BINANCE_DEPTH_PATH,
            params={"symbol": symbol, "limit": limit},
        )

    async def collect_to_parquet(
        self,
        symbols: list[str],
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        out_dir: Path,
    ) -> Path:
        """Convenience: fetch + write a single parquet for ``symbols``."""
        frames = []
        for s in symbols:
            frames.append(await self.fetch_funding_history(s, start, end))
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else frames[0]
        out_path = out_dir / f"binance_funding_{to_ms(start)}_{to_ms(end)}.parquet"
        write_parquet(df, out_path)
        return out_path
