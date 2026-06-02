"""Bybit V5 collector (linear perps)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..normalization.normalize_funding import normalize_bybit_funding
from ..normalization.normalize_prices import normalize_bybit_tickers
from ..utils.time import to_ms, to_utc
from .base import CollectorContext, request_json


BYBIT_FUNDING_PATH = "/v5/market/funding/history"
BYBIT_TICKERS_PATH = "/v5/market/tickers"


class BybitCollector:
    def __init__(
        self,
        ctx: Optional[CollectorContext] = None,
        category: str = "linear",
        funding_interval_hours: int = 8,
    ):
        self.ctx = ctx or CollectorContext(
            venue="bybit",
            rest_base="https://api.bybit.com",
            rate_limit_rps=10,
        )
        self.category = category
        self.funding_interval_hours = funding_interval_hours

    async def fetch_funding_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        page_size: int = 200,
    ) -> pd.DataFrame:
        """Bybit returns up to 200 records per call, descending. We page backwards."""
        start_ms = to_ms(start)
        end_ms = to_ms(end)
        all_items: list[dict] = []
        cursor_end = end_ms
        while True:
            params = {
                "category": self.category,
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": cursor_end,
                "limit": page_size,
            }
            payload = await request_json(self.ctx, "GET", BYBIT_FUNDING_PATH, params=params)
            if not payload or payload.get("retCode") != 0:
                if payload:
                    self.ctx.logger.warning("bybit funding non-zero retCode: %s", payload.get("retMsg"))
                break
            items = (payload.get("result") or {}).get("list") or []
            if not items:
                break
            all_items.extend(items)
            oldest = int(items[-1]["fundingRateTimestamp"])
            if oldest <= start_ms or len(items) < page_size:
                break
            cursor_end = oldest - 1

        df = normalize_bybit_funding(
            {"result": {"list": all_items}},
            issues=self.ctx.issues,
            funding_interval_hours=self.funding_interval_hours,
        )
        self.ctx.logger.info(
            "bybit funding %s: %d rows (%s -> %s)",
            symbol, len(df), to_utc(start).isoformat(), to_utc(end).isoformat(),
        )
        return df

    async def fetch_tickers(self, symbol: Optional[str] = None) -> pd.DataFrame:
        params: dict = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        payload = await request_json(self.ctx, "GET", BYBIT_TICKERS_PATH, params=params)
        return normalize_bybit_tickers(payload, issues=self.ctx.issues)
