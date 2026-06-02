"""OKX V5 collector (perpetual swaps)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..normalization.normalize_funding import normalize_okx_funding
from ..normalization.normalize_prices import normalize_okx_tickers
from ..utils.time import to_ms, to_utc
from .base import CollectorContext, request_json


OKX_FUNDING_HISTORY = "/api/v5/public/funding-rate-history"
OKX_TICKERS = "/api/v5/market/tickers"


class OKXCollector:
    def __init__(
        self,
        ctx: Optional[CollectorContext] = None,
        funding_interval_hours: int = 8,
    ):
        self.ctx = ctx or CollectorContext(
            venue="okx",
            rest_base="https://www.okx.com",
            rate_limit_rps=10,
        )
        self.funding_interval_hours = funding_interval_hours

    async def fetch_funding_history(
        self,
        inst_id: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        page_size: int = 100,
    ) -> pd.DataFrame:
        """OKX paginates with `before`/`after` timestamps (ms strings)."""
        start_ms = to_ms(start)
        end_ms = to_ms(end)
        rows: list[dict] = []
        after = end_ms
        while True:
            params = {
                "instId": inst_id,
                "before": str(start_ms),
                "after": str(after),
                "limit": str(page_size),
            }
            payload = await request_json(self.ctx, "GET", OKX_FUNDING_HISTORY, params=params)
            if not payload or payload.get("code") not in ("0", 0):
                if payload:
                    self.ctx.logger.warning("okx funding non-zero code: %s", payload.get("msg"))
                break
            data = payload.get("data") or []
            if not data:
                break
            rows.extend(data)
            oldest = int(data[-1]["fundingTime"])
            if oldest <= start_ms or len(data) < page_size:
                break
            after = oldest - 1

        df = normalize_okx_funding(
            {"data": rows},
            issues=self.ctx.issues,
            funding_interval_hours=self.funding_interval_hours,
        )
        self.ctx.logger.info(
            "okx funding %s: %d rows (%s -> %s)",
            inst_id, len(df), to_utc(start).isoformat(), to_utc(end).isoformat(),
        )
        return df

    async def fetch_tickers(self, inst_type: str = "SWAP") -> pd.DataFrame:
        payload = await request_json(
            self.ctx, "GET", OKX_TICKERS, params={"instType": inst_type}
        )
        return normalize_okx_tickers(payload, issues=self.ctx.issues)
