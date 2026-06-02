"""Hyperliquid info-API collector.

Hyperliquid uses a single ``/info`` POST endpoint for nearly everything.
Funding is hourly. ``fundingHistory`` is paginated by ``startTime``/``endTime``
in ms; the API returns up to 500 records per call.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..normalization.normalize_funding import (
    normalize_hyperliquid_funding,
    normalize_hyperliquid_predicted,
)
from ..normalization.normalize_prices import normalize_hyperliquid_meta_and_ctxs
from ..utils.time import to_ms, to_utc
from .base import CollectorContext, request_json


HL_INFO_PATH = "/info"


class HyperliquidCollector:
    def __init__(self, ctx: Optional[CollectorContext] = None):
        self.ctx = ctx or CollectorContext(
            venue="hyperliquid",
            rest_base="https://api.hyperliquid.xyz",
            rate_limit_rps=5,
        )

    async def fetch_meta_and_ctxs(self) -> pd.DataFrame:
        body = {"type": "metaAndAssetCtxs"}
        payload = await request_json(self.ctx, "POST", HL_INFO_PATH, json_body=body)
        return normalize_hyperliquid_meta_and_ctxs(payload, issues=self.ctx.issues)

    async def fetch_predicted_fundings(self) -> pd.DataFrame:
        """Hyperliquid ``predictedFundings`` covers predicted rates across
        Hyperliquid itself + the major CEXes. Returns a normalized funding
        frame keyed by venue/symbol with ``predicted_funding_rate`` filled.
        """
        body = {"type": "predictedFundings"}
        payload = await request_json(self.ctx, "POST", HL_INFO_PATH, json_body=body)
        return normalize_hyperliquid_predicted(payload, issues=self.ctx.issues)

    async def fetch_l2_book(self, coin: str, *, n_levels: int = 20) -> dict:
        """Raw L2 book snapshot: ``{coin, time, levels: [bids, asks]}``."""
        body = {"type": "l2Book", "coin": coin, "nSigFigs": None}
        payload = await request_json(self.ctx, "POST", HL_INFO_PATH, json_body=body)
        # Trim to n_levels each side for downstream depth math
        if isinstance(payload, dict) and isinstance(payload.get("levels"), list) \
                and len(payload["levels"]) >= 2:
            payload["levels"] = [
                payload["levels"][0][:n_levels],
                payload["levels"][1][:n_levels],
            ]
        return payload

    async def fetch_funding_history(
        self,
        coin: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        page_size: int = 500,
    ) -> pd.DataFrame:
        start_ms = to_ms(start)
        end_ms = to_ms(end)
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            body = {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": cursor,
                "endTime": end_ms,
            }
            payload = await request_json(self.ctx, "POST", HL_INFO_PATH, json_body=body)
            if not payload:
                break
            rows.extend(payload)
            last_ts = int(payload[-1]["time"])
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            if len(payload) < page_size:
                break

        df = normalize_hyperliquid_funding(
            rows, coin=coin, issues=self.ctx.issues, funding_interval_hours=1,
        )
        self.ctx.logger.info(
            "hyperliquid funding %s: %d rows (%s -> %s)",
            coin, len(df), to_utc(start).isoformat(), to_utc(end).isoformat(),
        )
        return df
