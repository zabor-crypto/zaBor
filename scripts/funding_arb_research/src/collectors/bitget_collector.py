"""Bitget USDT-M futures collector.

Public REST endpoints (Bitget v2, base ``https://api.bitget.com``); no
API key required for funding/market data. ``productType`` is fixed at
``USDT-FUTURES`` (linear perp, USDT-margined) — Bitget v2's term for
what other venues call "linear perp".

Endpoints used (all public):

  * ``/api/v2/mix/market/contracts``           — instrument metadata,
                                                  including ``fundInterval``
                                                  per symbol (the funding
                                                  cadence in hours)
  * ``/api/v2/mix/market/history-fund-rate``   — paginated funding history
  * ``/api/v2/mix/market/current-fund-rate``   — current/in-progress funding
  * ``/api/v2/mix/market/funding-time``        — next funding settle ts
  * ``/api/v2/mix/market/ticker``              — mark/index/last/OI in one
  * ``/api/v2/mix/market/open-interest``       — explicit OI snapshot
  * ``/api/v2/mix/market/query-position-lever``— per-symbol MMR tier ladder

The MMR tier table is the load-bearing input for the realism budget's
maintenance-margin requirement (Bitget hedge mode); we never approximate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..utils.time import to_ms, to_utc
from .base import CollectorContext, request_json


_BITGET_BASE = "https://api.bitget.com"
_PRODUCT_TYPE = "USDT-FUTURES"

# Endpoint paths
_PATH_CONTRACTS = "/api/v2/mix/market/contracts"
_PATH_FUNDING_HIST = "/api/v2/mix/market/history-fund-rate"
_PATH_CURRENT_FUNDING = "/api/v2/mix/market/current-fund-rate"
_PATH_FUNDING_TIME = "/api/v2/mix/market/funding-time"
_PATH_TICKER = "/api/v2/mix/market/ticker"
_PATH_OPEN_INTEREST = "/api/v2/mix/market/open-interest"
_PATH_MMR_LEVER = "/api/v2/mix/market/query-position-lever"


def _check_v2(payload: dict, ctx_logger, endpoint: str) -> list[dict] | dict | None:
    """Return ``payload['data']`` after asserting ``code == "00000"``."""
    if not isinstance(payload, dict):
        ctx_logger.warning("bitget %s: non-dict response %.200s", endpoint, str(payload))
        return None
    code = payload.get("code")
    if code not in ("00000", 0):
        ctx_logger.error("bitget %s: code=%s msg=%s", endpoint, code, payload.get("msg"))
        return None
    return payload.get("data")


class BitgetCollector:
    """Public REST collector for Bitget USDT-M futures."""

    def __init__(self, ctx: Optional[CollectorContext] = None) -> None:
        self.ctx = ctx or CollectorContext(
            venue="bitget",
            rest_base=_BITGET_BASE,
            rate_limit_rps=10.0,
        )

    # ------------------------------------------------------------------ #
    # Contracts / metadata
    # ------------------------------------------------------------------ #

    async def fetch_contracts(self) -> pd.DataFrame:
        """All USDT-FUTURES contracts. Includes ``fundInterval`` (hours)."""
        payload = await request_json(
            self.ctx, "GET", _PATH_CONTRACTS,
            params={"productType": _PRODUCT_TYPE},
        )
        data = _check_v2(payload, self.ctx.logger, "contracts") or []
        rows = []
        for c in data:
            try:
                rows.append({
                    "venue": "bitget",
                    "symbol": c.get("symbol"),
                    "base_asset": c.get("baseCoin"),
                    "quote_asset": c.get("quoteCoin"),
                    "instrument_type": "perp",
                    "fund_interval_hours": _to_int(c.get("fundInterval")),
                    "min_lever": _to_float(c.get("minLever")),
                    "max_lever": _to_float(c.get("maxLever")),
                    "maker_fee_rate": _to_float(c.get("makerFeeRate")),
                    "taker_fee_rate": _to_float(c.get("takerFeeRate")),
                    "min_trade_num": _to_float(c.get("minTradeNum")),
                    "min_trade_usdt": _to_float(c.get("minTradeUSDT")),
                    "size_multiplier": _to_float(c.get("sizeMultiplier")),
                    "price_place": _to_int(c.get("pricePlace")),
                    "volume_place": _to_int(c.get("volumePlace")),
                    "symbol_status": c.get("symbolStatus"),
                    "launch_time": _to_ts_ms(c.get("launchTime")),
                    "delivery_time": _to_ts_ms(c.get("deliveryTime")),
                    "off_time": _to_ts_ms(c.get("offTime")),
                    "source": "bitget/contracts",
                })
            except Exception:  # noqa: BLE001 — never let a single bad row break collection
                self.ctx.logger.exception("contracts row parse failed: %.200s", str(c))
        df = pd.DataFrame(rows)
        self.ctx.logger.info("bitget contracts: %d rows", len(df))
        return df

    # ------------------------------------------------------------------ #
    # Funding
    # ------------------------------------------------------------------ #

    async def fetch_funding_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        *,
        page_size: int = 100,
    ) -> pd.DataFrame:
        """Paginate ``history-fund-rate`` from ``end`` backwards until ``start``.

        Bitget's response is ordered newest-first and uses 1-based ``pageNo``.
        We paginate until either:
          - the oldest row in the page is ≤ ``start_ms``, or
          - the page returns < ``page_size`` rows (end of history).
        """
        start_ms, end_ms = to_ms(start), to_ms(end)
        rows: list[dict] = []
        page_no = 1
        while True:
            payload = await request_json(
                self.ctx, "GET", _PATH_FUNDING_HIST,
                params={
                    "symbol": symbol,
                    "productType": _PRODUCT_TYPE,
                    "pageSize": page_size,
                    "pageNo": page_no,
                },
            )
            data = _check_v2(payload, self.ctx.logger, "history-fund-rate") or []
            if not data:
                break
            rows.extend(data)
            oldest_ms = min(int(r["fundingTime"]) for r in data if r.get("fundingTime"))
            if oldest_ms <= start_ms or len(data) < page_size:
                break
            page_no += 1
            if page_no > 200:  # safety: ~20k rows is plenty
                self.ctx.logger.warning("bitget funding pagination cap hit at pageNo=%d", page_no)
                break

        # Filter to the requested window and normalize.
        df = _normalize_bitget_funding_history(rows, symbol, start_ms, end_ms)
        self.ctx.logger.info(
            "bitget funding %s: %d rows in [%s, %s]",
            symbol, len(df), to_utc(start).isoformat(), to_utc(end).isoformat(),
        )
        return df

    async def fetch_current_funding(self, symbol: str) -> dict[str, Any]:
        """Current/in-progress funding rate (predicted-funding proxy)."""
        payload = await request_json(
            self.ctx, "GET", _PATH_CURRENT_FUNDING,
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
        )
        data = _check_v2(payload, self.ctx.logger, "current-fund-rate") or []
        if not data:
            return {}
        row = data[0]
        return {
            "symbol": row.get("symbol"),
            "funding_rate": _to_float(row.get("fundingRate")),
            "venue": "bitget",
            "source": "bitget/current-fund-rate",
        }

    async def fetch_funding_time(self, symbol: str) -> dict[str, Any]:
        """Returns next funding settle timestamp + cadence."""
        payload = await request_json(
            self.ctx, "GET", _PATH_FUNDING_TIME,
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
        )
        data = _check_v2(payload, self.ctx.logger, "funding-time") or []
        if not data:
            return {}
        row = data[0]
        return {
            "symbol": row.get("symbol"),
            "next_funding_time": _to_ts_ms(row.get("nextFundingTime")),
            "rate_period_hours": _to_int(row.get("ratePeriod")),
        }

    # ------------------------------------------------------------------ #
    # Mark / OI
    # ------------------------------------------------------------------ #

    async def fetch_ticker(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Snapshot ticker. ``symbol=None`` returns all USDT-FUTURES tickers."""
        params: dict[str, Any] = {"productType": _PRODUCT_TYPE}
        if symbol:
            params["symbol"] = symbol
        payload = await request_json(self.ctx, "GET", _PATH_TICKER, params=params)
        data = _check_v2(payload, self.ctx.logger, "ticker") or []
        if isinstance(data, dict):
            data = [data]
        rows = []
        for t in data:
            rows.append({
                "venue": "bitget",
                "symbol": t.get("symbol"),
                "mark_price": _to_float(t.get("markPrice")),
                "index_price": _to_float(t.get("indexPrice")),
                "last_price": _to_float(t.get("lastPr")),
                "funding_rate": _to_float(t.get("fundingRate")),
                "next_funding_time": _to_ts_ms(t.get("nextFundingTime")),
                "open_interest_base": _to_float(t.get("holdingAmount")),
                "quote_volume_24h": _to_float(t.get("quoteVolume")),
                "ts": _to_ts_ms(t.get("ts")),
                "source": "bitget/ticker",
            })
        return pd.DataFrame(rows)

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        payload = await request_json(
            self.ctx, "GET", _PATH_OPEN_INTEREST,
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
        )
        data = _check_v2(payload, self.ctx.logger, "open-interest")
        if not data:
            return {}
        # data shape: {"openInterestList":[{"symbol","size"}], "ts": ...}
        items = data.get("openInterestList") if isinstance(data, dict) else None
        if not items:
            return {}
        row = items[0]
        return {
            "symbol": row.get("symbol"),
            "open_interest_base": _to_float(row.get("size")),
            "ts": _to_ts_ms(data.get("ts")),
            "source": "bitget/open-interest",
        }

    # ------------------------------------------------------------------ #
    # MMR tier ladder
    # ------------------------------------------------------------------ #

    async def fetch_mmr_tiers(self, symbol: str) -> pd.DataFrame:
        """Per-symbol leverage tier ladder with maintenance-margin rate.

        Schema returned by Bitget v2 ``query-position-lever``:
          ``{level, startUnit, endUnit, leverage, keepMarginRate}``

        ``keepMarginRate`` is the maintenance-margin rate at that tier
        (decimal, not percent). This is what the engine's Bitget MMR
        model consumes — do not approximate.
        """
        payload = await request_json(
            self.ctx, "GET", _PATH_MMR_LEVER,
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
        )
        data = _check_v2(payload, self.ctx.logger, "query-position-lever") or []
        rows = []
        for tier in data:
            rows.append({
                "venue": "bitget",
                "symbol": symbol,
                "level": _to_int(tier.get("level")),
                "start_unit": _to_float(tier.get("startUnit")),
                "end_unit": _to_float(tier.get("endUnit")),
                "max_leverage": _to_float(tier.get("leverage")),
                "mmr": _to_float(tier.get("keepMarginRate")),
                "source": "bitget/query-position-lever",
            })
        df = pd.DataFrame(rows).sort_values("level").reset_index(drop=True) if rows else pd.DataFrame()
        return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _to_ts_ms(v: Any) -> pd.Timestamp | None:
    if v is None or v == "" or v == "0":
        return None
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return None


def _normalize_bitget_funding_history(
    raw: list[dict],
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Bitget history-fund-rate row → canonical funding panel row.

    Bitget rows have ``{symbol, fundingRate, fundingTime}``. We emit the
    canonical funding-panel columns and let ``conform_funding`` upstream
    cast/reindex if needed (this collector returns its own minimal frame
    so the existing collector pipeline stays opt-in for Bitget).
    """
    if not raw:
        return pd.DataFrame()
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    rows = []
    for r in raw:
        try:
            t_ms = int(r["fundingTime"])
            if t_ms < start_ms or t_ms > end_ms:
                continue
            rows.append({
                "timestamp_utc": pd.Timestamp(t_ms, unit="ms", tz="UTC"),
                "venue": "bitget",
                "symbol": symbol,
                "base_asset": base,
                "quote_asset": "USDT",
                "instrument_type": "perp",
                "funding_rate": float(r["fundingRate"]),
                "source": "bitget/history-fund-rate",
            })
        except (KeyError, ValueError, TypeError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["timestamp_utc", "symbol"]).sort_values("timestamp_utc").reset_index(drop=True)
