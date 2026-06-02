"""Per-symbol funding-cadence registry.

Different perps settle funding at different cadences (8h, 4h, 2h, 1h).
The engine uses this map to know **when** to credit funding pnl per
leg — getting it wrong by a factor of 2 doubles or halves expected
APR. Sources of truth, by venue:

  Bitget:     ``/api/v2/mix/market/contracts``  → ``fundInterval`` (hours)
  Bybit:      ``/v5/market/instruments-info``    → ``fundingInterval`` (minutes)
  Binance:    documented default 8h; the ``fundingIntervalHours`` field in
              ``exchangeInfo`` is ``null`` for every symbol as of 2026-04, so
              we cannot trust it. Treat as 8h unless evidence otherwise.
  OKX:        documented default 8h; some assets moved to 4h. Inferring
              cleanly from ``/api/v5/public/funding-rate`` (``nextFundingTime``
              minus ``fundingTime``) is correct but requires per-symbol
              probes — deferred.
  Hyperliquid: 1h, venue-uniform.

Output: ``data/static/funding_cadence.csv`` with columns
``venue, symbol, base_asset, fund_interval_hours, source``.

The trigger code joins on ``(venue, base_asset)`` and the cost model
uses ``fund_interval_hours`` to convert between per-period and per-year
funding rates.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from ..utils.io import write_parquet
from ..utils.logging import get_logger
from .base import CollectorContext, request_json
from .bitget_collector import BitgetCollector

_LOG = get_logger("funding_arb.collectors.funding_cadence")


async def _bitget_cadence() -> pd.DataFrame:
    async with CollectorContext(venue="bitget", rest_base="https://api.bitget.com",
                                 rate_limit_rps=10.0) as ctx:
        df = await BitgetCollector(ctx=ctx).fetch_contracts()
    if df.empty:
        return df
    out = df[["symbol", "base_asset", "fund_interval_hours"]].assign(
        venue="bitget", source="bitget/contracts.fundInterval",
    )
    return out[["venue", "symbol", "base_asset", "fund_interval_hours", "source"]]


async def _bybit_cadence() -> pd.DataFrame:
    async with CollectorContext(venue="bybit", rest_base="https://api.bybit.com",
                                 rate_limit_rps=10.0) as ctx:
        rows = []
        cursor = None
        while True:
            params = {"category": "linear", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = await request_json(ctx, "GET", "/v5/market/instruments-info", params=params)
            result = (payload or {}).get("result") or {}
            for inst in result.get("list") or []:
                if inst.get("contractType") != "LinearPerpetual":
                    continue
                fi_min = inst.get("fundingInterval")
                hours = int(fi_min) // 60 if fi_min not in (None, "", 0) else None
                rows.append({
                    "venue": "bybit",
                    "symbol": inst.get("symbol"),
                    "base_asset": inst.get("baseCoin"),
                    "fund_interval_hours": hours,
                    "source": "bybit/instruments-info.fundingInterval",
                })
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
    return pd.DataFrame(rows)


async def _binance_cadence() -> pd.DataFrame:
    """Default 8h per documented behavior. ``fundingIntervalHours`` field is null."""
    async with CollectorContext(venue="binance", rest_base="https://fapi.binance.com",
                                 rate_limit_rps=10.0) as ctx:
        payload = await request_json(ctx, "GET", "/fapi/v1/exchangeInfo")
    rows = []
    for s in (payload or {}).get("symbols") or []:
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") not in ("USDT", "USDC"):
            continue
        rows.append({
            "venue": "binance",
            "symbol": s.get("symbol"),
            "base_asset": s.get("baseAsset"),
            "fund_interval_hours": 8,
            "source": "binance/exchangeInfo.assumed_8h",
        })
    return pd.DataFrame(rows)


async def _okx_cadence() -> pd.DataFrame:
    """Default 8h. Per-symbol 4h detection via /funding-rate is deferred."""
    async with CollectorContext(venue="okx", rest_base="https://www.okx.com",
                                 rate_limit_rps=10.0) as ctx:
        payload = await request_json(ctx, "GET", "/api/v5/public/instruments",
                                      params={"instType": "SWAP"})
    rows = []
    for inst in (payload or {}).get("data") or []:
        if inst.get("ctType") != "linear":
            continue
        rows.append({
            "venue": "okx",
            "symbol": inst.get("instId"),
            "base_asset": inst.get("ctValCcy") or inst.get("baseCcy"),
            "fund_interval_hours": 8,
            "source": "okx/public/instruments.assumed_8h",
        })
    return pd.DataFrame(rows)


async def _hyperliquid_cadence() -> pd.DataFrame:
    async with CollectorContext(venue="hyperliquid", rest_base="https://api.hyperliquid.xyz",
                                 rate_limit_rps=5.0) as ctx:
        payload = await request_json(ctx, "POST", "/info", json_body={"type": "meta"})
    rows = []
    for u in (payload or {}).get("universe") or []:
        rows.append({
            "venue": "hyperliquid",
            "symbol": u.get("name"),
            "base_asset": u.get("name"),
            "fund_interval_hours": 1,
            "source": "hyperliquid/meta.documented_1h",
        })
    return pd.DataFrame(rows)


async def build_cadence(out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or (Path(__file__).resolve().parents[2] / "data" / "static")
    out_dir.mkdir(parents=True, exist_ok=True)
    fns = {
        "bitget": _bitget_cadence(), "bybit": _bybit_cadence(),
        "binance": _binance_cadence(), "okx": _okx_cadence(),
        "hyperliquid": _hyperliquid_cadence(),
    }
    frames = []
    for venue, res in zip(fns.keys(), await asyncio.gather(*fns.values(), return_exceptions=True)):
        if isinstance(res, Exception):
            _LOG.error("cadence fetch %s failed: %s", venue, res)
            continue
        _LOG.info("cadence %s: %d rows | distribution=%s",
                  venue, len(res),
                  res["fund_interval_hours"].value_counts().to_dict() if not res.empty else {})
        frames.append(res)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df.sort_values(["venue", "symbol"]).reset_index(drop=True)
    df.to_csv(out_dir / "funding_cadence.csv", index=False)
    write_parquet(df, out_dir / "funding_cadence.parquet")
    return df


if __name__ == "__main__":
    asyncio.run(build_cadence())
