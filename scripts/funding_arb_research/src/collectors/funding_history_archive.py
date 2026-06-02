"""18-month multi-venue funding-history archive for Step 4 backtests.

Pipeline:
  1. Pull Bitget USDT-FUTURES tickers, sort by ``quoteVolume``, take top-N.
  2. For each base asset in that top-N, paginate funding-rate history on
     every venue that lists it (Bitget + Binance + Bybit + OKX + HL),
     spanning ``--days`` lookback.
  3. Persist per-(venue, symbol) parquet under
     ``data/normalized/funding_history/<venue>/<symbol>.parquet``.

Output schema (canonical, minimal):
  ``timestamp_utc, venue, symbol, base_asset, funding_rate, source``

Deliberately *minimal*: this is the input to the event-engine backtest.
Mark/index/OI come from later runs against the live ticker WS.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import write_parquet, project_root
from ..utils.logging import get_logger
from .base import CollectorContext, request_json
from .bitget_collector import BitgetCollector

_LOG = get_logger("funding_arb.collectors.funding_history_archive")
_OUT_BASE = project_root() / "data" / "normalized" / "funding_history"


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #

async def top_n_bitget_by_volume(n: int = 30) -> pd.DataFrame:
    """Pull ALL Bitget USDT-FUTURES tickers via the plural endpoint, sorted by quoteVolume."""
    async with CollectorContext(venue="bitget", rest_base="https://api.bitget.com",
                                 rate_limit_rps=10.0) as ctx:
        payload = await request_json(
            ctx, "GET", "/api/v2/mix/market/tickers",
            params={"productType": "USDT-FUTURES"},
        )
    if not isinstance(payload, dict) or payload.get("code") not in ("00000", 0):
        return pd.DataFrame()
    rows = []
    for t in payload.get("data") or []:
        try:
            rows.append({
                "symbol": t.get("symbol"),
                "quote_volume_24h": float(t.get("quoteVolume", 0) or 0),
                "last_price": float(t.get("lastPr", 0) or 0),
                "funding_rate": float(t.get("fundingRate", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["quote_volume_24h"]).sort_values("quote_volume_24h", ascending=False)
    # Filter out non-vanilla suffixes (e.g. tokenized stocks may use suffixes); keep XXXUSDT.
    df = df[df["symbol"].str.endswith("USDT")]
    return df.head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-venue funding-history fetchers
# --------------------------------------------------------------------------- #

async def _bitget_funding(ctx: CollectorContext, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    page_no = 1
    while page_no <= 500:  # 500*100 = 50k rows ≈ 16 years at 8h cadence; safety cap
        payload = await request_json(
            ctx, "GET", "/api/v2/mix/market/history-fund-rate",
            params={"symbol": symbol, "productType": "USDT-FUTURES",
                    "pageSize": 100, "pageNo": page_no},
        )
        if not isinstance(payload, dict) or payload.get("code") not in ("00000", 0):
            break
        data = payload.get("data") or []
        if not data:
            break
        rows.extend(data)
        oldest_ms = min(int(r["fundingTime"]) for r in data if r.get("fundingTime"))
        if oldest_ms <= start_ms or len(data) < 100:
            break
        page_no += 1
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    out_rows = []
    for r in rows:
        try:
            t = int(r["fundingTime"])
            if t < start_ms or t > end_ms:
                continue
            out_rows.append({
                "timestamp_utc": pd.Timestamp(t, unit="ms", tz="UTC"),
                "venue": "bitget",
                "symbol": symbol,
                "base_asset": base,
                "funding_rate": float(r["fundingRate"]),
                "source": "bitget/history-fund-rate",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(out_rows).drop_duplicates(subset=["timestamp_utc", "symbol"])


async def _binance_funding(ctx: CollectorContext, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        payload = await request_json(
            ctx, "GET", "/fapi/v1/fundingRate",
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        last_ts = int(payload[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(payload) < 1000:
            break
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    out = []
    for r in rows:
        try:
            t = int(r["fundingTime"])
            if t < start_ms or t > end_ms:
                continue
            out.append({
                "timestamp_utc": pd.Timestamp(t, unit="ms", tz="UTC"),
                "venue": "binance", "symbol": symbol, "base_asset": base,
                "funding_rate": float(r["fundingRate"]),
                "source": "binance/fundingRate",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(out).drop_duplicates(subset=["timestamp_utc", "symbol"])


async def _bybit_funding(ctx: CollectorContext, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor_end = end_ms
    while True:
        payload = await request_json(
            ctx, "GET", "/v5/market/funding/history",
            params={"category": "linear", "symbol": symbol,
                    "startTime": start_ms, "endTime": cursor_end, "limit": 200},
        )
        items = ((payload or {}).get("result") or {}).get("list") or []
        if not items:
            break
        rows.extend(items)
        oldest = min(int(r["fundingRateTimestamp"]) for r in items)
        if oldest <= start_ms or len(items) < 200:
            break
        cursor_end = oldest - 1
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    out = []
    for r in rows:
        try:
            t = int(r["fundingRateTimestamp"])
            if t < start_ms or t > end_ms:
                continue
            out.append({
                "timestamp_utc": pd.Timestamp(t, unit="ms", tz="UTC"),
                "venue": "bybit", "symbol": symbol, "base_asset": base,
                "funding_rate": float(r["fundingRate"]),
                "source": "bybit/funding/history",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(out).drop_duplicates(subset=["timestamp_utc", "symbol"])


async def _okx_funding(ctx: CollectorContext, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX semantics: ``after`` = return rows older than this ts. Page from now backwards."""
    rows: list[dict] = []
    cursor_after: Optional[int] = None
    while True:
        params: dict = {"instId": symbol, "limit": 100}
        if cursor_after is not None:
            params["after"] = cursor_after
        payload = await request_json(
            ctx, "GET", "/api/v5/public/funding-rate-history", params=params,
        )
        items = (payload or {}).get("data") or []
        if not items:
            break
        rows.extend(items)
        oldest = min(int(r["fundingTime"]) for r in items)
        if oldest <= start_ms or len(items) < 100:
            break
        cursor_after = oldest - 1
    base = symbol.split("-")[0]
    out = []
    for r in rows:
        try:
            t = int(r["fundingTime"])
            if t < start_ms or t > end_ms:
                continue
            out.append({
                "timestamp_utc": pd.Timestamp(t, unit="ms", tz="UTC"),
                "venue": "okx", "symbol": symbol, "base_asset": base,
                "funding_rate": float(r["fundingRate"]),
                "source": "okx/funding-rate-history",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(out).drop_duplicates(subset=["timestamp_utc", "symbol"])


async def _hl_funding(ctx: CollectorContext, coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """HL `fundingHistory`: returns rows with {coin, fundingRate, premium, time}."""
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        payload = await request_json(
            ctx, "POST", "/info",
            json_body={"type": "fundingHistory", "coin": coin,
                       "startTime": cursor, "endTime": end_ms},
        )
        items = payload if isinstance(payload, list) else []
        if not items:
            break
        rows.extend(items)
        last_ts = int(items[-1]["time"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(items) < 500:
            break
    out = []
    for r in rows:
        try:
            t = int(r["time"])
            if t < start_ms or t > end_ms:
                continue
            out.append({
                "timestamp_utc": pd.Timestamp(t, unit="ms", tz="UTC"),
                "venue": "hyperliquid", "symbol": coin, "base_asset": coin,
                "funding_rate": float(r["fundingRate"]),
                "source": "hyperliquid/fundingHistory",
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(out).drop_duplicates(subset=["timestamp_utc", "symbol"])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

VENUE_REST = {
    "bitget": ("https://api.bitget.com", _bitget_funding, lambda c: f"{c}USDT"),
    "binance": ("https://fapi.binance.com", _binance_funding, lambda c: f"{c}USDT"),
    "bybit": ("https://api.bybit.com", _bybit_funding, lambda c: f"{c}USDT"),
    "okx": ("https://www.okx.com", _okx_funding, lambda c: f"{c}-USDT-SWAP"),
    "hyperliquid": ("https://api.hyperliquid.xyz", _hl_funding, lambda c: c),
}


async def archive_one_venue(venue: str, coins: list[str], start_ms: int, end_ms: int) -> dict[str, pd.DataFrame]:
    base_url, fetcher, sym_fn = VENUE_REST[venue]
    out: dict[str, pd.DataFrame] = {}
    async with CollectorContext(venue=venue, rest_base=base_url, rate_limit_rps=10.0) as ctx:
        for coin in coins:
            sym = sym_fn(coin)
            try:
                df = await fetcher(ctx, sym, start_ms, end_ms)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("%s %s funding fetch failed: %s", venue, sym, e)
                continue
            if df.empty:
                _LOG.info("%s %s: empty (likely not listed)", venue, sym)
                continue
            out[sym] = df.sort_values("timestamp_utc").reset_index(drop=True)
            _LOG.info("%s %s: %d rows %s..%s", venue, sym, len(df),
                      df["timestamp_utc"].min(), df["timestamp_utc"].max())
    return out


async def build(top_n: int = 30, days: int = 540, out_base: Path | None = None) -> dict:
    """Build the funding-history archive for top-N Bitget universe over `days` lookback."""
    out_base = out_base or _OUT_BASE
    out_base.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=days)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    _LOG.info("=== universe selection: top %d Bitget by 24h volume ===", top_n)
    universe = await top_n_bitget_by_volume(top_n)
    if universe.empty:
        _LOG.error("empty Bitget universe — abort"); return {}
    coins = [s.removesuffix("USDT") for s in universe["symbol"].tolist()]
    _LOG.info("coins: %s", coins)
    universe.to_csv(out_base / "_universe.csv", index=False)

    summary = {}
    for venue in VENUE_REST:
        _LOG.info("=== %s funding history (last %dd, %d coins) ===", venue, days, len(coins))
        per_sym = await archive_one_venue(venue, coins, start_ms, end_ms)
        venue_dir = out_base / venue
        venue_dir.mkdir(parents=True, exist_ok=True)
        for sym, df in per_sym.items():
            write_parquet(df, venue_dir / f"{sym}.parquet")
        summary[venue] = {"symbols": len(per_sym),
                          "rows": int(sum(len(d) for d in per_sym.values()))}

    _LOG.info("archive complete: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--days", type=int, default=540)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(build(top_n=args.top_n, days=args.days))
