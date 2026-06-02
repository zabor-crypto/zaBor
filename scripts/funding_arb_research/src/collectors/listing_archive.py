"""Listing-archive: per-venue perp launch (and delisting) timestamps.

The new-listing trigger needs to know **when** each (venue, coin) perp
went live. All four major CEX venues expose this in their public
instrument-metadata endpoints — no HTML scraping required:

  * Bitget:  ``/api/v2/mix/market/contracts``  → ``launchTime``, ``offTime``,
                                                  ``deliveryTime``
  * Binance: ``/fapi/v1/exchangeInfo``           → ``onboardDate``,
                                                  ``deliveryDate`` (per
                                                  ``contracts`` entry)
  * Bybit:   ``/v5/market/instruments-info``    → ``launchTime``, ``deliveryTime``
                                                  (category=linear)
  * OKX:     ``/api/v5/public/instruments``     → ``listTime``, ``expTime``
                                                  (instType=SWAP)
  * HL:      ``/info {type:meta}``              → no launch ts; we record
                                                  the universe order with
                                                  ``listing_unknown=True``.

Output: a single CSV ``data/static/listing_archive.csv`` with columns

  ``venue, symbol, base_asset, quote_asset, instrument_type,
    launch_ts_utc, delist_ts_utc, listing_unknown, source``

so downstream code can ask ``is_new_listing(coin, t, max_age_days)`` by
joining on ``(venue, base_asset)`` and comparing ``t - launch_ts_utc``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io import write_parquet
from ..utils.logging import get_logger
from .base import CollectorContext, request_json
from .bitget_collector import BitgetCollector

_LOG = get_logger("funding_arb.collectors.listing_archive")


async def fetch_bitget_listings(infer_launch_pages: int = 0) -> pd.DataFrame:
    """Bitget perp listings.

    The ``/contracts`` endpoint returns ``launchTime: ""`` for all active
    contracts, so we infer launch dates by probing the funding-history
    endpoint and taking the earliest observed ``fundingTime`` per symbol.
    Set ``infer_launch_pages`` > 0 to enable inference; each page is 100
    rows ≈ 33d at 8h cadence or 16d at 4h cadence. ``infer_launch_pages=2``
    covers any contract launched in the last ~33 days, which is what the
    new-listing trigger actually cares about. Older contracts retain
    ``listing_unknown=True`` and the trigger correctly skips them.
    """
    async with CollectorContext(venue="bitget", rest_base="https://api.bitget.com",
                                 rate_limit_rps=10.0) as ctx:
        bc = BitgetCollector(ctx=ctx)
        contracts = await bc.fetch_contracts()
        inferred: dict[str, pd.Timestamp] = {}
        if infer_launch_pages > 0 and not contracts.empty:
            _LOG.info("inferring Bitget launch dates: pages=%d for %d symbols",
                      infer_launch_pages, len(contracts))
            for sym in contracts["symbol"]:
                inferred_ts = await _infer_bitget_launch(ctx, sym, infer_launch_pages)
                if inferred_ts is not None:
                    inferred[sym] = inferred_ts
            _LOG.info("inferred launch ts for %d / %d Bitget symbols",
                      len(inferred), len(contracts))

    if contracts.empty:
        return contracts
    out = contracts.rename(columns={"launch_time": "launch_ts_utc"}).copy()
    if inferred:
        out["launch_ts_utc"] = out.apply(
            lambda r: inferred.get(r["symbol"], r["launch_ts_utc"]), axis=1
        )
    out["delist_ts_utc"] = out.get("off_time")
    out["listing_unknown"] = out["launch_ts_utc"].isna()
    return out[["venue", "symbol", "base_asset", "quote_asset", "instrument_type",
                "launch_ts_utc", "delist_ts_utc", "listing_unknown"]].assign(
        source="bitget/contracts+history-fund-rate"
    )


async def _infer_bitget_launch(ctx: CollectorContext, symbol: str, max_pages: int) -> pd.Timestamp | None:
    """Page Bitget funding history backwards; return earliest observed ts.

    Returns None if pagination doesn't terminate within ``max_pages`` (the
    contract is older than the probed window — a non-result that's also
    correct: this symbol is not a "new listing").
    """
    page_no = 1
    earliest_ms: int | None = None
    while page_no <= max_pages:
        try:
            payload = await request_json(
                ctx, "GET", "/api/v2/mix/market/history-fund-rate",
                params={"symbol": symbol, "productType": "USDT-FUTURES",
                        "pageSize": 100, "pageNo": page_no},
            )
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict) or payload.get("code") not in ("00000", 0):
            return None
        data = payload.get("data") or []
        if not data:
            break
        try:
            page_min = min(int(r["fundingTime"]) for r in data if r.get("fundingTime"))
        except (KeyError, ValueError, TypeError):
            return None
        earliest_ms = page_min if earliest_ms is None else min(earliest_ms, page_min)
        if len(data) < 100:
            # End of history reached — earliest_ms is the actual launch.
            return pd.Timestamp(earliest_ms, unit="ms", tz="UTC")
        page_no += 1
    # Hit page cap → contract is older than the probed window. Keep unknown.
    return None


async def fetch_binance_listings() -> pd.DataFrame:
    async with CollectorContext(venue="binance", rest_base="https://fapi.binance.com",
                                 rate_limit_rps=10.0) as ctx:
        payload = await request_json(ctx, "GET", "/fapi/v1/exchangeInfo")
    rows = []
    for sym in (payload or {}).get("symbols") or []:
        if sym.get("contractType") != "PERPETUAL":
            continue
        if sym.get("quoteAsset") not in ("USDT", "USDC"):
            continue
        rows.append({
            "venue": "binance",
            "symbol": sym.get("symbol"),
            "base_asset": sym.get("baseAsset"),
            "quote_asset": sym.get("quoteAsset"),
            "instrument_type": "perp",
            "launch_ts_utc": _ms(sym.get("onboardDate")),
            "delist_ts_utc": _ms(sym.get("deliveryDate")) if sym.get("status") == "PENDING_TRADING" else None,
            "listing_unknown": sym.get("onboardDate") in (None, "", 0),
            "source": "binance/exchangeInfo",
        })
    return pd.DataFrame(rows)


async def fetch_bybit_listings() -> pd.DataFrame:
    async with CollectorContext(venue="bybit", rest_base="https://api.bybit.com",
                                 rate_limit_rps=10.0) as ctx:
        # Paginate just in case (>1000 perps eventually).
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
                if inst.get("quoteCoin") not in ("USDT", "USDC"):
                    continue
                rows.append({
                    "venue": "bybit",
                    "symbol": inst.get("symbol"),
                    "base_asset": inst.get("baseCoin"),
                    "quote_asset": inst.get("quoteCoin"),
                    "instrument_type": "perp",
                    "launch_ts_utc": _ms(inst.get("launchTime")),
                    "delist_ts_utc": _ms(inst.get("deliveryTime")) if inst.get("deliveryTime") not in (None, "", "0") else None,
                    "listing_unknown": inst.get("launchTime") in (None, "", "0"),
                    "source": "bybit/instruments-info",
                })
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
    return pd.DataFrame(rows)


async def fetch_okx_listings() -> pd.DataFrame:
    async with CollectorContext(venue="okx", rest_base="https://www.okx.com",
                                 rate_limit_rps=10.0) as ctx:
        payload = await request_json(ctx, "GET", "/api/v5/public/instruments",
                                      params={"instType": "SWAP"})
    rows = []
    for inst in (payload or {}).get("data") or []:
        # OKX SWAP includes both linear and inverse — keep linear only (settle in quote).
        if inst.get("ctType") != "linear":
            continue
        rows.append({
            "venue": "okx",
            "symbol": inst.get("instId"),
            "base_asset": inst.get("ctValCcy") or inst.get("baseCcy"),
            "quote_asset": inst.get("settleCcy"),
            "instrument_type": "perp",
            "launch_ts_utc": _ms(inst.get("listTime")),
            "delist_ts_utc": _ms(inst.get("expTime")) if inst.get("expTime") not in (None, "", "0") else None,
            "listing_unknown": inst.get("listTime") in (None, "", "0"),
            "source": "okx/public/instruments",
        })
    return pd.DataFrame(rows)


async def fetch_hyperliquid_listings() -> pd.DataFrame:
    """HL exposes no launch timestamp; we record the universe with listing_unknown=True."""
    async with CollectorContext(venue="hyperliquid", rest_base="https://api.hyperliquid.xyz",
                                 rate_limit_rps=5.0) as ctx:
        payload = await request_json(ctx, "POST", "/info", json_body={"type": "meta"})
    rows = []
    for u in (payload or {}).get("universe") or []:
        rows.append({
            "venue": "hyperliquid",
            "symbol": u.get("name"),
            "base_asset": u.get("name"),
            "quote_asset": "USD",
            "instrument_type": "perp",
            "launch_ts_utc": None,
            "delist_ts_utc": None,
            "listing_unknown": True,
            "source": "hyperliquid/meta",
        })
    return pd.DataFrame(rows)


async def build_archive(
    out_dir: Path | None = None,
    bitget_infer_launch_pages: int = 2,
) -> pd.DataFrame:
    """Run all fetchers concurrently and write the merged CSV + parquet."""
    out_dir = out_dir or (Path(__file__).resolve().parents[2] / "data" / "static")
    out_dir.mkdir(parents=True, exist_ok=True)
    fetchers = {
        "bitget": fetch_bitget_listings(infer_launch_pages=bitget_infer_launch_pages),
        "binance": fetch_binance_listings(),
        "bybit": fetch_bybit_listings(),
        "okx": fetch_okx_listings(),
        "hyperliquid": fetch_hyperliquid_listings(),
    }
    results = await asyncio.gather(*fetchers.values(), return_exceptions=True)
    frames = []
    for venue, res in zip(fetchers.keys(), results):
        if isinstance(res, Exception):
            _LOG.error("listing fetch failed for %s: %s", venue, res)
            continue
        _LOG.info("listing fetch %s: %d rows", venue, len(res))
        frames.append(res)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df.sort_values(["venue", "base_asset"]).reset_index(drop=True)
    df.to_csv(out_dir / "listing_archive.csv", index=False)
    write_parquet(df, out_dir / "listing_archive.parquet")
    return df


def _ms(v: Any) -> pd.Timestamp | None:
    if v is None or v == "" or v == "0" or v == 0:
        return None
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return None


# CLI
if __name__ == "__main__":
    asyncio.run(build_archive())
