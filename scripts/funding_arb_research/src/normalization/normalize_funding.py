"""Per-venue normalizers for funding data into the canonical schema."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..utils.io import hash_payload
from ..utils.time import to_utc
from .schema import SchemaIssue, conform_funding, empty_funding_frame


# --------------------------------------------------------------------------- #
# Binance
# --------------------------------------------------------------------------- #

def normalize_binance_funding(
    payload: list[dict],
    *,
    issues: list[SchemaIssue] | None = None,
    funding_interval_hours: int = 8,
) -> pd.DataFrame:
    """Normalize Binance USD-M futures /fundingRate response.

    Schema (live API at time of authoring):
        symbol, fundingTime (ms), fundingRate, markPrice
    """
    if not payload:
        return empty_funding_frame()
    rows = []
    for raw in payload:
        try:
            symbol = raw["symbol"]
            ts = to_utc(int(raw["fundingTime"]))
            funding_rate = float(raw["fundingRate"])
            mark = float(raw.get("markPrice")) if raw.get("markPrice") not in (None, "") else float("nan")
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue(
                    venue="binance",
                    endpoint="fundingRate",
                    field=str(e),
                    expected="symbol/fundingTime/fundingRate present",
                    observed=str(raw)[:120],
                ))
            continue

        # Symbol parsing: BTCUSDT -> BTC / USDT (best-effort).
        base, quote = _split_binance_symbol(symbol)
        rows.append({
            "timestamp_utc": ts,
            "venue": "binance",
            "symbol": symbol,
            "base_asset": base,
            "quote_asset": quote,
            "instrument_type": "perp",
            "funding_rate": funding_rate,
            "funding_interval_hours": funding_interval_hours,
            "mark_price": mark,
            "source": "binance/fundingRate",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_funding(pd.DataFrame(rows))


def _split_binance_symbol(sym: str) -> tuple[str, str]:
    for q in ("USDT", "USDC", "BUSD", "USD"):
        if sym.endswith(q):
            return sym[: -len(q)], q
    return sym, ""


# --------------------------------------------------------------------------- #
# Bybit
# --------------------------------------------------------------------------- #

def normalize_bybit_funding(
    payload: dict,
    *,
    issues: list[SchemaIssue] | None = None,
    funding_interval_hours: int = 8,
) -> pd.DataFrame:
    """Normalize Bybit V5 funding/history.

    Returns under: result.list[{symbol, fundingRate, fundingRateTimestamp}]
    """
    if not payload or "result" not in payload:
        return empty_funding_frame()
    items = (payload.get("result") or {}).get("list") or []
    rows = []
    for raw in items:
        try:
            symbol = raw["symbol"]
            ts = to_utc(int(raw["fundingRateTimestamp"]))
            funding_rate = float(raw["fundingRate"])
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue(
                    venue="bybit", endpoint="funding/history",
                    field=str(e),
                    expected="symbol/fundingRate/fundingRateTimestamp",
                    observed=str(raw)[:120],
                ))
            continue
        base, quote = _split_binance_symbol(symbol)  # same convention
        rows.append({
            "timestamp_utc": ts,
            "venue": "bybit",
            "symbol": symbol,
            "base_asset": base,
            "quote_asset": quote,
            "instrument_type": "perp",
            "funding_rate": funding_rate,
            "funding_interval_hours": funding_interval_hours,
            "source": "bybit/funding/history",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_funding(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# OKX
# --------------------------------------------------------------------------- #

def normalize_okx_funding(
    payload: dict,
    *,
    issues: list[SchemaIssue] | None = None,
    funding_interval_hours: int = 8,
) -> pd.DataFrame:
    """Normalize OKX /api/v5/public/funding-rate-history.

    items: instId, realizedRate, fundingTime (ms str), fundingRate, ...
    """
    if not payload or "data" not in payload:
        return empty_funding_frame()
    rows = []
    for raw in payload["data"]:
        try:
            inst_id = raw["instId"]                   # e.g. BTC-USDT-SWAP
            ts = to_utc(int(raw["fundingTime"]))
            funding_rate = float(raw.get("realizedRate") or raw.get("fundingRate"))
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue(
                    venue="okx", endpoint="funding-rate-history",
                    field=str(e),
                    expected="instId/fundingTime/realizedRate or fundingRate",
                    observed=str(raw)[:120],
                ))
            continue
        parts = inst_id.split("-")
        base = parts[0] if len(parts) >= 1 else ""
        quote = parts[1] if len(parts) >= 2 else ""
        rows.append({
            "timestamp_utc": ts,
            "venue": "okx",
            "symbol": inst_id,
            "base_asset": base,
            "quote_asset": quote,
            "instrument_type": "perp",
            "funding_rate": funding_rate,
            "funding_interval_hours": funding_interval_hours,
            "source": "okx/funding-rate-history",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_funding(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# Hyperliquid
# --------------------------------------------------------------------------- #

def normalize_binance_premium_index(
    payload: list[dict] | dict,
    *,
    issues: list[SchemaIssue] | None = None,
    funding_interval_hours: int = 8,
) -> pd.DataFrame:
    """Normalize Binance ``/fapi/v1/premiumIndex`` rows.

    Binance does not expose a single "next-period predicted funding" field.
    The premiumIndex stream's ``lastFundingRate`` is the *currently
    accruing* funding rate (i.e. our best live estimate before settlement
    finalizes), which is exactly what we want as a predicted_funding_rate
    proxy at any point in time. We also capture markPrice / indexPrice.
    """
    items = payload if isinstance(payload, list) else [payload]
    rows = []
    for raw in items:
        try:
            symbol = raw["symbol"]
            ts = to_utc(int(raw["time"]))
            mark = float(raw.get("markPrice")) if raw.get("markPrice") not in (None, "") else float("nan")
            idx = float(raw.get("indexPrice")) if raw.get("indexPrice") not in (None, "") else float("nan")
            last_rate = raw.get("lastFundingRate")
            predicted = float(last_rate) if last_rate not in (None, "") else float("nan")
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue(
                    venue="binance", endpoint="premiumIndex",
                    field=str(e),
                    expected="symbol/time/lastFundingRate",
                    observed=str(raw)[:120],
                ))
            continue
        base, quote = _split_binance_symbol(symbol)
        rows.append({
            "timestamp_utc": ts,
            "venue": "binance",
            "symbol": symbol,
            "base_asset": base, "quote_asset": quote,
            "instrument_type": "perp",
            "funding_rate": float("nan"),
            "predicted_funding_rate": predicted,
            "funding_interval_hours": funding_interval_hours,
            "mark_price": mark, "index_price": idx,
            "source": "binance/premiumIndex",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_funding(pd.DataFrame(rows))


_HL_PREDICTED_VENUE_MAP: dict[str, tuple[str, str]] = {
    # raw key in HL payload -> (canonical_venue, quote_for_symbol)
    "hlperp": ("hyperliquid", ""),
    "binperp": ("binance", "USDT"),
    "bybitperp": ("bybit", "USDT"),
    "okxperp": ("okx", "USDT"),
}


def _hl_canonical_symbol(venue: str, coin: str, quote: str) -> str:
    if venue == "hyperliquid":
        return coin
    if venue == "okx":
        return f"{coin}-USDT-SWAP"
    return f"{coin}{quote}"


def normalize_hyperliquid_predicted(
    payload: list,
    *,
    issues: list[SchemaIssue] | None = None,
) -> pd.DataFrame:
    """Normalize Hyperliquid ``predictedFundings`` payload.

    The HL payload tags the source venues as ``hlperp`` / ``binperp`` /
    ``bybitperp`` / ``okxperp``. We *canonicalize* both venue and symbol
    here so downstream lookups by ``(venue='binance', symbol='BTCUSDT')``
    just work — surfacing a schema issue if a venue key is unfamiliar.

    Payload shape:
        [
          [coin, [[venue, {fundingRate, nextFundingTime, fundingIntervalHours}], ...]],
          ...
        ]
    """
    rows = []
    if not isinstance(payload, list):
        return empty_funding_frame()
    for entry in payload:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        coin = entry[0]
        venue_list = entry[1]
        if not isinstance(venue_list, list):
            continue
        for vrow in venue_list:
            if not isinstance(vrow, list) or len(vrow) < 2:
                continue
            raw_venue, info = vrow[0], vrow[1]
            if not isinstance(info, dict):
                continue
            mapped = _HL_PREDICTED_VENUE_MAP.get(str(raw_venue).lower())
            if mapped is None:
                if issues is not None:
                    issues.append(SchemaIssue(
                        venue="hyperliquid", endpoint="predictedFundings",
                        field="venue",
                        expected="hlperp|binperp|bybitperp|okxperp",
                        observed=str(raw_venue),
                    ))
                continue
            canon_venue, quote = mapped
            try:
                rate = float(info.get("fundingRate")) if info.get("fundingRate") not in (None, "") else float("nan")
                interval = int(info.get("fundingIntervalHours") or 8)
                nft = info.get("nextFundingTime")
                # Treat 0/missing/negative as "no future settlement scheduled"
                # and stamp the row at collection time instead. This avoids
                # 1970 epoch leakage into time windows.
                if nft in (None, "") or (isinstance(nft, (int, float)) and nft <= 0):
                    ts = to_utc(pd.Timestamp.utcnow())
                else:
                    ts = to_utc(int(nft))
            except (KeyError, ValueError, TypeError) as e:
                if issues is not None:
                    issues.append(SchemaIssue(
                        venue="hyperliquid", endpoint="predictedFundings",
                        field=str(e),
                        expected="fundingRate/nextFundingTime/fundingIntervalHours",
                        observed=str(info)[:120],
                    ))
                continue
            rows.append({
                "timestamp_utc": ts,
                "venue": canon_venue,
                "symbol": _hl_canonical_symbol(canon_venue, coin, quote),
                "base_asset": coin,
                "quote_asset": "USD" if canon_venue == "hyperliquid" else "USDT",
                "instrument_type": "perp",
                "funding_rate": float("nan"),
                "predicted_funding_rate": rate,
                "funding_interval_hours": interval,
                "source": "hyperliquid/predictedFundings",
                "raw_payload_hash": hash_payload(vrow),
            })
    return conform_funding(pd.DataFrame(rows))


def normalize_hyperliquid_funding(
    payload: list[dict],
    *,
    coin: str,
    issues: list[SchemaIssue] | None = None,
    funding_interval_hours: int = 1,
) -> pd.DataFrame:
    """Hyperliquid /info ``fundingHistory`` returns:
        [{coin, fundingRate, premium, time}]
    """
    if not payload:
        return empty_funding_frame()
    rows = []
    for raw in payload:
        try:
            ts = to_utc(int(raw["time"]))
            rate = float(raw["fundingRate"])
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue(
                    venue="hyperliquid", endpoint="fundingHistory",
                    field=str(e),
                    expected="coin/fundingRate/time",
                    observed=str(raw)[:120],
                ))
            continue
        rows.append({
            "timestamp_utc": ts,
            "venue": "hyperliquid",
            "symbol": coin,
            "base_asset": coin,
            "quote_asset": "USD",
            "instrument_type": "perp",
            "funding_rate": rate,
            "funding_interval_hours": funding_interval_hours,
            "source": "hyperliquid/fundingHistory",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_funding(pd.DataFrame(rows))


def stitch_funding(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate normalized frames; drop exact duplicates."""
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return empty_funding_frame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp_utc", "venue", "symbol"]).sort_values("timestamp_utc")
    return df.reset_index(drop=True)
