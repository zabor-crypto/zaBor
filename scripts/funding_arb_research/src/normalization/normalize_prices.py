"""Per-venue normalizers for price/orderbook snapshots."""
from __future__ import annotations

import pandas as pd

from ..utils.io import hash_payload
from ..utils.time import to_utc
from .schema import SchemaIssue, conform_price, empty_price_frame


def _mid(bid: float | None, ask: float | None) -> float:
    if bid is None or ask is None:
        return float("nan")
    try:
        return (float(bid) + float(ask)) / 2.0
    except (TypeError, ValueError):
        return float("nan")


def normalize_binance_book_ticker(
    payload: list[dict] | dict,
    *,
    issues: list[SchemaIssue] | None = None,
) -> pd.DataFrame:
    """Binance bookTicker payload (single dict or list)."""
    items = payload if isinstance(payload, list) else [payload]
    rows = []
    for raw in items:
        try:
            sym = raw["symbol"]
            ts = to_utc(int(raw["time"])) if "time" in raw else to_utc(pd.Timestamp.utcnow())
            bid = float(raw["bidPrice"])
            ask = float(raw["askPrice"])
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue("binance", "bookTicker", str(e),
                                          "symbol/bidPrice/askPrice", str(raw)[:120]))
            continue
        rows.append({
            "timestamp_utc": ts,
            "venue": "binance", "symbol": sym,
            "base_asset": sym[:-4] if sym.endswith("USDT") else sym,
            "quote_asset": "USDT" if sym.endswith("USDT") else "",
            "instrument_type": "perp",
            "mid_price": _mid(bid, ask), "bid": bid, "ask": ask,
            "source": "binance/bookTicker",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_price(pd.DataFrame(rows))


def normalize_bybit_tickers(payload: dict, *, issues: list[SchemaIssue] | None = None) -> pd.DataFrame:
    items = ((payload or {}).get("result") or {}).get("list") or []
    rows = []
    for raw in items:
        try:
            sym = raw["symbol"]
            bid = float(raw["bid1Price"]) if raw.get("bid1Price") not in (None, "") else float("nan")
            ask = float(raw["ask1Price"]) if raw.get("ask1Price") not in (None, "") else float("nan")
            mark = float(raw.get("markPrice")) if raw.get("markPrice") not in (None, "") else float("nan")
            idx = float(raw.get("indexPrice")) if raw.get("indexPrice") not in (None, "") else float("nan")
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue("bybit", "tickers", str(e),
                                          "symbol/bid1/ask1", str(raw)[:120]))
            continue
        rows.append({
            "timestamp_utc": to_utc(pd.Timestamp.utcnow()),
            "venue": "bybit", "symbol": sym,
            "base_asset": sym[:-4] if sym.endswith("USDT") else sym,
            "quote_asset": "USDT" if sym.endswith("USDT") else "",
            "instrument_type": "perp",
            "mid_price": _mid(bid, ask), "bid": bid, "ask": ask,
            "mark_price": mark, "index_price": idx,
            "source": "bybit/tickers",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_price(pd.DataFrame(rows))


def normalize_okx_tickers(payload: dict, *, issues: list[SchemaIssue] | None = None) -> pd.DataFrame:
    items = (payload or {}).get("data") or []
    rows = []
    for raw in items:
        try:
            sym = raw["instId"]
            bid = float(raw["bidPx"]) if raw.get("bidPx") not in (None, "") else float("nan")
            ask = float(raw["askPx"]) if raw.get("askPx") not in (None, "") else float("nan")
            ts = to_utc(int(raw["ts"])) if raw.get("ts") else to_utc(pd.Timestamp.utcnow())
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue("okx", "tickers", str(e),
                                          "instId/bidPx/askPx/ts", str(raw)[:120]))
            continue
        parts = sym.split("-")
        rows.append({
            "timestamp_utc": ts,
            "venue": "okx", "symbol": sym,
            "base_asset": parts[0] if parts else "",
            "quote_asset": parts[1] if len(parts) > 1 else "",
            "instrument_type": "perp",
            "mid_price": _mid(bid, ask), "bid": bid, "ask": ask,
            "source": "okx/tickers",
            "raw_payload_hash": hash_payload(raw),
        })
    return conform_price(pd.DataFrame(rows))


def normalize_hyperliquid_meta_and_ctxs(
    payload: list, *, issues: list[SchemaIssue] | None = None
) -> pd.DataFrame:
    """Hyperliquid ``metaAndAssetCtxs`` returns ``[meta, [ctx_per_coin]]``.

    ctx fields: markPx, midPx, oraclePx, openInterest, funding, premium, ...
    """
    rows = []
    if not isinstance(payload, list) or len(payload) < 2:
        return empty_price_frame()
    meta, ctxs = payload[0], payload[1]
    universe = meta.get("universe", []) if isinstance(meta, dict) else []
    for u, ctx in zip(universe, ctxs):
        try:
            coin = u["name"]
            mark = float(ctx.get("markPx")) if ctx.get("markPx") not in (None, "") else float("nan")
            mid = float(ctx.get("midPx")) if ctx.get("midPx") not in (None, "") else float("nan")
            oracle = float(ctx.get("oraclePx")) if ctx.get("oraclePx") not in (None, "") else float("nan")
        except (KeyError, ValueError, TypeError) as e:
            if issues is not None:
                issues.append(SchemaIssue("hyperliquid", "metaAndAssetCtxs",
                                          str(e), "name/markPx/midPx", str(ctx)[:120]))
            continue
        rows.append({
            "timestamp_utc": to_utc(pd.Timestamp.utcnow()),
            "venue": "hyperliquid", "symbol": coin,
            "base_asset": coin, "quote_asset": "USD",
            "instrument_type": "perp",
            "mid_price": mid, "bid": float("nan"), "ask": float("nan"),
            "mark_price": mark, "index_price": oracle,
            "source": "hyperliquid/metaAndAssetCtxs",
            "raw_payload_hash": hash_payload(ctx),
        })
    return conform_price(pd.DataFrame(rows))
