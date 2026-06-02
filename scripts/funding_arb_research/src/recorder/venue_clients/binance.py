"""Binance USD-M futures WS recorder.

Combined stream endpoint:
  ``wss://fstream.binance.com/stream?streams=<a>/<b>/...``

Subscribed streams per symbol (all lowercased):
  * ``<sym>@depth20@100ms``   — top-20 bid/ask snapshots (configurable)
  * ``<sym>@aggTrade``         — aggregated public trades
  * ``<sym>@markPrice@1s``     — mark + index + funding rate + next funding time

``depth_levels`` defaults to ``"20"`` which uses ``<sym>@depth20@100ms``
(top-20 partial-depth stream — the venue pushes a *full* top-20
snapshot every 100ms, no diff seed needed). Pass ``"full"`` for
``<sym>@depth@100ms`` (raw L2 diff). Other valid levels: ``5``, ``10``.

Server protocol-level ping every 3 minutes; aiohttp ``autoping`` replies
automatically. No text-frame ping needed.

Symbol mapping: canonical ``BTC`` → Binance symbol ``BTCUSDT``.

References:
  - https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import VenueClient, VenueClientConfig

_QUOTE = "USDT"

# Binance Futures uses a "1000" prefix for high-supply tokens to keep
# tick sizes practical. Map canonical base → Binance Futures symbol.
_BASE_PREFIX_OVERRIDES = {
    "PEPE": "1000PEPE", "BONK": "1000BONK", "FLOKI": "1000FLOKI",
    "LUNC": "1000LUNC", "SATS": "1000SATS", "SHIB": "1000SHIB",
    "RATS": "1000RATS", "XEC":  "1000XEC",  "CAT":  "1000CAT",
    "CHEEMS": "1000CHEEMS",
}


def _binance_base(coin: str) -> str:
    return _BASE_PREFIX_OVERRIDES.get(coin.upper(), coin.upper())


_DEPTH_TO_STREAM = {
    "full": "depth@100ms",
    # Default cadence is 500ms — partial-book is a fixed-cadence snapshot
    # stream (every 100/250/500ms regardless of change). 500ms gives 2 Hz
    # which is plenty for slippage modeling at $100K-$1M notional on liquid
    # pairs and cuts disk volume 5× vs 100ms.
    "20": "depth20@500ms", "10": "depth10@500ms", "5": "depth5@500ms",
    "20@100ms": "depth20@100ms", "10@100ms": "depth10@100ms", "5@100ms": "depth5@100ms",
    "20@250ms": "depth20@250ms",
}


class BinanceClient(VenueClient):

    def __init__(self, cfg, writer, is_partial_book: bool = True) -> None:
        super().__init__(cfg, writer)
        self.is_partial_book = is_partial_book

    @classmethod
    def build(cls, coins: list[str], writer, depth_levels: str = "20") -> "BinanceClient":
        coins_u = [c.upper() for c in coins]
        depth_stream = _DEPTH_TO_STREAM.get(depth_levels, "depth20@100ms")
        is_partial = depth_levels != "full"
        streams = []
        for c in coins_u:
            base = _binance_base(c)
            sym_l = f"{base.lower()}{_QUOTE.lower()}"
            # NOTE: @aggTrade and @markPrice silently return 0 messages on the
            # Binance Futures WS gateway from AWS Tokyo region (verified
            # 2026-04-27 and re-confirmed 2026-05-03 after the REST ban). Use
            # @trade; collect mark+funding via REST /fapi/v1/premiumIndex when
            # the IP ban lifts — the binance_mark_rest poller retries automatically.
            streams.extend([f"{sym_l}@{depth_stream}", f"{sym_l}@trade"])
        url = "wss://fstream.binance.com/stream?streams=" + "/".join(streams)
        cfg = VenueClientConfig(
            venue="binance",
            coins=coins_u,
            ws_url=url,
            ping_interval_s=180.0,
        )
        return cls(cfg, writer, is_partial_book=is_partial)

    def venue_symbol(self, coin: str) -> str:
        return f"{_binance_base(coin)}{_QUOTE}"

    def subscribe_payloads(self) -> list[dict[str, Any]]:
        # Streams are encoded in the connect URL; no explicit subscribe needed.
        return []

    def handle_message(self, msg: dict[str, Any], recv_ts_utc: float) -> None:
        stream = msg.get("stream")
        data = msg.get("data") or {}
        if not stream or not isinstance(data, dict):
            return
        # stream looks like "btcusdt@depth@100ms" / "btcusdt@aggTrade" / "btcusdt@markPrice@1s"
        parts = stream.split("@")
        sym_l = parts[0]
        kind = parts[1] if len(parts) > 1 else ""
        sym = sym_l.upper()
        coin = self._coin_from_sym(sym)
        if coin is None:
            return
        recv_ts = pd.Timestamp(recv_ts_utc, unit="s", tz="UTC")
        if kind.startswith("depth"):
            self._on_depth(coin, sym, data, recv_ts)
        elif kind in ("trade", "aggTrade"):
            self._on_trade(coin, sym, data, recv_ts)
        elif kind == "markPrice":
            self._on_mark(coin, sym, data, recv_ts)

    @staticmethod
    def _coin_from_sym(sym: str) -> str | None:
        if not sym.endswith(_QUOTE):
            return None
        base = sym[: -len(_QUOTE)]
        # Reverse the 1000-prefix mapping so files land under the canonical coin.
        if base.startswith("1000") and base[4:] in _BASE_PREFIX_OVERRIDES:
            return base[4:]
        return base

    def _on_depth(self, coin: str, sym: str, d: dict, recv: pd.Timestamp) -> None:
        """Handle both ``@depth@100ms`` (diff, has E/T/U/u/b/a) and
        ``@depth<N>@100ms`` (partial-book snapshot, has lastUpdateId/bids/asks).
        """
        if d.get("e") == "depthUpdate" or "U" in d:
            # depthUpdate envelope. On futures, partial-book streams (depth20@100ms,
            # depth10/5) reuse this envelope but each message is a full top-N
            # snapshot, not a diff. We tag is_snapshot from the stream type.
            ts = _ms(d.get("T")) or _ms(d.get("E"))
            seq_first, seq_last = d.get("U"), d.get("u")
            bids, asks = d.get("b") or [], d.get("a") or []
            is_snap = self.is_partial_book
        else:
            # Bare partial-book payload (no depthUpdate envelope) — spot style
            ts = recv
            seq_first = seq_last = d.get("lastUpdateId")
            bids, asks = d.get("bids") or [], d.get("asks") or []
            is_snap = True
        for side, levels in (("bid", bids), ("ask", asks)):
            for lvl in levels:
                if len(lvl) < 2:
                    continue
                self.writer.put("book", "binance", coin, {
                    "timestamp_utc": ts,
                    "recv_ts_utc": recv,
                    "venue": "binance",
                    "symbol": sym,
                    "side": side,
                    "price": float(lvl[0]),
                    "size": float(lvl[1]),
                    "is_snapshot": is_snap,
                    "seq_first": seq_first,
                    "seq_last": seq_last,
                })

    def _on_trade(self, coin: str, sym: str, d: dict, recv: pd.Timestamp) -> None:
        # @trade trades carry id under "t"; @aggTrade carries it under "a".
        side = "sell" if d.get("m") else "buy"  # m = buyer is maker → taker is seller
        trade_id = d.get("t") if d.get("t") is not None else d.get("a")
        self.writer.put("trades", "binance", coin, {
            "timestamp_utc": _ms(d.get("T")),
            "recv_ts_utc": recv,
            "venue": "binance",
            "symbol": sym,
            "price": float(d.get("p")),
            "size": float(d.get("q")),
            "side": side,
            "trade_id": str(trade_id) if trade_id is not None else None,
        })

    def _on_mark(self, coin: str, sym: str, d: dict, recv: pd.Timestamp) -> None:
        self.writer.put("mark", "binance", coin, {
            "timestamp_utc": _ms(d.get("E")),
            "recv_ts_utc": recv,
            "venue": "binance",
            "symbol": sym,
            "mark_price": _f(d.get("p")),
            "index_price": _f(d.get("i")),
            "est_settle_price": _f(d.get("P")),
            "funding_rate": _f(d.get("r")),
            "next_funding_time": _ms(d.get("T")),
        })


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ms(v):
    if v is None or v == "":
        return None
    try:
        return pd.Timestamp(int(v), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return None
