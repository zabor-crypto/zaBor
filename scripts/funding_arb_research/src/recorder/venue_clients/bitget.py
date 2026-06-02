"""Bitget USDT-M futures WS recorder.

Endpoint: ``wss://ws.bitget.com/v2/ws/public``
InstType: ``USDT-FUTURES`` (linear perp).
Channels subscribed (book channel configurable via ``depth_levels``):
  * ``books`` / ``books15`` / ``books5``  — full or top-N L2
  * ``trade``   — public trades
  * ``ticker``  — mark/index/funding/last (one stream gives everything)

``depth_levels`` defaults to ``"15"`` (Bitget v2 ``books15`` channel ≈
top-15 levels both sides) which cuts disk volume ~10× vs full ``books``
diff. Pass ``"full"`` to record the full L2 diff. ``"5"`` is an even
tighter alternative for storage-bound deployments.

Bitget v2 expects a literal text ``ping`` every ~25s and replies
``pong``. WS-protocol pings are not enough.

Symbol mapping: canonical ``BTC`` → Bitget instId ``BTCUSDT``.
Coin extraction strips the ``USDT`` quote suffix.

References:
  - https://www.bitget.com/api-doc/contract/websocket/public/Books-Channel
  - https://www.bitget.com/api-doc/contract/websocket/public/Tickers-Channel
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import VenueClient, VenueClientConfig

_INST_TYPE = "USDT-FUTURES"
_QUOTE = "USDT"


_DEPTH_TO_CHANNEL = {"full": "books", "15": "books15", "5": "books5", "1": "books1"}


class BitgetClient(VenueClient):

    def __init__(self, cfg, writer, books_channel: str = "books15") -> None:
        super().__init__(cfg, writer)
        self.books_channel = books_channel

    @classmethod
    def build(cls, coins: list[str], writer, depth_levels: str = "15") -> "BitgetClient":
        cfg = VenueClientConfig(
            venue="bitget",
            coins=[c.upper() for c in coins],
            ws_url="wss://ws.bitget.com/v2/ws/public",
            ping_interval_s=25.0,
        )
        channel = _DEPTH_TO_CHANNEL.get(depth_levels, "books15")
        return cls(cfg, writer, books_channel=channel)

    def venue_symbol(self, coin: str) -> str:
        return f"{coin.upper()}{_QUOTE}"

    def text_ping_payload(self):
        return "ping"

    def subscribe_payloads(self) -> list[dict[str, Any]]:
        args = []
        for coin in self.cfg.coins:
            sym = self.venue_symbol(coin)
            for channel in (self.books_channel, "trade", "ticker"):
                args.append({"instType": _INST_TYPE, "channel": channel, "instId": sym})
        # Bitget allows up to ~50 args per subscribe; chunk to be safe.
        chunks = [args[i:i + 30] for i in range(0, len(args), 30)]
        return [{"op": "subscribe", "args": chunk} for chunk in chunks]

    # ---- message dispatch ----

    def handle_message(self, msg: dict[str, Any], recv_ts_utc: float) -> None:
        if msg.get("event") in ("subscribe", "error", "login"):
            if msg.get("event") == "error":
                self.log.error("subscribe error: %s", msg)
            return
        arg = msg.get("arg") or {}
        channel = arg.get("channel")
        inst = arg.get("instId", "")
        coin = self._coin_from_inst(inst)
        if coin is None:
            return
        action = msg.get("action")  # "snapshot" | "update"
        data = msg.get("data") or []
        if channel and channel.startswith("books"):
            self._on_books(coin, inst, action, data, recv_ts_utc)
        elif channel == "trade":
            self._on_trade(coin, inst, data, recv_ts_utc)
        elif channel == "ticker":
            self._on_ticker(coin, inst, data, recv_ts_utc)

    @staticmethod
    def _coin_from_inst(inst: str) -> str | None:
        if not inst.endswith(_QUOTE):
            return None
        return inst[: -len(_QUOTE)]

    def _on_books(self, coin: str, sym: str, action: str | None, data: list[dict], recv: float) -> None:
        is_snap = (action == "snapshot")
        for entry in data:
            ts = self._ts(entry.get("ts"))
            seq = entry.get("seq")
            for side, levels in (("bid", entry.get("bids") or []), ("ask", entry.get("asks") or [])):
                for lvl in levels:
                    if len(lvl) < 2:
                        continue
                    self.writer.put("book", "bitget", coin, {
                        "timestamp_utc": ts,
                        "recv_ts_utc": pd.Timestamp(recv, unit="s", tz="UTC"),
                        "venue": "bitget",
                        "symbol": sym,
                        "side": side,
                        "price": float(lvl[0]),
                        "size": float(lvl[1]),
                        "is_snapshot": is_snap,
                        "seq": int(seq) if seq is not None else None,
                    })

    def _on_trade(self, coin: str, sym: str, data: list[dict], recv: float) -> None:
        for t in data:
            self.writer.put("trades", "bitget", coin, {
                "timestamp_utc": self._ts(t.get("ts")),
                "recv_ts_utc": pd.Timestamp(recv, unit="s", tz="UTC"),
                "venue": "bitget",
                "symbol": sym,
                "price": float(t.get("price")),
                "size": float(t.get("size")),
                "side": t.get("side"),
                "trade_id": str(t.get("tradeId")) if t.get("tradeId") is not None else None,
            })

    def _on_ticker(self, coin: str, sym: str, data: list[dict], recv: float) -> None:
        for t in data:
            ts = self._ts(t.get("ts"))
            row = {
                "timestamp_utc": ts,
                "recv_ts_utc": pd.Timestamp(recv, unit="s", tz="UTC"),
                "venue": "bitget",
                "symbol": sym,
                "mark_price": _f(t.get("markPrice")),
                "index_price": _f(t.get("indexPrice")),
                "last_price": _f(t.get("lastPr")),
                "funding_rate": _f(t.get("fundingRate")),
                "next_funding_time": self._ts(t.get("nextFundingTime")),
                "open_interest": _f(t.get("holdingAmount")),
            }
            self.writer.put("mark", "bitget", coin, row)

    @staticmethod
    def _ts(v: Any) -> pd.Timestamp | None:
        if v is None or v == "":
            return None
        try:
            return pd.Timestamp(int(v), unit="ms", tz="UTC")
        except (TypeError, ValueError):
            return None


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
