"""Raw -> normalized stream event transformer."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Optional

from v1.contracts.event_types import (
    AggTradeEvent,
    DepthDeltaEvent,
    ForceOrderEvent,
    MarkPriceEvent,
    NormalizedMarketEvent,
    RawMarketEvent,
    SCHEMA_VERSION_V1,
    STREAM_AGG_TRADE,
    STREAM_DEPTH,
    STREAM_FORCE_ORDER,
    STREAM_MARK_PRICE,
)
from v1.normalize.validators import validate_raw_event


class EventNormalizer:
    def normalize(self, raw_event: RawMarketEvent) -> Optional[NormalizedMarketEvent]:
        validation = validate_raw_event(raw_event)
        if not validation.valid:
            return None

        payload = json.loads(raw_event.payload_json)
        data = payload.get("data", payload)

        if raw_event.stream == STREAM_FORCE_ORDER:
            return self._normalize_force_order(raw_event, data)
        if raw_event.stream == STREAM_AGG_TRADE:
            return self._normalize_agg_trade(raw_event, data)
        if raw_event.stream == STREAM_DEPTH:
            return self._normalize_depth(raw_event, data)
        if raw_event.stream == STREAM_MARK_PRICE:
            return self._normalize_mark_price(raw_event, data)
        return None

    @staticmethod
    def _normalize_force_order(raw: RawMarketEvent, data: dict) -> ForceOrderEvent:
        order = data["o"]
        qty = Decimal(str(order.get("q", "0")))
        price = Decimal(str(order.get("p", "0")))
        avg_price = Decimal(str(order.get("ap", order.get("p", "0"))))
        return ForceOrderEvent(
            event_id=raw.event_id,
            schema_version=SCHEMA_VERSION_V1,
            symbol=raw.symbol,
            exchange_ts_ms=raw.exchange_ts_ms,
            recv_mono_ts_ns=raw.recv_mono_ts_ns,
            side=str(order.get("S", "SELL")).upper(),
            qty=qty,
            price=price,
            avg_price=avg_price,
            status=str(order.get("X", "")),
            trade_ts_ms=int(order.get("T", raw.exchange_ts_ms)),
            derived_notional=qty * price,
            lossy_snapshot_flag=True,
        )

    @staticmethod
    def _normalize_agg_trade(raw: RawMarketEvent, data: dict) -> AggTradeEvent:
        maker_is_buyer = bool(data.get("m", False))
        maker_side = "BUY" if maker_is_buyer else "SELL"
        taker_side = "SELL" if maker_is_buyer else "BUY"
        return AggTradeEvent(
            event_id=raw.event_id,
            schema_version=SCHEMA_VERSION_V1,
            symbol=raw.symbol,
            exchange_ts_ms=raw.exchange_ts_ms,
            recv_mono_ts_ns=raw.recv_mono_ts_ns,
            trade_id=int(data.get("a", data.get("t", 0))),
            price=Decimal(str(data.get("p", "0"))),
            qty=Decimal(str(data.get("q", "0"))),
            maker_side=maker_side,
            taker_side=taker_side,
        )

    @staticmethod
    def _normalize_depth(raw: RawMarketEvent, data: dict) -> DepthDeltaEvent:
        bid_deltas = [
            [Decimal(str(level[0])), Decimal(str(level[1]))]
            for level in data.get("b", [])
            if len(level) >= 2
        ]
        ask_deltas = [
            [Decimal(str(level[0])), Decimal(str(level[1]))]
            for level in data.get("a", [])
            if len(level) >= 2
        ]
        checksum = data.get("cs")
        return DepthDeltaEvent(
            event_id=raw.event_id,
            schema_version=SCHEMA_VERSION_V1,
            symbol=raw.symbol,
            exchange_ts_ms=raw.exchange_ts_ms,
            recv_mono_ts_ns=raw.recv_mono_ts_ns,
            first_update_id=int(data.get("U", 0)),
            final_update_id=int(data.get("u", 0)),
            prev_final_update_id=int(data["pu"]) if "pu" in data else None,
            bid_deltas=bid_deltas,
            ask_deltas=ask_deltas,
            checksum=int(checksum) if checksum is not None else None,
            snapshot_ref=None,
        )

    @staticmethod
    def _normalize_mark_price(raw: RawMarketEvent, data: dict) -> MarkPriceEvent:
        return MarkPriceEvent(
            event_id=raw.event_id,
            schema_version=SCHEMA_VERSION_V1,
            symbol=raw.symbol,
            exchange_ts_ms=raw.exchange_ts_ms,
            recv_mono_ts_ns=raw.recv_mono_ts_ns,
            mark_price=Decimal(str(data.get("p", "0"))),
            index_price=Decimal(str(data.get("i"))) if data.get("i") is not None else None,
            funding_rate=Decimal(str(data.get("r"))) if data.get("r") is not None else None,
            next_funding_ts_ms=int(data.get("T")) if data.get("T") is not None else None,
        )
