"""Canonical event contracts for Binance continuation v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union


SCHEMA_VERSION_V1 = "v1"

VENUE_BINANCE_UM = "binance_um_futures"

STREAM_FORCE_ORDER = "forceOrder"
STREAM_AGG_TRADE = "aggTrade"
STREAM_DEPTH = "depth"
STREAM_MARK_PRICE = "markPrice"

MANDATORY_STREAMS = (
    STREAM_FORCE_ORDER,
    STREAM_AGG_TRADE,
    STREAM_DEPTH,
    # markPrice excluded: Binance selectively drops this stream from certain ASNs.
    # last_mark_price defaults to 0.0 and is logged-only; not used for entry gating.
)

MVP_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT")


@dataclass(frozen=True)
class RawMarketEvent:
    """Immutable pre-normalization market event captured from exchange streams."""

    event_id: str
    schema_version: str
    venue: str
    stream: str
    symbol: str
    conn_id: str
    exchange_ts_ms: int
    recv_wall_ts_ns: int
    recv_mono_ts_ns: int
    ingest_seq: int
    stream_local_seq: int
    payload_json: str
    payload_hash: str
    parse_status: Literal["ok", "parse_error"]
    gap_flag: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForceOrderEvent:
    event_id: str
    schema_version: str
    symbol: str
    exchange_ts_ms: int
    recv_mono_ts_ns: int
    side: Literal["BUY", "SELL"]
    qty: Decimal
    price: Decimal
    avg_price: Decimal
    status: str
    trade_ts_ms: int
    derived_notional: Decimal
    lossy_snapshot_flag: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("qty", "price", "avg_price", "derived_notional"):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class AggTradeEvent:
    event_id: str
    schema_version: str
    symbol: str
    exchange_ts_ms: int
    recv_mono_ts_ns: int
    trade_id: int
    price: Decimal
    qty: Decimal
    maker_side: Literal["BUY", "SELL"]
    taker_side: Literal["BUY", "SELL"]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("price", "qty"):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class DepthDeltaEvent:
    event_id: str
    schema_version: str
    symbol: str
    exchange_ts_ms: int
    recv_mono_ts_ns: int
    first_update_id: int
    final_update_id: int
    prev_final_update_id: Optional[int]
    bid_deltas: List[List[Decimal]]
    ask_deltas: List[List[Decimal]]
    checksum: Optional[int]
    snapshot_ref: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bid_deltas"] = [[str(px), str(qty)] for px, qty in self.bid_deltas]
        data["ask_deltas"] = [[str(px), str(qty)] for px, qty in self.ask_deltas]
        return data


@dataclass(frozen=True)
class MarkPriceEvent:
    event_id: str
    schema_version: str
    symbol: str
    exchange_ts_ms: int
    recv_mono_ts_ns: int
    mark_price: Decimal
    index_price: Optional[Decimal]
    funding_rate: Optional[Decimal]
    next_funding_ts_ms: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mark_price"] = str(self.mark_price)
        if self.index_price is not None:
            data["index_price"] = str(self.index_price)
        if self.funding_rate is not None:
            data["funding_rate"] = str(self.funding_rate)
        return data


@dataclass(frozen=True)
class FeatureSnapshotEvent:
    event_id: str
    schema_version: str
    episode_id: str
    decision_ts_ms: int
    symbol: str
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeStateEvent:
    event_id: str
    schema_version: str
    decision_id: str
    symbol: str
    decision_ts_ms: int
    regime: str
    tradable: bool
    no_trade_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionIntentEvent:
    event_id: str
    schema_version: str
    episode_id: str
    decision_id: str
    symbol: str
    decision_ts_ms: int
    action: Literal["NO_TRADE", "ENTER_LONG", "ENTER_SHORT"]
    score: float
    score_components: Dict[str, float]
    expected_edge_bps: float
    slippage_budget_bps: float
    no_trade_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskGateEvent:
    event_id: str
    schema_version: str
    decision_id: str
    symbol: str
    decision_ts_ms: int
    approved: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderIntentEvent:
    event_id: str
    schema_version: str
    decision_id: str
    order_intent_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: Decimal
    limit_price: Decimal
    ttl_ms: int
    max_slippage_bps: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["qty"] = str(self.qty)
        data["limit_price"] = str(self.limit_price)
        return data


@dataclass(frozen=True)
class OrderUpdateEvent:
    event_id: str
    schema_version: str
    order_intent_id: str
    order_id: Optional[str]
    symbol: str
    status: str
    exchange_ts_ms: int
    recv_mono_ts_ns: int
    detail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FillEvent:
    event_id: str
    schema_version: str
    order_intent_id: str
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    fill_price: Decimal
    fill_qty: Decimal
    exchange_ts_ms: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fill_price"] = str(self.fill_price)
        data["fill_qty"] = str(self.fill_qty)
        return data


@dataclass(frozen=True)
class PositionStateEvent:
    event_id: str
    schema_version: str
    position_id: str
    symbol: str
    side: Literal["LONG", "SHORT", "FLAT"]
    qty: Decimal
    avg_entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    exchange_ts_ms: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("qty", "avg_entry_price", "mark_price", "unrealized_pnl"):
            data[key] = str(data[key])
        return data


NormalizedMarketEvent = Union[
    ForceOrderEvent,
    AggTradeEvent,
    DepthDeltaEvent,
    MarkPriceEvent,
]
