import json

from v1.contracts.event_types import (
    RawMarketEvent,
    STREAM_AGG_TRADE,
    STREAM_DEPTH,
    STREAM_FORCE_ORDER,
    STREAM_MARK_PRICE,
)
from v1.ingest.binance_ws_client import BinanceWsClient
from v1.normalize.normalizer import EventNormalizer


def _raw(stream: str, payload: dict) -> RawMarketEvent:
    return RawMarketEvent(
        event_id="e",
        schema_version="v1",
        venue="binance_um_futures",
        stream=stream,
        symbol="BTCUSDT",
        conn_id="c",
        exchange_ts_ms=1,
        recv_wall_ts_ns=2,
        recv_mono_ts_ns=3,
        ingest_seq=4,
        stream_local_seq=5,
        payload_json=json.dumps({"stream": "btcusdt@x", "data": payload}),
        payload_hash="h",
        parse_status="ok",
        gap_flag=False,
    )


def test_normalize_force_order() -> None:
    payload = {"o": {"S": "SELL", "q": "2", "p": "100", "ap": "99", "X": "FILLED", "T": 7}}
    event = EventNormalizer().normalize(_raw(STREAM_FORCE_ORDER, payload))
    assert event is not None
    assert event.side == "SELL"
    assert str(event.derived_notional) == "200"


def test_normalize_agg_trade() -> None:
    payload = {"t": 10, "p": "101", "q": "0.5", "m": False, "T": 8}
    event = EventNormalizer().normalize(_raw(STREAM_AGG_TRADE, payload))
    assert event is not None
    assert event.taker_side == "BUY"


def test_normalize_depth() -> None:
    payload = {"U": 1, "u": 2, "b": [["100", "1"]], "a": [["101", "2"]]}
    event = EventNormalizer().normalize(_raw(STREAM_DEPTH, payload))
    assert event is not None
    assert event.first_update_id == 1
    assert event.final_update_id == 2


def test_normalize_mark_price() -> None:
    payload = {"p": "100", "E": 11, "r": "0.0001", "T": 12}
    event = EventNormalizer().normalize(_raw(STREAM_MARK_PRICE, payload))
    assert event is not None
    assert str(event.mark_price) == "100"


def test_ws_client_url_uses_global_force_order_stream() -> None:
    client = BinanceWsClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        streams=["forceOrder", "aggTrade", "depth"],
        on_raw_event=lambda e: None,
    )
    url = client._combined_stream_url()
    # Market-wide stream appears exactly once
    assert url.count("!forceOrder@arr") == 1
    # Per-symbol forceOrder subscriptions must NOT be present
    assert "btcusdt@forceOrder" not in url
    assert "ethusdt@forceOrder" not in url
    # Per-symbol trade/depth still present for both symbols
    assert "btcusdt@trade" in url
    assert "ethusdt@trade" in url


def test_ws_client_extract_symbol_from_force_order_arr() -> None:
    data = {"e": "forceOrder", "E": 123, "o": {"s": "SOLUSDT", "S": "SELL", "q": "10", "p": "50"}}
    symbol = BinanceWsClient._extract_symbol(data, "!forceOrder@arr")
    assert symbol == "SOLUSDT"


def test_ws_client_normalize_stream_name_force_order_arr() -> None:
    assert BinanceWsClient._normalize_stream_name("!forceOrder@arr") == STREAM_FORCE_ORDER


# --- regression: aggTrade identifier field -----------------------------------
#
# Binance sends `a` (aggregate trade id) on @aggTrade; `t` belongs to the
# @trade stream, which this pipeline does not subscribe to. The validator used
# to require `t`, so every real aggTrade event failed validation and was dropped
# before any decision was produced -- visible only as `normalized_drop_total`.


def test_agg_trade_with_binance_identifier_is_accepted() -> None:
    """A Binance-shaped aggTrade payload must normalize, not drop."""
    event = _raw(STREAM_AGG_TRADE, {"a": 77, "p": "100", "q": "1", "m": False, "T": 2000})
    normalized = EventNormalizer().normalize(event)
    assert normalized is not None
    assert normalized.trade_id == 77


def test_agg_trade_with_legacy_identifier_still_accepted() -> None:
    """Captures recorded under the earlier `t` assumption must still replay."""
    event = _raw(STREAM_AGG_TRADE, {"t": 42, "p": "100", "q": "1", "m": False, "T": 2000})
    normalized = EventNormalizer().normalize(event)
    assert normalized is not None
    assert normalized.trade_id == 42


def test_agg_trade_without_any_identifier_is_rejected() -> None:
    """Dropping the identifier entirely is still a validation failure."""
    event = _raw(STREAM_AGG_TRADE, {"p": "100", "q": "1", "m": False, "T": 2000})
    assert EventNormalizer().normalize(event) is None


def test_agg_trade_missing_price_is_still_rejected() -> None:
    """The alternative-key rule must not weaken the other requirements."""
    event = _raw(STREAM_AGG_TRADE, {"a": 1, "q": "1", "m": False, "T": 2000})
    assert EventNormalizer().normalize(event) is None
