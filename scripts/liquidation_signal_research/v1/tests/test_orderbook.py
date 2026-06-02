from decimal import Decimal

from v1.book.local_book import LocalOrderBookEngine
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import DepthDeltaEvent


def test_orderbook_applies_snapshot_and_delta() -> None:
    book = LocalOrderBookEngine(["BTCUSDT"])
    book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100"), Decimal("2"))],
            asks=[(Decimal("101"), Decimal("3"))],
            exchange_ts_ms=1,
        )
    )

    event = DepthDeltaEvent(
        event_id="d1",
        schema_version="v1",
        symbol="BTCUSDT",
        exchange_ts_ms=2,
        recv_mono_ts_ns=3,
        first_update_id=101,
        final_update_id=102,
        prev_final_update_id=100,
        bid_deltas=[[Decimal("100"), Decimal("1")]],
        ask_deltas=[[Decimal("101"), Decimal("4")]],
        checksum=None,
        snapshot_ref=None,
    )
    applied = book.apply_depth_delta(event)
    assert applied is True
    assert book.is_healthy("BTCUSDT") is True


def test_orderbook_gap_marks_desync() -> None:
    book = LocalOrderBookEngine(["BTCUSDT"])
    book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100"), Decimal("2"))],
            asks=[(Decimal("101"), Decimal("3"))],
            exchange_ts_ms=1,
        )
    )

    gap = DepthDeltaEvent(
        event_id="d2",
        schema_version="v1",
        symbol="BTCUSDT",
        exchange_ts_ms=3,
        recv_mono_ts_ns=4,
        first_update_id=110,
        final_update_id=111,
        prev_final_update_id=None,
        bid_deltas=[],
        ask_deltas=[],
        checksum=None,
        snapshot_ref=None,
    )
    applied = book.apply_depth_delta(gap)
    assert applied is False
    assert book.is_healthy("BTCUSDT") is False
