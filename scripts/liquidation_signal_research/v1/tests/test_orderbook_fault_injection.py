from decimal import Decimal

from v1.book.local_book import LocalOrderBookEngine
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import DepthDeltaEvent


def test_out_of_order_stale_delta_is_ignored_without_desync() -> None:
    book = LocalOrderBookEngine(["BTCUSDT"])
    book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=200,
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("1"))],
            exchange_ts_ms=1,
        )
    )

    stale = DepthDeltaEvent(
        event_id="stale",
        schema_version="v1",
        symbol="BTCUSDT",
        exchange_ts_ms=2,
        recv_mono_ts_ns=3,
        first_update_id=190,
        final_update_id=195,
        prev_final_update_id=None,
        bid_deltas=[[Decimal("99"), Decimal("5")]],
        ask_deltas=[[Decimal("102"), Decimal("5")]],
        checksum=None,
        snapshot_ref=None,
    )

    applied = book.apply_depth_delta(stale)
    assert applied is True
    assert book.is_healthy("BTCUSDT") is True


def test_prev_update_mismatch_forces_desync() -> None:
    book = LocalOrderBookEngine(["BTCUSDT"])
    book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=200,
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("1"))],
            exchange_ts_ms=1,
        )
    )

    mismatch = DepthDeltaEvent(
        event_id="mismatch",
        schema_version="v1",
        symbol="BTCUSDT",
        exchange_ts_ms=3,
        recv_mono_ts_ns=4,
        first_update_id=203,
        final_update_id=204,
        prev_final_update_id=999,
        bid_deltas=[],
        ask_deltas=[],
        checksum=None,
        snapshot_ref=None,
    )

    applied = book.apply_depth_delta(mismatch)
    assert applied is False
    assert book.is_healthy("BTCUSDT") is False
    assert "prev_final_mismatch" in book.desync_reason("BTCUSDT")


def test_bridge_event_accepts_prev_id_mismatch_once_after_snapshot() -> None:
    book = LocalOrderBookEngine(["BTCUSDT"])
    book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=200,
            bids=[(Decimal("100"), Decimal("1"))],
            asks=[(Decimal("101"), Decimal("1"))],
            exchange_ts_ms=1,
        )
    )

    bridge = DepthDeltaEvent(
        event_id="bridge",
        schema_version="v1",
        symbol="BTCUSDT",
        exchange_ts_ms=2,
        recv_mono_ts_ns=3,
        first_update_id=199,
        final_update_id=201,
        prev_final_update_id=999,
        bid_deltas=[],
        ask_deltas=[],
        checksum=None,
        snapshot_ref=None,
    )

    applied = book.apply_depth_delta(bridge)
    assert applied is True
    assert book.is_healthy("BTCUSDT") is True
