import asyncio
import json
from decimal import Decimal

import pytest

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import RawMarketEvent
from v1.contracts.ids import stable_payload_hash


class FakeSnapshotClient:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []

    async def fetch_snapshot(self, symbol: str, limit: int = 1000):
        self.calls.append((symbol, limit))
        return self.snapshots[symbol]


def _depth_raw(symbol: str, ts_ms: int, ingest_seq: int, payload: dict) -> RawMarketEvent:
    wrapped = {"stream": f"{symbol.lower()}@depth", "data": payload}
    payload_json = json.dumps(wrapped, separators=(",", ":"), ensure_ascii=True)
    return RawMarketEvent(
        event_id=f"depth-{ingest_seq}",
        schema_version="v1",
        venue="binance_um_futures",
        stream="depth",
        symbol=symbol,
        conn_id="test",
        exchange_ts_ms=ts_ms,
        recv_wall_ts_ns=ts_ms * 1_000_000,
        recv_mono_ts_ns=ts_ms * 1_000_000,
        ingest_seq=ingest_seq,
        stream_local_seq=ingest_seq,
        payload_json=payload_json,
        payload_hash=stable_payload_hash(wrapped),
        parse_status="ok",
        gap_flag=False,
    )


@pytest.mark.asyncio
async def test_bootstrap_snapshots_marks_books_ready(tmp_path) -> None:
    snapshots = {
        "BTCUSDT": OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100.0"), Decimal("2.0"))],
            asks=[(Decimal("100.1"), Decimal("2.0"))],
            exchange_ts_ms=1_000,
        ),
        "ETHUSDT": OrderBookSnapshot(
            symbol="ETHUSDT",
            last_update_id=200,
            bids=[(Decimal("2000.0"), Decimal("3.0"))],
            asks=[(Decimal("2000.2"), Decimal("3.0"))],
            exchange_ts_ms=1_000,
        ),
    }
    fake = FakeSnapshotClient(snapshots)
    engine = ContinuationEngine(
        RuntimeConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            streams=["forceOrder", "aggTrade", "depth", "markPrice"],
            data_root=tmp_path / "raw",
            telemetry_root=tmp_path / "telemetry",
            snapshot_resync_cooldown_ms=0,
        ),
        snapshot_client=fake,
    )

    await engine._bootstrap_snapshots()

    assert len(fake.calls) == 2
    assert engine._book_ready("BTCUSDT") is True
    assert engine._book_ready("ETHUSDT") is True


@pytest.mark.asyncio
async def test_live_gap_triggers_snapshot_resync(tmp_path) -> None:
    snapshots = {
        "BTCUSDT": OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=300,
            bids=[(Decimal("100.0"), Decimal("1.5"))],
            asks=[(Decimal("100.1"), Decimal("1.5"))],
            exchange_ts_ms=2_000,
        )
    }
    fake = FakeSnapshotClient(snapshots)
    engine = ContinuationEngine(
        RuntimeConfig(
            symbols=["BTCUSDT"],
            streams=["forceOrder", "aggTrade", "depth", "markPrice"],
            data_root=tmp_path / "raw",
            telemetry_root=tmp_path / "telemetry",
            snapshot_resync_cooldown_ms=0,
        ),
        snapshot_client=fake,
    )

    engine.book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("99.9"), Decimal("3.0"))],
            asks=[(Decimal("100.0"), Decimal("3.0"))],
            exchange_ts_ms=1_000,
        )
    )
    engine._sync_pending["BTCUSDT"] = False

    gap = _depth_raw(
        symbol="BTCUSDT",
        ts_ms=1_100,
        ingest_seq=1,
        payload={"U": 150, "u": 151, "pu": 50, "b": [["99.9", "1.0"]], "a": [["100.0", "1.0"]]},
    )

    engine._process_raw_event_common(gap, assume_stream_healthy=False)
    await asyncio.sleep(0.05)

    assert len(fake.calls) >= 1
    assert engine._book_ready("BTCUSDT") is True


@pytest.mark.asyncio
async def test_replay_path_does_not_trigger_snapshot_resync(tmp_path) -> None:
    snapshots = {
        "BTCUSDT": OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=300,
            bids=[(Decimal("100.0"), Decimal("1.5"))],
            asks=[(Decimal("100.1"), Decimal("1.5"))],
            exchange_ts_ms=2_000,
        )
    }
    fake = FakeSnapshotClient(snapshots)
    engine = ContinuationEngine(
        RuntimeConfig(
            symbols=["BTCUSDT"],
            streams=["forceOrder", "aggTrade", "depth", "markPrice"],
            data_root=tmp_path / "raw",
            telemetry_root=tmp_path / "telemetry",
            snapshot_resync_cooldown_ms=0,
        ),
        snapshot_client=fake,
    )

    engine.book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("99.9"), Decimal("3.0"))],
            asks=[(Decimal("100.0"), Decimal("3.0"))],
            exchange_ts_ms=1_000,
        )
    )
    engine._sync_pending["BTCUSDT"] = False

    gap = _depth_raw(
        symbol="BTCUSDT",
        ts_ms=1_100,
        ingest_seq=1,
        payload={"U": 150, "u": 151, "pu": 50, "b": [["99.9", "1.0"]], "a": [["100.0", "1.0"]]},
    )

    engine._process_raw_event_common(gap, assume_stream_healthy=True)
    await asyncio.sleep(0.01)

    assert fake.calls == []
    assert engine._book_ready("BTCUSDT") is False


@pytest.mark.asyncio
async def test_sync_pending_depth_buffer_replays_after_snapshot(tmp_path) -> None:
    snapshots = {
        "BTCUSDT": OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=200,
            bids=[(Decimal("100.0"), Decimal("1.0"))],
            asks=[(Decimal("100.1"), Decimal("1.0"))],
            exchange_ts_ms=2_000,
        )
    }
    fake = FakeSnapshotClient(snapshots)
    engine = ContinuationEngine(
        RuntimeConfig(
            symbols=["BTCUSDT"],
            streams=["forceOrder", "aggTrade", "depth", "markPrice"],
            data_root=tmp_path / "raw",
            telemetry_root=tmp_path / "telemetry",
            snapshot_resync_cooldown_ms=0,
        ),
        snapshot_client=fake,
    )

    buffered = _depth_raw(
        symbol="BTCUSDT",
        ts_ms=2_050,
        ingest_seq=1,
        payload={"U": 199, "u": 201, "pu": 999, "b": [["100.0", "2.0"]], "a": [["100.1", "2.0"]]},
    )

    engine._process_raw_event_common(buffered, assume_stream_healthy=False)
    await asyncio.sleep(0.05)

    assert len(fake.calls) >= 1
    assert engine._book_ready("BTCUSDT") is True
    assert engine.metrics.counters.get("book_resync_replay_applied_total", 0.0) >= 1.0
