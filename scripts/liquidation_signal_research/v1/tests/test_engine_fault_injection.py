import asyncio
import json
from decimal import Decimal

import pytest

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import RawMarketEvent
from v1.contracts.ids import stable_payload_hash


class FailingSnapshotClient:
    async def fetch_snapshot(self, symbol: str, limit: int = 1000):
        raise RuntimeError(f"snapshot_unavailable:{symbol}")


def _raw(stream: str, payload: dict, ts_ms: int, seq: int) -> RawMarketEvent:
    wrapped = {"stream": f"btcusdt@{stream}", "data": payload}
    return RawMarketEvent(
        event_id=f"{stream}-{seq}",
        schema_version="v1",
        venue="binance_um_futures",
        stream=stream,
        symbol="BTCUSDT",
        conn_id="fault-test",
        exchange_ts_ms=ts_ms,
        recv_wall_ts_ns=ts_ms * 1_000_000,
        recv_mono_ts_ns=ts_ms * 1_000_000,
        ingest_seq=seq,
        stream_local_seq=seq,
        payload_json=json.dumps(wrapped, separators=(",", ":"), ensure_ascii=True),
        payload_hash=stable_payload_hash(wrapped),
        parse_status="ok",
        gap_flag=False,
    )


@pytest.mark.asyncio
async def test_gap_fault_keeps_engine_in_no_trade_state_until_resync(tmp_path) -> None:
    engine = ContinuationEngine(
        RuntimeConfig(
            symbols=["BTCUSDT"],
            streams=["forceOrder", "aggTrade", "depth", "markPrice"],
            data_root=tmp_path / "raw",
            telemetry_root=tmp_path / "telemetry",
            snapshot_resync_cooldown_ms=0,
            decision_interval_ms=1,
        ),
        snapshot_client=FailingSnapshotClient(),
    )

    engine.book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100.0"), Decimal("2.0"))],
            asks=[(Decimal("100.1"), Decimal("2.0"))],
            exchange_ts_ms=1_000,
        )
    )
    engine._sync_pending["BTCUSDT"] = False

    gap = _raw(
        "depth",
        {"U": 150, "u": 151, "pu": 50, "b": [["100.0", "1.0"]], "a": [["100.1", "1.0"]]},
        ts_ms=1_100,
        seq=1,
    )
    decision = engine._process_raw_event_common(gap, assume_stream_healthy=False)
    await asyncio.sleep(0.02)

    assert decision is not None
    assert decision["risk_approved"] is False
    assert engine._book_ready("BTCUSDT") is False
    assert engine._sync_pending["BTCUSDT"] is True
    assert any("book_desynced_or_stale" == reason for reason in decision["risk_reasons"])

    for task in list(engine._resync_tasks.values()):
        task.cancel()
    if engine._resync_tasks:
        await asyncio.gather(*engine._resync_tasks.values(), return_exceptions=True)


def test_engine_fails_fast_when_calibration_gate_enabled_without_path(tmp_path) -> None:
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        execution_paper_mode=True,
        shadow_mode=False,
        enable_paper_calibration_gate=True,
        calibration_overrides_path=None,
    )
    with pytest.raises(RuntimeError):
        ContinuationEngine(cfg)


def test_engine_fails_fast_on_invalid_calibration_overrides(tmp_path) -> None:
    bad = tmp_path / "bad_overrides.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": "bad_schema",
                "symbols": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        execution_paper_mode=True,
        shadow_mode=False,
        enable_paper_calibration_gate=True,
        calibration_overrides_path=bad,
    )
    with pytest.raises(RuntimeError):
        ContinuationEngine(cfg)
