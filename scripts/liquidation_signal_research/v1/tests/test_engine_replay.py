import json
from decimal import Decimal

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import RawMarketEvent
from v1.contracts.ids import stable_payload_hash


def _raw(stream: str, symbol: str, ts_ms: int, ingest_seq: int, payload: dict) -> RawMarketEvent:
    wrapped = {"stream": f"{symbol.lower()}@{stream}", "data": payload}
    payload_json = json.dumps(wrapped, separators=(",", ":"), ensure_ascii=True)
    return RawMarketEvent(
        event_id=f"{stream}-{ingest_seq}",
        schema_version="v1",
        venue="binance_um_futures",
        stream=stream,
        symbol=symbol,
        conn_id="replay",
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


def _run_once(tmp_path):
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        decision_interval_ms=200,
    )
    engine = ContinuationEngine(cfg)
    engine.book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100.0"), Decimal("3.0"))],
            asks=[(Decimal("100.1"), Decimal("3.0"))],
            exchange_ts_ms=900,
        )
    )

    events = [
        _raw("depth", "BTCUSDT", 1000, 1, {"U": 101, "u": 101, "pu": 100, "b": [["100.0", "3.5"]], "a": [["100.1", "2.5"]]}),
        _raw("aggTrade", "BTCUSDT", 1100, 2, {"a": 1, "p": "100.1", "q": "10", "m": False, "T": 1100}),
        _raw("forceOrder", "BTCUSDT", 1200, 3, {"o": {"S": "SELL", "q": "20", "p": "100.0", "ap": "100.0", "X": "FILLED", "T": 1200}}),
        _raw("depth", "BTCUSDT", 1300, 4, {"U": 102, "u": 102, "pu": 101, "b": [["100.0", "1.2"]], "a": [["100.1", "0.8"]]}),
        _raw("markPrice", "BTCUSDT", 1400, 5, {"p": "100.0", "E": 1400, "r": "0.0001", "T": 2000}),
        _raw("aggTrade", "BTCUSDT", 1600, 6, {"a": 2, "p": "99.9", "q": "8", "m": True, "T": 1600}),
    ]

    out = []
    for event in events:
        decision = engine.process_raw_event_for_replay(event)
        if decision is not None:
            out.append(decision)
    return out


def test_engine_replay_is_deterministic(tmp_path) -> None:
    left = _run_once(tmp_path / "left")
    right = _run_once(tmp_path / "right")
    assert left == right
