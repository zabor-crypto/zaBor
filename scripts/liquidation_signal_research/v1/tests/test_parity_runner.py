import json

from v1.contracts.ids import stable_payload_hash
from v1.replay.parity_runner import run_parity
from v1.store.jsonl_partition_store import JsonlPartitionStore


def test_parity_runner_outputs_report(tmp_path) -> None:
    data_root = tmp_path / "raw"
    telemetry_root = tmp_path / "telemetry"
    telemetry_root.mkdir(parents=True, exist_ok=True)

    payload = {"stream": "btcusdt@aggTrade", "data": {"a": 1, "p": "100", "q": "1", "m": False, "T": 1000}}
    record = {
        "event_id": "e1",
        "schema_version": "v1",
        "venue": "binance_um_futures",
        "stream": "aggTrade",
        "symbol": "BTCUSDT",
        "conn_id": "c",
        "exchange_ts_ms": 1000,
        "recv_wall_ts_ns": 100,
        "recv_mono_ts_ns": 100,
        "ingest_seq": 1,
        "stream_local_seq": 1,
        "payload_json": json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
        "payload_hash": stable_payload_hash(payload),
        "parse_status": "ok",
        "gap_flag": False,
    }
    store = JsonlPartitionStore(data_root)
    store.append_record(
        schema_name="raw_market_event_v1",
        schema_version="v1",
        symbol="BTCUSDT",
        stream="aggTrade",
        event_ts_ms=1000,
        record=record,
    )
    store.close()

    report = run_parity(
        data_root=data_root,
        telemetry_root=telemetry_root,
        symbols=["BTCUSDT"],
        start_ts_ms=None,
        end_ts_ms=None,
        replay_run_id="r1",
    )

    assert report["replay_run_id"] == "r1"
    assert report["raw_count"] == 1
    assert "parity" in report
