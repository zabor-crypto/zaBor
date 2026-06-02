import json

from v1.contracts.ids import stable_payload_hash
from v1.store.jsonl_partition_store import JsonlPartitionStore
from v1.store.reader import JsonlPartitionReader
from v1.replay.replayer import DeterministicReplayer


def _raw_record(event_id: str, recv_mono: int, ingest_seq: int) -> dict:
    payload = {"stream": "btcusdt@aggTrade", "data": {"a": ingest_seq, "p": "100", "q": "1", "m": False, "T": 1000}}
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return {
        "event_id": event_id,
        "schema_version": "v1",
        "venue": "binance_um_futures",
        "stream": "aggTrade",
        "symbol": "BTCUSDT",
        "conn_id": "c",
        "exchange_ts_ms": 1000,
        "recv_wall_ts_ns": recv_mono,
        "recv_mono_ts_ns": recv_mono,
        "ingest_seq": ingest_seq,
        "stream_local_seq": ingest_seq,
        "payload_json": payload_json,
        "payload_hash": stable_payload_hash(payload),
        "parse_status": "ok",
        "gap_flag": False,
    }


def test_store_reader_and_replayer_are_deterministic(tmp_path) -> None:
    store_root = tmp_path / "raw"
    store = JsonlPartitionStore(store_root)
    store.append_record(
        schema_name="raw_market_event_v1",
        schema_version="v1",
        symbol="BTCUSDT",
        stream="aggTrade",
        event_ts_ms=1000,
        record=_raw_record("e2", recv_mono=20, ingest_seq=2),
    )
    store.append_record(
        schema_name="raw_market_event_v1",
        schema_version="v1",
        symbol="BTCUSDT",
        stream="aggTrade",
        event_ts_ms=1000,
        record=_raw_record("e1", recv_mono=10, ingest_seq=1),
    )
    store.close()

    reader = JsonlPartitionReader(store_root)
    ordered = list(reader.iter_records(schema_name="raw_market_event_v1"))
    assert [r["event_id"] for r in ordered] == ["e1", "e2"]

    replayer = DeterministicReplayer(store_root)

    def process(event):
        return {"event_id": event.event_id, "ingest_seq": event.ingest_seq}

    left = replayer.run(replay_run_id="r1", process_raw_event=process)
    right = replayer.run(replay_run_id="r2", process_raw_event=process)
    assert left.decisions == right.decisions
