from v1.contracts.ids import MonotonicSequencer, make_raw_event_id, stable_payload_hash


def test_stable_payload_hash_is_deterministic() -> None:
    payload = {"b": 2, "a": [1, 2, 3]}
    left = stable_payload_hash(payload)
    right = stable_payload_hash(payload)
    assert left == right


def test_make_raw_event_id_is_deterministic() -> None:
    payload_hash = stable_payload_hash({"x": 1})
    left = make_raw_event_id("binance", "aggTrade", "BTCUSDT", 1, 2, payload_hash)
    right = make_raw_event_id("binance", "aggTrade", "BTCUSDT", 1, 2, payload_hash)
    assert left == right


def test_monotonic_sequencer_increments() -> None:
    seq = MonotonicSequencer(start=10)
    assert seq.next() == 11
    assert seq.next() == 12
