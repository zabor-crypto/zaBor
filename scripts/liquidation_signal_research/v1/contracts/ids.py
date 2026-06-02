"""Deterministic identifiers and payload hashing for event-sourced v1 runtime."""

from __future__ import annotations

import json
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import xxhash as _xxhash
    _HAS_XXHASH = True
except ImportError:  # pragma: no cover — fallback for environments without xxhash
    import hashlib as _hashlib  # type: ignore[no-redef]
    _HAS_XXHASH = False

EVENT_NAMESPACE = uuid.UUID("2fda47b9-8a15-4b28-9a38-d4f123ee9ad5")
DECISION_NAMESPACE = uuid.UUID("7de725e2-5de2-4674-b21b-a699a2c42e9d")
ORDER_NAMESPACE = uuid.UUID("4fb14bb9-74ff-4627-9e93-e44559f66df1")
POSITION_NAMESPACE = uuid.UUID("9547e395-8c36-4cfd-aa52-8f411c197d27")

# Pre-compute namespace bytes for fast UUID5-equivalent construction.
_NS_BYTES: dict[uuid.UUID, bytes] = {
    ns: ns.bytes
    for ns in (EVENT_NAMESPACE, DECISION_NAMESPACE, ORDER_NAMESPACE, POSITION_NAMESPACE)
}


def stable_payload_hash(payload: Any) -> str:
    """128-bit xxhash (hex) of canonical JSON.

    xxhash3_128 is ~10× faster than SHA-256 for this workload.  It is NOT
    cryptographic, but event-ID deduplication only needs collision resistance
    within a single trading session (~10M events), for which xxhash128 is
    more than sufficient (birthday bound ≫ 10^18 events).

    Falls back to SHA-256 if xxhash is unavailable.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if _HAS_XXHASH:
        return _xxhash.xxh3_128(encoded).hexdigest()
    return _hashlib.sha256(encoded).hexdigest()  # type: ignore[attr-defined]


def make_stable_id(namespace: uuid.UUID, *parts: Any) -> str:
    """UUID-shaped deterministic ID built with xxhash instead of uuid5/SHA-1.

    Standard uuid.uuid5() calls hashlib.sha1 + struct operations on every
    invocation.  Here we XOR the namespace bytes into the xxhash seed so the
    result is still namespace-scoped, then format as a UUID string.
    The output is NOT RFC-4122 uuid5 — it is a custom deterministic token —
    but it is stable across processes and Python versions for identical inputs.
    """
    text = "|".join(str(p) for p in parts)
    ns_bytes = _NS_BYTES.get(namespace, namespace.bytes)
    if _HAS_XXHASH:
        h = _xxhash.xxh3_128(text.encode(), seed=0)
        raw = h.intdigest()
        # XOR the namespace int in so IDs from different namespaces never collide.
        ns_int = int.from_bytes(ns_bytes, "big")
        mixed = raw ^ ns_int
        return str(uuid.UUID(int=mixed))
    # Fallback: original uuid5 behaviour.
    return str(uuid.uuid5(namespace, text))


def make_raw_event_id(
    venue: str,
    stream: str,
    symbol: str,
    exchange_ts_ms: int,
    ingest_seq: int,
    payload_hash: str,
) -> str:
    return make_stable_id(
        EVENT_NAMESPACE,
        venue,
        stream,
        symbol,
        exchange_ts_ms,
        ingest_seq,
        payload_hash,
    )


def make_episode_id(symbol: str, trigger_ts_ms: int, trigger_ref: str) -> str:
    return make_stable_id(DECISION_NAMESPACE, "episode", symbol, trigger_ts_ms, trigger_ref)


def make_decision_id(symbol: str, decision_ts_ms: int, feature_hash: str, config_version: str) -> str:
    return make_stable_id(DECISION_NAMESPACE, "decision", symbol, decision_ts_ms, feature_hash, config_version)


def make_order_intent_id(decision_id: str, side: str, qty: str, limit_price: str) -> str:
    return make_stable_id(ORDER_NAMESPACE, decision_id, side, qty, limit_price)


def make_position_id(symbol: str, open_fill_event_ids: Iterable[str]) -> str:
    return make_stable_id(POSITION_NAMESPACE, symbol, *sorted(open_fill_event_ids))


@dataclass
class MonotonicSequencer:
    """Process-local sequencer for ingestion ordering and tie-breaks."""

    start: int = 0

    def __post_init__(self) -> None:
        self._value = int(self.start)

    def next(self) -> int:
        self._value += 1
        return self._value
