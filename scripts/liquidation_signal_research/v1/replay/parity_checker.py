"""Replay/live parity helpers for deterministic decision-path validation."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Tuple


def stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decision_fingerprint(decision: Mapping[str, object]) -> str:
    subset = {
        "decision_id": decision.get("decision_id"),
        "symbol": decision.get("symbol"),
        "decision_ts_ms": decision.get("decision_ts_ms"),
        "action": decision.get("action"),
        "score": round(float(decision.get("score", 0.0)), 8),
        "score_components": decision.get("score_components", {}),
    }
    return stable_hash(subset)


def compare_decision_streams(
    live: Iterable[Mapping[str, object]],
    replay: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    live_hashes = [decision_fingerprint(item) for item in live]
    replay_hashes = [decision_fingerprint(item) for item in replay]

    matched = 0
    mismatches: List[Tuple[int, str, str]] = []
    limit = min(len(live_hashes), len(replay_hashes))
    for idx in range(limit):
        left = live_hashes[idx]
        right = replay_hashes[idx]
        if left == right:
            matched += 1
        else:
            mismatches.append((idx, left, right))

    parity_ratio = 1.0
    denom = max(len(live_hashes), len(replay_hashes), 1)
    parity_ratio = matched / denom

    return {
        "live_count": len(live_hashes),
        "replay_count": len(replay_hashes),
        "matched_count": matched,
        "parity_ratio": parity_ratio,
        "mismatches": mismatches[:25],
    }
