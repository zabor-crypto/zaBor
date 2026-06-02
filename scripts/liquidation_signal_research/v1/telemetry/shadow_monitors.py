"""Shadow-mode monitoring and rollout summary helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


_SCORE_BUCKETS = [
    (-10.0, 0.0),
    (0.0, 0.15),
    (0.15, 0.30),
    (0.30, 0.50),
    (0.50, 1.00),
    (1.00, 10.0),
]


def _score_bucket(score: float) -> str:
    for lo, hi in _SCORE_BUCKETS:
        if lo <= score < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "[overflow]"


@dataclass
class ShadowSummary:
    total_decisions: int
    action_counts: Dict[str, int]
    no_trade_reason_counts: Dict[str, int]
    risk_reason_counts: Dict[str, int]
    score_bucket_counts: Dict[str, int]
    desync_events: int
    resync_attempts: int
    resync_success: int
    resync_failures: int


class ShadowMonitor:
    """Collects shadow-run distributions for rollout gating."""

    def __init__(self) -> None:
        self.total_decisions = 0
        self.action_counts: Counter[str] = Counter()
        self.no_trade_reason_counts: Counter[str] = Counter()
        self.risk_reason_counts: Counter[str] = Counter()
        self.score_bucket_counts: Counter[str] = Counter()

        self.desync_events = 0
        self.resync_attempts = 0
        self.resync_success = 0
        self.resync_failures = 0

    @staticmethod
    def _split_reason_blob(reason_blob: str) -> List[str]:
        if not reason_blob:
            return []
        return [part.strip() for part in reason_blob.split(";") if len(part.strip()) > 0]

    def observe_decision(self, payload: Mapping[str, object]) -> None:
        self.total_decisions += 1
        action = str(payload.get("action", "UNKNOWN"))
        self.action_counts[action] += 1

        score = float(payload.get("score", 0.0) or 0.0)
        self.score_bucket_counts[_score_bucket(score)] += 1

        no_trade_reason = str(payload.get("no_trade_reason", "") or "")
        for reason in self._split_reason_blob(no_trade_reason):
            self.no_trade_reason_counts[reason] += 1

        for reason in payload.get("risk_reasons", []) or []:
            self.risk_reason_counts[str(reason)] += 1

    def observe_risk_event(self, payload: Mapping[str, object]) -> None:
        event = str(payload.get("event", ""))
        if event == "book_desync":
            self.desync_events += 1
        elif event == "book_resync_start":
            self.resync_attempts += 1
        elif event == "book_resync_ok":
            self.resync_success += 1
        elif event == "book_resync_failed":
            self.resync_failures += 1

    def summary(self) -> ShadowSummary:
        return ShadowSummary(
            total_decisions=self.total_decisions,
            action_counts=dict(self.action_counts),
            no_trade_reason_counts=dict(self.no_trade_reason_counts),
            risk_reason_counts=dict(self.risk_reason_counts),
            score_bucket_counts=dict(self.score_bucket_counts),
            desync_events=self.desync_events,
            resync_attempts=self.resync_attempts,
            resync_success=self.resync_success,
            resync_failures=self.resync_failures,
        )
