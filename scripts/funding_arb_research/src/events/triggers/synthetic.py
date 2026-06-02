"""Synthetic trigger — engine smoke test only. Not used in production.

Fires every ``check_interval_hours`` for each coin in the universe,
emitting a constant-edge ``FundingEvent`` with a fixed cross-venue
funding gap. Used to verify the engine's heap, allocation, settlement,
and exit logic without depending on real-trigger correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..event import FundingEvent
from .base import MarketSnapshot, Trigger


@dataclass
class SyntheticTrigger(Trigger):
    coins: list[str] = field(default_factory=lambda: ["BTC"])
    check_interval_hours: int = 8
    venue_signal_template: dict[str, float] = field(default_factory=lambda: {
        "bitget": -0.50, "binance": 0.05,
    })
    expected_apr: float = 0.40
    name: str = "synthetic"

    def next_check_at(self, t: pd.Timestamp) -> pd.Timestamp:
        return t + pd.Timedelta(hours=self.check_interval_hours)

    def evaluate(self, snap: MarketSnapshot) -> list[FundingEvent]:
        return [FundingEvent(
            coin=c,
            timestamp_utc=snap.t,
            trigger_type="synthetic",
            venue_signal=dict(self.venue_signal_template),
            expected_funding_pnl_apr=self.expected_apr,
            confidence=0.9,
            metadata={"reason": "smoke"},
        ) for c in self.coins]
