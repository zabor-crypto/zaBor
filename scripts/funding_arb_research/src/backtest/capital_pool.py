"""Single shared capital pool with EV-priority allocation.

Locked design rule: priority for a candidate event is

    priority = expected_funding_pnl_apr / round_trip_cost_apr

This is the Sharpe-shaped, units-free score the user requested. A
trigger that fires on a 200% APR event with 50% APR-equivalent costs
(priority=4) outranks one firing on 60% APR with 5% costs (priority=12)
*only if* the second has a higher score — so the formula correctly
rewards both raw edge and cost efficiency.

When free capital is scarce, ``allocate_batch`` greedily takes events
in priority order until the pool is exhausted. Pre-existing open
positions are not re-evaluated — capital they hold is locked until
they close.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from ..events.event import FundingEvent, Position
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.backtest.capital_pool")


@dataclass(frozen=True)
class AllocationRequest:
    event: FundingEvent
    requested_capital_usd: float
    estimated_round_trip_cost_apr: float

    def priority(self) -> float:
        """EV-priority. Higher = better."""
        cost = max(self.estimated_round_trip_cost_apr, 1e-6)
        return (self.event.expected_funding_pnl_apr * self.event.confidence) / cost


@dataclass
class AllocationResult:
    event: FundingEvent
    approved_capital_usd: float
    skipped_reason: Optional[str] = None


@dataclass
class CapitalPool:
    total_capital_usd: float
    per_event_max_pct: float = 0.10           # cap one event at 10% of total
    min_event_capital_usd: float = 1_000.0    # below this, skip (rounding noise)
    used_capital_usd: float = 0.0

    def free_capital_usd(self) -> float:
        return max(self.total_capital_usd - self.used_capital_usd, 0.0)

    def reserve(self, capital_usd: float) -> None:
        self.used_capital_usd += capital_usd

    def release(self, capital_usd: float) -> None:
        self.used_capital_usd = max(0.0, self.used_capital_usd - capital_usd)

    def per_event_cap_usd(self) -> float:
        return self.total_capital_usd * self.per_event_max_pct

    def allocate_batch(self, requests: Iterable[AllocationRequest]) -> list[AllocationResult]:
        """Greedy, EV-priority-ordered allocation against current free capital.

        Each request's approved capital is the min of:
          - requested,
          - per-event cap,
          - free capital remaining.

        Requests below ``min_event_capital_usd`` after capping are skipped.
        """
        ranked = sorted(requests, key=AllocationRequest.priority, reverse=True)
        results: list[AllocationResult] = []
        free = self.free_capital_usd()
        cap = self.per_event_cap_usd()
        for req in ranked:
            if free <= self.min_event_capital_usd:
                results.append(AllocationResult(req.event, 0.0, "no_free_capital"))
                continue
            allowed = min(req.requested_capital_usd, cap, free)
            if allowed < self.min_event_capital_usd:
                results.append(AllocationResult(req.event, 0.0, "below_min_event_capital"))
                continue
            self.reserve(allowed)
            free -= allowed
            results.append(AllocationResult(req.event, allowed))
        return results

    def release_for_position(self, position: Position) -> None:
        self.release(position.used_capital_usd())
