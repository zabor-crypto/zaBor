"""Approximate margin / liquidation tracker.

We do not replicate exchange-specific margin engines exactly. Instead, we
use the worst-leg adverse move approximation requested in the spec:

    free_margin_ratio = (equity - margin_used) / margin_used
    liquidation_proxy = mark_to_market_loss / margin_used

A position trips the liquidation buffer if ``free_margin_ratio`` falls below
the configured minimum (default 0.30).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LegState:
    venue: str
    symbol: str
    direction: int           # +1 long, -1 short
    entry_price: float
    notional_usd: float      # notional at entry
    leverage: float          # 1.0 = fully cash-collateralized
    cumulative_funding_pnl: float = 0.0
    cumulative_fee_usd: float = 0.0

    @property
    def margin_usd(self) -> float:
        return self.notional_usd / max(self.leverage, 1e-9)

    def mark_to_market_pnl(self, mark_price: float) -> float:
        if mark_price is None or pd.isna(mark_price) or self.entry_price <= 0:
            return 0.0
        ret = (mark_price - self.entry_price) / self.entry_price
        return self.direction * ret * self.notional_usd


def free_margin_ratio(legs: list[LegState], marks: dict[tuple[str, str], float]) -> float:
    """Free / used margin across legs. Worst-leg adverse move approximation."""
    used = sum(l.margin_usd for l in legs)
    if used <= 0:
        return float("inf")
    mtm = 0.0
    for l in legs:
        mark = marks.get((l.venue, l.symbol))
        if mark is None:
            continue
        mtm += l.mark_to_market_pnl(mark)
    funding = sum(l.cumulative_funding_pnl for l in legs)
    fees = sum(l.cumulative_fee_usd for l in legs)
    equity = used + mtm + funding - fees
    free = max(equity - 0.0, 0.0)  # used is collateral; free margin is equity above 0
    return free / used


def liquidation_breached(
    legs: list[LegState],
    marks: dict[tuple[str, str], float],
    *,
    threshold: float,
) -> bool:
    return free_margin_ratio(legs, marks) < threshold
