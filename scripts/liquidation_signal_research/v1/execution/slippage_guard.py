"""Slippage guard for aggressive-limit orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class SlippageCheck:
    allowed: bool
    expected_slippage_bps: float
    reason: str = ""


class SlippageGuard:
    def check(
        self,
        *,
        side: str,
        limit_price: Decimal,
        best_bid: Decimal,
        best_ask: Decimal,
        max_slippage_bps: float,
    ) -> SlippageCheck:
        if best_ask <= 0 or best_bid <= 0:
            return SlippageCheck(False, 9999.0, "invalid_tob")

        if side == "BUY":
            expected = ((limit_price - best_ask) / best_ask) * Decimal("10000")
        else:
            expected = ((best_bid - limit_price) / best_bid) * Decimal("10000")

        expected_bps = max(0.0, float(expected))
        allowed = expected_bps <= float(max_slippage_bps)
        reason = "" if allowed else "slippage_cap_exceeded"
        return SlippageCheck(allowed=allowed, expected_slippage_bps=expected_bps, reason=reason)
