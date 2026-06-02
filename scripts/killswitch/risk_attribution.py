#!/usr/bin/env python3
"""PnL attribution and risk ranking for the kill-switch.

Uses PnL *delta* (change over the trigger window) rather than current PnL,
so a position that was profitable but deteriorated is correctly counted as a
drawdown contributor even if its absolute PnL is still positive.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PositionSnapshot:
    ts: int
    scope: Any                     # Scope object from killswitch
    symbol: str
    side: str                      # "long" | "short"
    contracts: float
    notional_usdt: float
    entry_price: Optional[float]
    mark_price: Optional[float]
    liquidation_price: Optional[float]
    margin_usdt: Optional[float]
    leverage: Optional[float]
    unrealized_pnl_usdt: float
    pnl_pct_on_margin: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionResult:
    source: str               # "LONG" | "SHORT" | "MIXED" | "UNKNOWN"
    long_negative_delta: float
    short_negative_delta: float
    total_negative_delta: float
    long_pct: float
    short_pct: float


def compute_pnl_attribution(
    current_positions: List[PositionSnapshot],
    reference_positions: List[PositionSnapshot],
    source_threshold: float = 0.65,
) -> AttributionResult:
    """Determine which side is responsible for the current drawdown.

    A position still positive but with pnl_delta < 0 IS counted (e.g. SHORT SOL
    was +800 USDT and is now +100 USDT contributes -700 USDT to the short side).

    Args:
        current_positions:   Positions as of now.
        reference_positions: Positions near start of the trigger window (or oldest
                             available snapshot within the lookback period).
        source_threshold:    Fraction of total negative delta on one side that
                             classifies it as the dominant source.
    """
    ref_map: Dict[Tuple[str, str], float] = {
        (p.symbol, p.side): p.unrealized_pnl_usdt for p in reference_positions
    }

    long_neg = 0.0
    short_neg = 0.0

    for pos in current_positions:
        ref_pnl = ref_map.get((pos.symbol, pos.side), 0.0)
        delta = pos.unrealized_pnl_usdt - ref_pnl
        if delta < 0:
            if pos.side == "long":
                long_neg += abs(delta)
            else:
                short_neg += abs(delta)

    total = long_neg + short_neg
    if total <= 0:
        return AttributionResult("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)

    long_pct = long_neg / total
    short_pct = short_neg / total

    if short_pct >= source_threshold:
        source = "SHORT"
    elif long_pct >= source_threshold:
        source = "LONG"
    else:
        source = "MIXED"

    return AttributionResult(source, long_neg, short_neg, total, long_pct, short_pct)


def _liquidation_risk_score(pos: PositionSnapshot) -> float:
    """0.0 = safe, 1.0 = at/past liquidation price."""
    if not pos.mark_price or not pos.liquidation_price:
        return 0.0
    if pos.mark_price <= 0 or pos.liquidation_price <= 0:
        return 0.0
    if pos.side == "long":
        dist_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price
    else:
        dist_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price
    if dist_pct <= 0:
        return 1.0
    # Score increases steeply as distance shrinks below 10%
    return max(0.0, min(1.0, 0.1 / dist_pct))


def compute_risk_score(
    pos: PositionSnapshot,
    pnl_delta: float,
    account_equity_usdt: float,
) -> float:
    """Composite risk score for one position.

    Components:
      0.40 × |pnl_delta|          — contribution to current drawdown
      0.25 × |min(unrealized,0)|  — current absolute loss
      0.20 × liquidation_risk     — proximity to forced liquidation
      0.10 × notional_share       — size relative to account
      0.05 × margin_loss          — fraction of margin already lost
    """
    liq_risk = _liquidation_risk_score(pos)
    notional_share = (
        min(pos.notional_usdt / account_equity_usdt, 1.0)
        if account_equity_usdt > 0
        else 0.0
    )
    margin_loss = 0.0
    if pos.margin_usdt and pos.margin_usdt > 0:
        margin_loss = min(abs(pos.unrealized_pnl_usdt) / pos.margin_usdt, 1.0)

    return (
        abs(pnl_delta) * 0.40
        + abs(min(pos.unrealized_pnl_usdt, 0.0)) * 0.25
        + liq_risk * 0.20
        + notional_share * 0.10
        + margin_loss * 0.05
    )


def rank_positions_by_risk(
    current_positions: List[PositionSnapshot],
    reference_positions: List[PositionSnapshot],
    account_equity_usdt: float,
    source: str,
    source_mode: str = "AUTO",
) -> List[Tuple[PositionSnapshot, float, float]]:
    """Return (position, pnl_delta, risk_score) sorted by risk_score descending.

    Candidates are filtered by direction based on source/source_mode:
      source_mode=AUTO  -> filter by attribution source (LONG/SHORT/MIXED)
      source_mode=LONG  -> only long positions
      source_mode=SHORT -> only short positions
      source_mode=MIXED -> all positions
    """
    ref_map: Dict[Tuple[str, str], float] = {
        (p.symbol, p.side): p.unrealized_pnl_usdt for p in reference_positions
    }

    effective = source if source_mode == "AUTO" else source_mode

    candidates = []
    for pos in current_positions:
        if effective == "LONG" and pos.side != "long":
            continue
        if effective == "SHORT" and pos.side != "short":
            continue
        # MIXED or UNKNOWN: include all

        ref_pnl = ref_map.get((pos.symbol, pos.side), 0.0)
        pnl_delta = pos.unrealized_pnl_usdt - ref_pnl
        score = compute_risk_score(pos, pnl_delta, account_equity_usdt)
        candidates.append((pos, pnl_delta, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
