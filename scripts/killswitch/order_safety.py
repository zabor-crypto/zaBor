#!/usr/bin/env python3
"""Order safety data structures for the kill-switch."""

from dataclasses import dataclass


@dataclass
class CloseInstruction:
    """Instruction to partially or fully close one futures position."""

    symbol: str
    side: str            # "long" | "short"
    contracts: float     # current total contracts (before applying close_fraction)
    close_fraction: float  # 0.0–1.0; 1.0 = full close
    reason: str
    risk_score: float
    pnl_delta_usdt: float
    unrealized_pnl_usdt: float
