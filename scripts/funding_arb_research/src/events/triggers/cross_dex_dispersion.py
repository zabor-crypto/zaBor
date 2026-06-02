"""Cross-venue funding-dispersion trigger.

Fires when, for a coin, the *signed* APR-equivalent funding spread
between any two allowed venues exceeds ``min_carry_apr``. The router
shorts the high-funding side and longs the low-funding side; signed
expected carry = ``max_apr - min_apr`` is positive by construction.

Crucially, this trigger does NOT bet on persistence: at each settle ts
we re-check the spread. The position exits the moment carry decays
below ``exit_threshold_apr`` (engine config). One captured settlement
on a 100%+ APR spread is the modal trade — closer to a high-Sharpe
short-duration carry harvest than a momentum bet.

Look-ahead discipline: trigger reads only ``timestamp_utc < t`` rows
from each panel. The "current" rate is the most-recent realized.

Cadence: the trigger asks to be re-evaluated at the soonest upcoming
settle across the considered venues × coins. If a venue is on 4h and
others on 8h, we wake at every 4h boundary.

Universe and pairs:
  ``coins`` is the canonical bases to scan. ``venues`` is the venues
  considered as potential legs. We enumerate all unordered pairs and
  emit at most one event per coin per check — the pair with the
  largest signed expected carry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import pandas as pd

from ..event import FundingEvent
from .base import MarketSnapshot, Trigger

_HOURS_PER_YEAR = 365.0 * 24.0


@dataclass
class CrossDexDispersionTrigger(Trigger):
    coins: list[str]
    venues: list[str] = field(default_factory=lambda: ["bitget", "binance", "bybit", "okx", "hyperliquid"])
    min_carry_apr: float = 0.30
    name: str = "cross_dex_dispersion"
    _next_check_cache: Optional[pd.Timestamp] = None

    def next_check_at(self, t: pd.Timestamp) -> pd.Timestamp:
        if self._next_check_cache is not None and self._next_check_cache > t:
            return self._next_check_cache
        return t + pd.Timedelta(hours=1)

    def evaluate(self, snap: MarketSnapshot) -> list[FundingEvent]:
        events: list[FundingEvent] = []
        candidates_for_next: list[pd.Timestamp] = []

        for coin in self.coins:
            # Per-venue latest realized apr at t (no look-ahead).
            apr_by_venue: dict[str, float] = {}
            for v in self.venues:
                sym = self._symbol(v, coin, snap)
                panel = snap.funding_panels.get((v, sym))
                if panel is None or panel.empty:
                    continue
                row = panel.iloc[-1]
                rate = row.get("funding_rate")
                if pd.isna(rate):
                    continue
                cad = self._cadence(v, sym, snap)
                apr_by_venue[v] = float(rate) * (_HOURS_PER_YEAR / cad)
                candidates_for_next.append(row["timestamp_utc"] + pd.Timedelta(hours=cad))
            if len(apr_by_venue) < 2:
                continue

            # Best pair = max signed (apr_high - apr_low).
            best_pair: Optional[tuple[str, str, float]] = None
            for v_a, v_b in combinations(apr_by_venue.keys(), 2):
                a, b = apr_by_venue[v_a], apr_by_venue[v_b]
                carry = abs(a - b)
                if best_pair is None or carry > best_pair[2]:
                    high, low = (v_a, v_b) if a > b else (v_b, v_a)
                    best_pair = (high, low, carry)
            if best_pair is None:
                continue
            high_v, low_v, carry = best_pair
            if carry < self.min_carry_apr:
                continue

            confidence = min(1.0, carry / (3.0 * self.min_carry_apr))
            events.append(FundingEvent(
                coin=coin,
                timestamp_utc=snap.t,
                trigger_type="cross_dex_dispersion",
                venue_signal=apr_by_venue,
                expected_funding_pnl_apr=carry,
                confidence=confidence,
                metadata={
                    "high_apr_venue": high_v,
                    "low_apr_venue": low_v,
                    "carry_apr": carry,
                    "min_carry_apr": self.min_carry_apr,
                },
            ))

        if candidates_for_next:
            self._next_check_cache = min(candidates_for_next)
        return events

    # ---- helpers (same idea as bitget_extreme; kept local for clarity) -- #

    def _cadence(self, venue: str, symbol: str, snap: MarketSnapshot) -> int:
        cm = snap.contracts_meta
        if cm is not None and not cm.empty:
            row = cm[(cm["venue"] == venue) & (cm["symbol"] == symbol)]
            if not row.empty and pd.notna(row.iloc[0].get("fund_interval_hours")):
                return int(row.iloc[0]["fund_interval_hours"])
        return 1 if venue == "hyperliquid" else 8

    def _symbol(self, venue: str, coin: str, snap: MarketSnapshot) -> str:
        cm = snap.contracts_meta
        if cm is not None and not cm.empty:
            row = cm[(cm["venue"] == venue) & (cm["base_asset"] == coin)]
            if not row.empty:
                return row.iloc[0]["symbol"]
        if venue == "hyperliquid":
            return coin
        if venue == "okx":
            return f"{coin}-USDT-SWAP"
        return f"{coin}USDT"
