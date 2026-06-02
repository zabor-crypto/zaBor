"""HL-vs-CEX cross-venue funding dispersion (exploratory).

Premise:
  Hyperliquid's ``predictedFundings`` payload exposes the next-period
  predicted funding rate for HL itself + Binance / Bybit / OKX in one call.
  At any given timestamp we can rank the venues by predicted-funding APR for
  a single coin and trade the widest spread (long the lowest, short the
  highest) — fully delta-neutral.

This differs from MVP-1 (HL/Binance interval) in three ways:

  1. Universe: any coin that exists on at least 2 of {hl, binance, bybit, okx}.
  2. Pair selection: dynamic — chooses the widest cross-venue spread per
     coin, per tick, instead of fixing HL/Binance.
  3. Entry threshold: applied on the *cross-venue* spread, which is
     mechanically larger than HL/Binance alone.

Cost reality: with maker rates the round-trip is ~6 bp; the spread needs to
be wide enough that even an 8-hour hold accrues > 6 bp of funding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..backtest.engine import CloseAll, OpenLeg, StrategyState
from ..backtest.funding_settlement import select_predicted_funding
from ..utils.time import annualize_funding


VENUE_INTERVALS_H = {
    "hyperliquid": 1,
    "binance": 8,
    "bybit": 8,
    "okx": 8,
}


def _venue_symbol(venue: str, base: str) -> str:
    if venue == "hyperliquid":
        return base
    if venue == "okx":
        return f"{base}-USDT-SWAP"
    return f"{base}USDT"


def _expected_apr(panel: pd.DataFrame, *, as_of: pd.Timestamp, interval_h: int) -> Optional[float]:
    rate = select_predicted_funding(panel, as_of=as_of)
    if rate is None:
        return None
    return annualize_funding(rate, interval_h)


@dataclass
class _OpenState:
    base: str
    venue_long: str
    venue_short: str
    entry_spread_apr: float


class HLCrossVenueDispersionStrategy:
    """Trade widest cross-venue funding spread per coin per tick."""

    def __init__(self, venues: tuple[str, ...] = ("hyperliquid", "binance", "bybit", "okx")):
        self.venues = venues
        self._open: Optional[_OpenState] = None

    def _venue_apr(self, state: StrategyState, venue: str, base: str) -> Optional[float]:
        sym = _venue_symbol(venue, base)
        panel = state.funding.get((venue, sym))
        if panel is None or panel.empty:
            return None
        return _expected_apr(
            panel, as_of=state.now,
            interval_h=VENUE_INTERVALS_H.get(venue, 8),
        )

    def __call__(self, state: StrategyState):
        cfg = state.config
        universe: list[str] = list(cfg.get("universe", ["BTC", "ETH", "SOL"]))
        entry_th = float(cfg.get("entry_threshold_apr", 0.30))
        exit_th = float(cfg.get("exit_threshold_apr", 0.05))
        per_trade_pct = float(cfg.get("per_trade_capital_pct", 0.10))
        role = cfg.get("role", "maker")

        if state.open_legs and self._open is not None:
            return self._maybe_exit(state, exit_th)

        # Find widest spread across the cross-section.
        best = None  # (abs_spread, base, venue_long, venue_short, spread)
        for base in universe:
            aprs: dict[str, float] = {}
            for v in self.venues:
                a = self._venue_apr(state, v, base)
                if a is not None:
                    aprs[v] = a
            if len(aprs) < 2:
                continue
            v_max = max(aprs, key=aprs.get)
            v_min = min(aprs, key=aprs.get)
            spread = aprs[v_max] - aprs[v_min]
            if spread <= 0 or v_max == v_min:
                continue
            if best is None or spread > best[0]:
                best = (spread, base, v_min, v_max, spread)

        if best is None or best[0] < entry_th:
            return None

        spread, base, v_long, v_short, _ = best
        per_leg = state.equity_usd * per_trade_pct
        legs = [
            OpenLeg(
                venue=v_long, direction=+1, symbol=_venue_symbol(v_long, base),
                notional_usd=per_leg, role=role,
                tag=f"long={v_long} short={v_short} spread_apr={spread:.4f}",
            ),
            OpenLeg(
                venue=v_short, direction=-1, symbol=_venue_symbol(v_short, base),
                notional_usd=per_leg, role=role,
                tag=f"long={v_long} short={v_short} spread_apr={spread:.4f}",
            ),
        ]
        self._open = _OpenState(
            base=base, venue_long=v_long, venue_short=v_short,
            entry_spread_apr=spread,
        )
        return legs

    def _maybe_exit(self, state, exit_th):
        op = self._open
        assert op is not None
        a_long = self._venue_apr(state, op.venue_long, op.base)
        a_short = self._venue_apr(state, op.venue_short, op.base)
        if a_long is None or a_short is None:
            return None
        spread = a_short - a_long
        if abs(spread) < exit_th:
            self._open = None
            return CloseAll(reason="spread_below_exit")
        # Sign flip ⇒ exit immediately (signal direction inverted).
        if spread < 0:
            self._open = None
            return CloseAll(reason="sign_flip")
        return None
