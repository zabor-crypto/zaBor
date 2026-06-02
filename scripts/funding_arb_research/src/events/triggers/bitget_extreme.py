"""Bitget extreme-funding trigger.

Fires when the most-recent realized Bitget funding rate, annualized via
the symbol's actual cadence (8h or 4h on Bitget), exceeds
``abs_threshold_apr``. The bet is on auto-correlation: extreme last
period → likely extreme this period. The backtest's job is to test
whether that persistence is real *after costs*, not assume it.

Look-ahead discipline: the trigger reads only rows strictly < ``t`` from
the funding panel passed in via ``MarketSnapshot``. The "predicted"
rate at time ``t`` is the most-recent observed realized rate. We never
peek at the next-settle row.

Hedge-venue signal: for each allowed hedge venue we inject the
most-recent realized apr at ``t``. The router uses this to pick the
cheapest fundable counter-leg.

Cadence: looked up from ``snap.contracts_meta`` per (venue, symbol).
Falls back to 8h when missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..event import FundingEvent
from .base import MarketSnapshot, Trigger

_HOURS_PER_YEAR = 365.0 * 24.0


@dataclass
class BitgetExtremeTrigger(Trigger):
    """Detects |predicted Bitget funding| ≥ ``abs_threshold_apr`` per symbol."""

    coins: list[str]                                        # canonical bases, e.g. ["BTC","DOGE",...]
    abs_threshold_apr: float = 1.00                         # 100% APR default
    hedge_venues: list[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "hyperliquid"])
    name: str = "bitget_extreme"
    # Engine pacing: ask to be re-checked after every funding settle on Bitget
    # (the cadence varies per coin: 8h or 4h). We wake at the soonest upcoming
    # settle across the universe.
    _last_eval_ts: Optional[pd.Timestamp] = None
    _next_check_cache: Optional[pd.Timestamp] = None
    _funding_panels_ref: Optional[dict] = None              # set by evaluate

    def next_check_at(self, t: pd.Timestamp) -> pd.Timestamp:
        # Without panel access here we just ask for a sane default. The engine
        # will pump us at this cadence; evaluate() does the precise per-symbol
        # filtering against actual settle rows.
        if self._next_check_cache is not None and self._next_check_cache > t:
            return self._next_check_cache
        return t + pd.Timedelta(hours=1)

    def evaluate(self, snap: MarketSnapshot) -> list[FundingEvent]:
        self._funding_panels_ref = snap.funding_panels
        events: list[FundingEvent] = []
        next_settle_candidates: list[pd.Timestamp] = []

        for coin in self.coins:
            sym = self._symbol("bitget", coin, snap)
            panel = snap.funding_panels.get(("bitget", sym))
            if panel is None or panel.empty:
                continue
            # Most-recent realized row (panel is already sliced < t).
            row = panel.iloc[-1]
            rate = row.get("funding_rate")
            if pd.isna(rate):
                continue
            cadence_h = self._cadence("bitget", sym, snap)
            apr = float(rate) * (_HOURS_PER_YEAR / cadence_h)
            if abs(apr) < self.abs_threshold_apr:
                continue

            # Build venue_signal (Bitget + each hedge venue's latest realized).
            venue_signal: dict[str, float] = {"bitget": apr}
            for hv in self.hedge_venues:
                hv_sym = self._symbol(hv, coin, snap)
                hv_panel = snap.funding_panels.get((hv, hv_sym))
                if hv_panel is None or hv_panel.empty:
                    continue
                hv_row = hv_panel.iloc[-1]
                hv_rate = hv_row.get("funding_rate")
                if pd.isna(hv_rate):
                    continue
                hv_cad = self._cadence(hv, hv_sym, snap)
                venue_signal[hv] = float(hv_rate) * (_HOURS_PER_YEAR / hv_cad)

            # Expected carry: receiver_apr − payer_apr.
            # Bitget is the receiver (long if apr<0, short if apr>0).
            # We approximate with abs(apr) − cheapest_hedge_apr_same_sign. Net rough estimate
            # so the capital pool can rank events; the real pnl uses realized rates.
            hedge_apr = self._best_hedge_apr(apr, venue_signal)
            expected_carry_apr = abs(apr) - abs(hedge_apr) if hedge_apr is not None else abs(apr)
            confidence = min(1.0, abs(apr) / (3.0 * self.abs_threshold_apr))

            events.append(FundingEvent(
                coin=coin,
                timestamp_utc=snap.t,
                trigger_type="bitget_extreme",
                venue_signal=venue_signal,
                expected_funding_pnl_apr=expected_carry_apr,
                confidence=confidence,
                metadata={
                    "bitget_symbol": sym,
                    "bitget_cadence_h": cadence_h,
                    "last_realized_rate": float(rate),
                    "abs_threshold_apr": self.abs_threshold_apr,
                },
            ))

        # Set next_check to the earliest upcoming settle across our Bitget universe.
        next_check = self._earliest_next_settle(snap)
        if next_check is not None:
            self._next_check_cache = next_check
        return events

    # ---- helpers ------------------------------------------------------ #

    def _earliest_next_settle(self, snap: MarketSnapshot) -> Optional[pd.Timestamp]:
        candidates: list[pd.Timestamp] = []
        for coin in self.coins:
            sym = self._symbol("bitget", coin, snap)
            panel = snap.funding_panels.get(("bitget", sym))
            if panel is None or panel.empty:
                continue
            cadence_h = self._cadence("bitget", sym, snap)
            last_ts = panel.iloc[-1]["timestamp_utc"]
            candidates.append(last_ts + pd.Timedelta(hours=cadence_h))
        if not candidates:
            return None
        return min(candidates)

    def _best_hedge_apr(self, recv_apr: float, signal: dict[str, float]) -> Optional[float]:
        """Pick the hedge venue whose apr is closest in magnitude but opposite sign — minimizes pnl drag on the hedge leg."""
        # Strictly: when bitget apr < 0 (we long bitget), we short hedge. Hedge short
        # *receives* funding when hedge apr > 0 → great. So we want max hedge apr.
        # When bitget apr > 0 (we short bitget), hedge long pays funding when hedge
        # apr > 0 → bad. We want min hedge apr.
        hv_signals = {k: v for k, v in signal.items() if k != "bitget"}
        if not hv_signals:
            return None
        return max(hv_signals.values()) if recv_apr < 0 else min(hv_signals.values())

    def _cadence(self, venue: str, symbol: str, snap: MarketSnapshot) -> int:
        cm = snap.contracts_meta
        if cm is not None and not cm.empty:
            row = cm[(cm["venue"] == venue) & (cm["symbol"] == symbol)]
            if not row.empty and pd.notna(row.iloc[0].get("fund_interval_hours")):
                return int(row.iloc[0]["fund_interval_hours"])
        if venue == "hyperliquid":
            return 1
        return 8

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
