"""MVP-2: Cross-CEX funding-persistence baseline (Binance / Bybit / OKX).

The strategy:

  * For each pair of venues (a, b) and each symbol in the universe, compute
    the funding spread S = funding_a - funding_b sampled at the funding
    interval (8h).
  * Estimate rolling mean, std, AR(1), and z-score over a window
    (default 168 hours = 21 settlements).
  * Enter when |zscore(S)| > z_entry AND AR(1) > ar_min AND projected net
    APR after fees > min_net_apr.
  * Exit when |zscore| < z_exit, or AR(1) drops below threshold, or
    realized vol doubles, or fee-adjusted expected return turns negative.

This is a baseline — the engine spec instructs us to mark the strategy
``REJECTED`` if it can't survive realistic costs. The decision happens in
``cross_cex_evaluate`` after the backtest completes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.engine import CloseAll, OpenLeg, StrategyState
from ..utils.time import annualize_funding


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def _ar1_coeff(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 8:
        return 0.0
    x = s.iloc[:-1].to_numpy()
    y = s.iloc[1:].to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _half_life(ar1: float) -> float:
    if ar1 <= 0 or ar1 >= 1:
        return float("inf")
    return float(np.log(0.5) / np.log(ar1))


@dataclass
class SpreadStats:
    mean: float
    std: float
    z: float
    ar1: float
    half_life_periods: float
    realized_vol: float
    baseline_vol: float


def compute_spread_stats(spread: pd.Series, *, baseline_window: int = 21) -> SpreadStats:
    s = spread.dropna()
    if s.empty:
        return SpreadStats(0.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    last = float(s.iloc[-1])
    z = (last - mean) / std if std > 0 else 0.0
    ar1 = _ar1_coeff(s)
    hl = _half_life(ar1)
    realized_vol = float(s.tail(8).std(ddof=1)) if len(s) >= 8 else std
    baseline_vol = float(s.tail(baseline_window).std(ddof=1)) if len(s) >= baseline_window else std
    return SpreadStats(mean, std, z, ar1, hl, realized_vol, baseline_vol)


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #

@dataclass
class CrossCEXPositionState:
    venue_long: str
    venue_short: str
    symbol: str
    entry_z: float


class CrossCEXPersistenceStrategy:
    """Trades the highest-|z| qualifying pair at each tick."""

    SUPPORTED_VENUES = ("binance", "bybit", "okx")

    def __init__(self, funding_interval_hours: int = 8):
        self.funding_interval_hours = funding_interval_hours
        self._open: Optional[CrossCEXPositionState] = None

    @staticmethod
    def _venue_symbol(venue: str, base: str) -> str:
        if venue == "okx":
            return f"{base}-USDT-SWAP"
        return f"{base}USDT"

    def _spread_series(
        self,
        funding: dict[tuple[str, str], pd.DataFrame],
        venue_a: str, venue_b: str, base: str,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        pa = funding.get((venue_a, self._venue_symbol(venue_a, base)))
        pb = funding.get((venue_b, self._venue_symbol(venue_b, base)))
        if pa is None or pb is None or pa.empty or pb.empty:
            return pd.Series(dtype=float)
        a = pa[pa["timestamp_utc"] < as_of].set_index("timestamp_utc")["funding_rate"]
        b = pb[pb["timestamp_utc"] < as_of].set_index("timestamp_utc")["funding_rate"]
        # Resample to a common 8h grid; forward-fill within a single interval
        a = a.resample(f"{self.funding_interval_hours}h").last().ffill(limit=1)
        b = b.resample(f"{self.funding_interval_hours}h").last().ffill(limit=1)
        joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        return (joined["a"] - joined["b"]).rename("spread")

    def __call__(self, state: StrategyState):
        cfg = state.config
        venues: list[str] = list(cfg.get("venues", self.SUPPORTED_VENUES))
        universe: list[str] = list(cfg.get("universe", ["BTC", "ETH", "SOL"]))
        z_entry = float(cfg.get("z_entry", 2.0))
        z_exit = float(cfg.get("z_exit", 0.5))
        ar_min = float(cfg.get("ar_min", 0.90))
        min_net_apr = float(cfg.get("min_net_apr", 0.08))
        per_trade_pct = float(cfg.get("per_trade_capital_pct", 0.10))
        rolling_hours = int(cfg.get("rolling_window_hours", 168))
        vol_double = float(cfg.get("vol_doubling_factor", 2.0))
        baseline_window = max(rolling_hours // self.funding_interval_hours, 8)

        # Exit logic if a trade is open
        if state.open_legs and self._open is not None:
            return self._maybe_exit(
                state, z_exit=z_exit, ar_min=ar_min,
                vol_double=vol_double, min_net_apr=min_net_apr,
                baseline_window=baseline_window,
            )

        # Entry logic: scan all venue pairs / symbols
        best = None  # (abs_z, venue_long, venue_short, base, stats, expected_net_apr)
        for base in universe:
            for i, va in enumerate(venues):
                for vb in venues[i + 1:]:
                    spread = self._spread_series(state.funding, va, vb, base, state.now)
                    if len(spread) < baseline_window:
                        continue
                    spread = spread.tail(baseline_window)
                    stats = compute_spread_stats(spread, baseline_window=baseline_window)
                    if abs(stats.z) < z_entry or stats.ar1 < ar_min:
                        continue

                    last_spread = float(spread.iloc[-1])
                    apr_edge = annualize_funding(abs(last_spread), self.funding_interval_hours)
                    expected_net_apr = apr_edge - 0.04   # crude allowance for round-trip cost
                    if expected_net_apr < min_net_apr:
                        continue
                    rec = (abs(stats.z), va, vb, base, stats, expected_net_apr, last_spread)
                    if best is None or rec[0] > best[0]:
                        best = rec

        if best is None:
            return None

        _, va, vb, base, stats, _net_apr, last_spread = best
        # If venue_a funding > venue_b funding -> short venue_a, long venue_b
        if last_spread > 0:
            venue_short, venue_long = va, vb
        else:
            venue_short, venue_long = vb, va

        per_leg = state.equity_usd * per_trade_pct
        legs = [
            OpenLeg(
                venue=venue_long, direction=+1,
                symbol=self._venue_symbol(venue_long, base),
                notional_usd=per_leg, role="taker",
                tag=f"z={stats.z:.2f}|ar1={stats.ar1:.2f}",
            ),
            OpenLeg(
                venue=venue_short, direction=-1,
                symbol=self._venue_symbol(venue_short, base),
                notional_usd=per_leg, role="taker",
                tag=f"z={stats.z:.2f}|ar1={stats.ar1:.2f}",
            ),
        ]
        self._open = CrossCEXPositionState(
            venue_long=venue_long, venue_short=venue_short,
            symbol=base, entry_z=stats.z,
        )
        return legs

    def _maybe_exit(self, state, *, z_exit, ar_min, vol_double, min_net_apr, baseline_window):
        op = self._open
        assert op is not None
        spread = self._spread_series(
            state.funding, op.venue_long, op.venue_short, op.symbol, state.now
        )
        if len(spread) < 8:
            return None
        spread = spread.tail(baseline_window)
        stats = compute_spread_stats(spread, baseline_window=baseline_window)

        if abs(stats.z) < z_exit:
            self._open = None
            return CloseAll(reason="z_below_exit")
        if stats.ar1 < ar_min:
            self._open = None
            return CloseAll(reason="ar1_decay")
        if stats.realized_vol > vol_double * stats.baseline_vol and stats.baseline_vol > 0:
            self._open = None
            return CloseAll(reason="vol_double")
        last_spread = float(spread.iloc[-1])
        apr_edge = annualize_funding(abs(last_spread), self.funding_interval_hours)
        if apr_edge - 0.04 < 0:
            self._open = None
            return CloseAll(reason="fee_adjusted_negative")
        return None


def cross_cex_evaluate(metrics: dict, *, min_net_apr: float = 0.08) -> str:
    """Decide whether the baseline is accepted or rejected."""
    if metrics.get("net_apr", 0.0) < min_net_apr:
        return "REJECTED"
    if metrics.get("max_drawdown", -1.0) < -0.20:
        return "REJECTED"
    if metrics.get("n_entries", 0) < 5:
        return "NEEDS MORE DATA"
    return "ACCEPTED"
