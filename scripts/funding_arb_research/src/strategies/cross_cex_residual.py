"""MVP-2 fallback: residual-funding model.

If the raw cross-CEX persistence baseline (``cross_cex_persistence``) ends
up REJECTED, this module is the next thing to try — *not* a parameter
sweep of the baseline.

The idea:

  * At each settlement timestamp, compute the cross-sectional mean funding
    across all venues for a given coin:

        mean_t  = mean_v(funding_{v,t})
        resid_t,v = funding_{v,t} - mean_t

  * Long the venue with the most negative residual; short the venue with
    the most positive residual. Total exposure stays delta-neutral on the
    underlying. The residual is the *idiosyncratic* funding component
    after stripping the common factor — it should mean-revert faster and
    be less contaminated by directional risk-on/risk-off swings.

  * Entry uses z-scored residual gap; exit on convergence, AR(1) decay,
    or fee-adjusted edge turning negative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.engine import CloseAll, OpenLeg, StrategyState
from ..utils.time import annualize_funding
from .cross_cex_persistence import compute_spread_stats


SUPPORTED_VENUES = ("binance", "bybit", "okx")


def _venue_symbol(venue: str, base: str) -> str:
    if venue == "okx":
        return f"{base}-USDT-SWAP"
    return f"{base}USDT"


def residual_panel(
    funding: dict[tuple[str, str], pd.DataFrame],
    *,
    base: str,
    venues: list[str],
    as_of: pd.Timestamp,
    interval_h: int,
    rolling_periods: int,
) -> pd.DataFrame:
    """Build a per-venue residual series for ``base``, indexed by settlement.

    Returns DataFrame with one column per venue and one row per settlement
    timestamp; each cell is the residual ``funding_v - mean_v(funding_{*,t})``
    after the common-factor mean is removed.
    """
    series: dict[str, pd.Series] = {}
    for v in venues:
        sym = _venue_symbol(v, base)
        panel = funding.get((v, sym))
        if panel is None or panel.empty:
            continue
        s = panel[panel["timestamp_utc"] < as_of].set_index("timestamp_utc")["funding_rate"]
        s = s.resample(f"{interval_h}h").last().ffill(limit=1)
        series[v] = s
    if len(series) < 2:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).dropna()
    if df.empty:
        return df
    df = df.tail(rolling_periods)
    common = df.mean(axis=1)
    return df.sub(common, axis=0)


@dataclass
class ResidualPositionState:
    base: str
    venue_long: str    # most-negative residual at entry
    venue_short: str   # most-positive residual at entry
    entry_gap: float


class CrossCEXResidualStrategy:
    """Residual-funding strategy across (binance, bybit, okx)."""

    def __init__(self, funding_interval_hours: int = 8):
        self.funding_interval_hours = funding_interval_hours
        self._open: Optional[ResidualPositionState] = None

    def __call__(self, state: StrategyState):
        cfg = state.config
        venues: list[str] = list(cfg.get("venues", SUPPORTED_VENUES))
        universe: list[str] = list(cfg.get("universe", ["BTC", "ETH", "SOL"]))
        z_entry = float(cfg.get("z_entry", 2.0))
        z_exit = float(cfg.get("z_exit", 0.5))
        ar_min = float(cfg.get("ar_min", 0.85))
        min_net_apr = float(cfg.get("min_net_apr", 0.05))
        per_trade_pct = float(cfg.get("per_trade_capital_pct", 0.10))
        rolling_hours = int(cfg.get("rolling_window_hours", 168))
        baseline_window = max(rolling_hours // self.funding_interval_hours, 8)

        if state.open_legs and self._open is not None:
            return self._maybe_exit(
                state, z_exit=z_exit, ar_min=ar_min,
                baseline_window=baseline_window, min_net_apr=min_net_apr,
            )

        best = None
        for base in universe:
            res = residual_panel(
                state.funding, base=base, venues=venues,
                as_of=state.now, interval_h=self.funding_interval_hours,
                rolling_periods=baseline_window,
            )
            if res.empty or len(res) < max(8, baseline_window // 2):
                continue
            last = res.iloc[-1]
            v_long = str(last.idxmin())
            v_short = str(last.idxmax())
            gap = float(last.max() - last.min())
            if v_long == v_short or gap <= 0:
                continue
            gap_series = res.max(axis=1) - res.min(axis=1)
            stats = compute_spread_stats(gap_series, baseline_window=baseline_window)
            if abs(stats.z) < z_entry or stats.ar1 < ar_min:
                continue
            apr_edge = annualize_funding(gap, self.funding_interval_hours)
            expected_net = apr_edge - 0.04
            if expected_net < min_net_apr:
                continue
            rec = (abs(stats.z), base, v_long, v_short, gap, stats)
            if best is None or rec[0] > best[0]:
                best = rec

        if best is None:
            return None

        _, base, v_long, v_short, gap, stats = best
        per_leg = state.equity_usd * per_trade_pct
        legs = [
            OpenLeg(
                venue=v_long, direction=+1,
                symbol=_venue_symbol(v_long, base),
                notional_usd=per_leg, role="taker",
                tag=f"residual_z={stats.z:.2f}",
            ),
            OpenLeg(
                venue=v_short, direction=-1,
                symbol=_venue_symbol(v_short, base),
                notional_usd=per_leg, role="taker",
                tag=f"residual_z={stats.z:.2f}",
            ),
        ]
        self._open = ResidualPositionState(
            base=base, venue_long=v_long, venue_short=v_short, entry_gap=gap,
        )
        return legs

    def _maybe_exit(self, state, *, z_exit, ar_min, baseline_window, min_net_apr):
        op = self._open
        assert op is not None
        venues = list({op.venue_long, op.venue_short, "binance", "bybit", "okx"})
        res = residual_panel(
            state.funding, base=op.base, venues=venues,
            as_of=state.now, interval_h=self.funding_interval_hours,
            rolling_periods=baseline_window,
        )
        if res.empty or op.venue_long not in res.columns or op.venue_short not in res.columns:
            return None
        gap_series = res[op.venue_short] - res[op.venue_long]
        stats = compute_spread_stats(gap_series, baseline_window=baseline_window)
        if abs(stats.z) < z_exit:
            self._open = None
            return CloseAll(reason="residual_z_below_exit")
        if stats.ar1 < ar_min:
            self._open = None
            return CloseAll(reason="ar1_decay")
        last_gap = float(gap_series.iloc[-1]) if len(gap_series) else 0.0
        apr_edge = annualize_funding(last_gap, self.funding_interval_hours)
        if apr_edge - 0.04 < 0:
            self._open = None
            return CloseAll(reason="fee_adjusted_negative")
        return None
