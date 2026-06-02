"""Performance metrics for the backtester.

Inputs are an equity curve (DataFrame) and trade log (DataFrame).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365 * 24 * 3600


def _yearly_factor(index: pd.DatetimeIndex) -> float:
    """Annualization factor based on observation frequency."""
    if len(index) < 2:
        return 1.0
    deltas = np.diff(index.view("int64")) / 1e9   # seconds
    avg = float(np.mean(deltas))
    if avg <= 0:
        return 1.0
    return SECONDS_PER_YEAR / avg


def equity_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if equity.empty:
        return 0.0, None, None
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    trough_idx = dd.idxmin()
    if pd.isna(trough_idx):
        return 0.0, None, None
    peak_idx = equity.loc[:trough_idx].idxmax()
    return float(dd.min()), peak_idx, trough_idx


def worst_day(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    daily = (1.0 + returns).resample("1D").prod() - 1.0
    return float(daily.min()) if not daily.empty else 0.0


def funding_capture_ratio(realized_funding_pnl: float, gross_funding_pnl: float) -> float:
    """Fraction of available funding edge actually captured after costs."""
    if gross_funding_pnl == 0:
        return 0.0
    return float(realized_funding_pnl / gross_funding_pnl)


def fee_drag(total_fees_usd: float, gross_pnl_usd: float) -> float:
    if gross_pnl_usd == 0:
        return 0.0
    return float(total_fees_usd / gross_pnl_usd)


def summarize_run(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    starting_capital: float,
) -> dict[str, Any]:
    """Compute the full metrics dict expected by the report builder."""
    if equity.empty:
        return {
            "starting_capital": starting_capital,
            "ending_capital": starting_capital,
            "gross_apr": 0.0,
            "net_apr": 0.0,
            "fee_drag": 0.0,
            "funding_capture_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_peak": None,
            "max_drawdown_trough": None,
            "worst_day": 0.0,
            "n_entries": 0,
            "avg_holding_hours": 0.0,
            "liquidation_buffer_min": None,
            "sharpe": 0.0,
            "per_symbol": {},
        }

    eq = equity.set_index("timestamp_utc")["equity_usd"]
    rets = equity_returns(eq)
    ppy = _yearly_factor(eq.index)
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    seconds = (eq.index[-1] - eq.index[0]).total_seconds()
    years = max(seconds / SECONDS_PER_YEAR, 1e-9)
    net_apr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0

    funding_pnl = float(trades.get("funding_pnl_usd", pd.Series([0.0])).sum()) \
        if not trades.empty else 0.0
    fee_pnl = float(trades.get("fees_usd", pd.Series([0.0])).sum()) \
        if not trades.empty else 0.0
    gross_pnl = float(eq.iloc[-1] - eq.iloc[0]) + fee_pnl   # add back fees for gross
    gross_apr = (1.0 + gross_pnl / starting_capital) ** (1.0 / years) - 1.0 \
        if (1.0 + gross_pnl / starting_capital) > 0 else -1.0

    mdd, peak, trough = max_drawdown(eq)
    n_entries = int(trades["entry_time"].nunique()) if "entry_time" in trades else len(trades)

    if not trades.empty and {"entry_time", "exit_time"}.issubset(trades.columns):
        durations = (
            pd.to_datetime(trades["exit_time"], utc=True)
            - pd.to_datetime(trades["entry_time"], utc=True)
        )
        avg_hold_hours = float(durations.mean().total_seconds() / 3600.0) if len(durations) else 0.0
    else:
        avg_hold_hours = 0.0

    liq_min = float(trades["liquidation_buffer_min"].min()) \
        if "liquidation_buffer_min" in trades and not trades.empty else None

    per_symbol: dict[str, Any] = {}
    if not trades.empty and "symbol" in trades:
        for sym, grp in trades.groupby("symbol"):
            sym_pnl = float(grp.get("net_pnl_usd", pd.Series([0.0])).sum())
            per_symbol[str(sym)] = {
                "trades": int(len(grp)),
                "net_pnl_usd": sym_pnl,
                "fees_usd": float(grp.get("fees_usd", pd.Series([0.0])).sum()),
                "funding_pnl_usd": float(grp.get("funding_pnl_usd", pd.Series([0.0])).sum()),
            }

    return {
        "starting_capital": starting_capital,
        "ending_capital": float(eq.iloc[-1]),
        "gross_apr": float(gross_apr),
        "net_apr": float(net_apr),
        "fee_drag": fee_drag(fee_pnl, gross_pnl),
        "funding_capture_ratio": funding_capture_ratio(
            funding_pnl - fee_pnl, max(funding_pnl, 1e-9)
        ),
        "max_drawdown": float(mdd),
        "max_drawdown_peak": str(peak) if peak is not None else None,
        "max_drawdown_trough": str(trough) if trough is not None else None,
        "worst_day": worst_day(rets),
        "n_entries": n_entries,
        "avg_holding_hours": avg_hold_hours,
        "liquidation_buffer_min": liq_min,
        "sharpe": sharpe(rets, ppy),
        "per_symbol": per_symbol,
    }
