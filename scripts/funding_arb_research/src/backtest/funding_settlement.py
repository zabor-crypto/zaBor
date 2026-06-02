"""Funding settlement: pays/receives are booked at actual funding times.

A long pays the funding rate; a short receives it. PnL convention:
    funding_pnl_per_unit_notional = -direction * funding_rate
where direction is +1 for long, -1 for short.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundingPnL:
    timestamp: pd.Timestamp
    venue: str
    symbol: str
    funding_rate: float
    direction: int          # +1 long, -1 short
    notional_usd: float
    pnl_usd: float


def funding_pnl(
    funding_df: pd.DataFrame,
    *,
    direction: int,
    notional_usd: float,
    venue: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[FundingPnL]:
    """Book a funding pnl entry for every settlement timestamp in [start, end].

    ``funding_df`` must already be filtered to the relevant venue+symbol; we
    keep the venue/symbol args explicit so the output is self-describing.
    """
    if funding_df.empty:
        return []
    mask = (
        (funding_df["timestamp_utc"] > start)
        & (funding_df["timestamp_utc"] <= end)
    )
    out: list[FundingPnL] = []
    for _, r in funding_df.loc[mask].iterrows():
        rate = float(r["funding_rate"])
        pnl = -direction * rate * notional_usd
        out.append(FundingPnL(
            timestamp=r["timestamp_utc"],
            venue=venue, symbol=symbol,
            funding_rate=rate, direction=direction,
            notional_usd=notional_usd, pnl_usd=pnl,
        ))
    return out


def select_predicted_funding(
    funding_df: pd.DataFrame, *, as_of: pd.Timestamp
) -> float | None:
    """Return the most recent ``predicted_funding_rate`` strictly before ``as_of``.

    Falls back to the most recent realized ``funding_rate`` if no predicted
    value is available — strategies decide whether the fallback is acceptable.
    """
    if funding_df.empty:
        return None
    df = funding_df[funding_df["timestamp_utc"] < as_of]
    if df.empty:
        return None
    if "predicted_funding_rate" in df.columns:
        pred = df["predicted_funding_rate"].dropna()
        if not pred.empty:
            return float(pred.iloc[-1])
    realized = df["funding_rate"].dropna()
    if realized.empty:
        return None
    return float(realized.iloc[-1])
