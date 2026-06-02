"""MVP-3: GMX v2 imbalance carry feasibility module.

This file does NOT execute trades. It produces a feasibility report.

Inputs:
  * a frame of GMX market snapshots (from ``GMXCollector.snapshot_to_frame``)
  * a panel of CEX hedge funding (typically Binance USDT-perp funding for the
    same coin)
  * a cost configuration

Outputs:
  * a DataFrame of per-market feasibility metrics
  * a ``feasibility_report`` dict containing per-market verdict and a global
    PASS / NEEDS MORE DATA / REJECT.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..utils.time import annualize_funding


# --------------------------------------------------------------------------- #
# Imbalance reconstruction
# --------------------------------------------------------------------------- #

def reconstruct_imbalance(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add ``imbalance_pct`` and ``imbalance_usd`` columns; drop bad rows."""
    if snapshots.empty:
        return snapshots
    df = snapshots.copy()
    df["long_oi"] = pd.to_numeric(df.get("long_oi"), errors="coerce")
    df["short_oi"] = pd.to_numeric(df.get("short_oi"), errors="coerce")
    df["imbalance_usd"] = df["long_oi"].fillna(0) - df["short_oi"].fillna(0)
    total_oi = df["long_oi"].fillna(0) + df["short_oi"].fillna(0)
    df["imbalance_pct"] = np.where(
        total_oi > 0, df["imbalance_usd"] / total_oi, 0.0,
    )
    return df


# --------------------------------------------------------------------------- #
# Funding/borrow estimation
# --------------------------------------------------------------------------- #

def estimate_gmx_carry_apr(
    imbalance_pct: float,
    *,
    base_funding_factor_per_second: float | None = None,
    borrowing_long_per_second: float | None = None,
    borrowing_short_per_second: float | None = None,
    fallback_carry_apr: float = 0.05,
) -> float:
    """Per-side carry an arb can capture on GMX v2.

    Convention (matches GMX v2 docs):
      * ``SAVED_FUNDING_FACTOR_PER_SECOND`` is signed. Positive means
        *longs pay shorts*; negative means shorts pay longs.
      * The arb takes the side opposite the imbalance: long-heavy book
        (imbalance > 0) ⇒ arb is short.
      * Cashflow to the arb per second per $size is ``-direction * F``
        where ``direction = -sign(imbalance)``. After algebra:

            arb_funding_apr = sign(imbalance) * F * SECONDS_YEAR

      * The arb also pays the per-side borrowing factor for whichever
        side it is on (longs pay long-side borrow, shorts pay short-side).

    Fallback (factors unknown):

        carry ~ sign(-imbalance) * fallback_carry_apr * |imbalance|

    Returned APR is signed: positive means the arb-direction earns carry.
    """
    seconds_year = 365 * 24 * 3600
    if base_funding_factor_per_second is not None and not np.isnan(base_funding_factor_per_second):
        funding_apr = (
            np.sign(imbalance_pct) * float(base_funding_factor_per_second) * seconds_year
        )
        # Borrowing paid by the side the arb actually sits on.
        # imbalance > 0 ⇒ arb is short ⇒ pays short-side borrow.
        if imbalance_pct > 0:
            arb_borrow = borrowing_short_per_second
        elif imbalance_pct < 0:
            arb_borrow = borrowing_long_per_second
        else:
            arb_borrow = None
        borrow_apr = (
            float(arb_borrow) * seconds_year
            if arb_borrow is not None and not np.isnan(arb_borrow)
            else 0.0
        )
        return float(funding_apr - borrow_apr)
    return fallback_carry_apr * np.sign(-imbalance_pct) * abs(imbalance_pct)


# --------------------------------------------------------------------------- #
# Feasibility per market
# --------------------------------------------------------------------------- #

@dataclass
class FeasibilityRow:
    market: str
    base_asset: str
    timestamp_utc: pd.Timestamp
    long_oi_usd: float
    short_oi_usd: float
    pool_value_usd: float
    imbalance_pct: float
    gmx_carry_apr: float
    cex_hedge_apr: float
    gmx_open_close_bps: float
    price_impact_bps: float
    hedge_round_trip_bps: float
    gmx_net_apr: float
    seven_day_carry_pct: float
    round_trip_cost_pct: float
    pool_liquidity_pass: bool
    cost_pass: bool
    apr_pass: bool
    verdict: str


def per_market_feasibility(
    snapshot_row: pd.Series,
    *,
    cex_hedge_funding_apr: float,
    cost_model: CostModel,
    min_pool_liquidity_usd: float,
    net_apr_threshold: float,
    max_round_trip_pct_of_carry: float,
    carry_horizon_days: int = 7,
    price_impact_bps_default: float = 8.0,
) -> FeasibilityRow:
    long_oi = float(snapshot_row.get("long_oi") or 0.0)
    short_oi = float(snapshot_row.get("short_oi") or 0.0)
    pool_value = float(snapshot_row.get("pool_value_usd") or 0.0)
    imbalance_pct = float(snapshot_row.get("imbalance_pct") or 0.0)

    def _f(name: str) -> float | None:
        v = snapshot_row.get(name)
        try:
            f = float(v)
            return f if not np.isnan(f) else None
        except (TypeError, ValueError):
            return None

    funding_factor = _f("funding_factor_per_second")
    borrow_long = _f("borrowing_factor_long_per_second")
    borrow_short = _f("borrowing_factor_short_per_second")

    gmx_carry_apr = estimate_gmx_carry_apr(
        imbalance_pct,
        base_funding_factor_per_second=funding_factor,
        borrowing_long_per_second=borrow_long,
        borrowing_short_per_second=borrow_short,
    )
    gmx_oc_bps = cost_model.fee_bps("gmx")
    hedge_rt_bps = cost_model.round_trip_fee_bps("binance")  # default hedge venue
    price_impact_bps = float(price_impact_bps_default)
    total_round_trip_bps = 2 * gmx_oc_bps + hedge_rt_bps + price_impact_bps
    round_trip_cost_pct = total_round_trip_bps * 1e-4

    gmx_net_apr = (
        gmx_carry_apr
        - cex_hedge_funding_apr
        - round_trip_cost_pct        # one-time cost amortized below as well
    )

    seven_day_carry_pct = (gmx_carry_apr - cex_hedge_funding_apr) * (carry_horizon_days / 365.0)

    pool_liq_pass = pool_value >= min_pool_liquidity_usd
    cost_pass = (
        seven_day_carry_pct > 0
        and round_trip_cost_pct < max_round_trip_pct_of_carry * abs(seven_day_carry_pct)
    )
    apr_pass = gmx_net_apr > net_apr_threshold

    if pool_liq_pass and cost_pass and apr_pass:
        verdict = "CANDIDATE"
    elif not pool_liq_pass:
        verdict = "INSUFFICIENT_LIQUIDITY"
    elif not cost_pass:
        verdict = "COSTS_TOO_HIGH"
    else:
        verdict = "BELOW_APR_THRESHOLD"

    return FeasibilityRow(
        market=str(snapshot_row.get("symbol", snapshot_row.get("market", ""))),
        base_asset=str(snapshot_row.get("base_asset", "")),
        timestamp_utc=pd.Timestamp(snapshot_row.get("timestamp_utc")),
        long_oi_usd=long_oi, short_oi_usd=short_oi, pool_value_usd=pool_value,
        imbalance_pct=imbalance_pct,
        gmx_carry_apr=gmx_carry_apr,
        cex_hedge_apr=cex_hedge_funding_apr,
        gmx_open_close_bps=gmx_oc_bps,
        price_impact_bps=price_impact_bps,
        hedge_round_trip_bps=hedge_rt_bps,
        gmx_net_apr=gmx_net_apr,
        seven_day_carry_pct=seven_day_carry_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        pool_liquidity_pass=pool_liq_pass,
        cost_pass=cost_pass,
        apr_pass=apr_pass,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Top-level feasibility runner
# --------------------------------------------------------------------------- #

def imbalance_distribution(history: pd.DataFrame) -> dict[str, Any]:
    """Summarize a multi-week hourly imbalance series for a population view.

    ``history`` must include ``imbalance_pct`` and ``timestamp_utc``.
    """
    if history is None or history.empty or "imbalance_pct" not in history.columns:
        return {"hours_of_data": 0}
    s = pd.to_numeric(history["imbalance_pct"], errors="coerce").dropna()
    if s.empty:
        return {"hours_of_data": 0}
    return {
        "hours_of_data": int(len(s)),
        "imbalance_p05": float(s.quantile(0.05)),
        "imbalance_p50": float(s.median()),
        "imbalance_p95": float(s.quantile(0.95)),
        "imbalance_abs_mean": float(s.abs().mean()),
        "fraction_above_30pct": float((s.abs() > 0.30).mean()),
    }


def run_feasibility(
    gmx_snapshots: pd.DataFrame,
    cex_funding_panels: dict[str, pd.DataFrame],
    *,
    cost_model: CostModel,
    config: dict[str, Any],
    history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the feasibility table and global verdict.

    ``cex_funding_panels``: maps base asset (e.g. ``"BTC"``) to a normalized
    Binance funding history frame.
    """
    if gmx_snapshots.empty:
        return pd.DataFrame(), {
            "verdict": "NEEDS MORE DATA",
            "reason": "no GMX snapshots",
            "data_completeness_score": 0.0,
            "candidates": 0,
        }

    df = reconstruct_imbalance(gmx_snapshots)
    rows: list[FeasibilityRow] = []
    universe = config.get("universe", []) or []

    for _, snap in df.iterrows():
        base = str(snap.get("base_asset") or "")
        if universe and base not in universe:
            continue
        cex_panel = cex_funding_panels.get(base)
        if cex_panel is None or cex_panel.empty:
            cex_apr = 0.0
        else:
            recent = cex_panel.tail(21)["funding_rate"].mean()
            interval_h = int(cex_panel["funding_interval_hours"].iloc[-1] or 8)
            cex_apr = annualize_funding(float(recent or 0.0), interval_h)

        row = per_market_feasibility(
            snap,
            cex_hedge_funding_apr=cex_apr,
            cost_model=cost_model,
            min_pool_liquidity_usd=float(config.get("min_pool_liquidity_usd", 5_000_000)),
            net_apr_threshold=float(config.get("net_apr_threshold", 0.10)),
            max_round_trip_pct_of_carry=float(config.get("max_round_trip_pct_of_carry", 0.20)),
            carry_horizon_days=int(config.get("carry_horizon_days", 7)),
            price_impact_bps_default=float(config.get("price_impact_bps_default", 8.0)),
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(), {
            "verdict": "NEEDS MORE DATA",
            "reason": "no rows after filtering",
            "data_completeness_score": 0.0,
            "candidates": 0,
        }

    out_df = pd.DataFrame([asdict(r) for r in rows])
    candidates = int((out_df["verdict"] == "CANDIDATE").sum())
    completeness = float(
        out_df[["long_oi_usd", "short_oi_usd", "pool_value_usd"]].notna().all(axis=1).mean()
    )
    if completeness < 0.5:
        global_verdict = "NEEDS MORE DATA"
    elif candidates >= 1:
        global_verdict = "PASS"
    else:
        global_verdict = "REJECT"

    distribution = imbalance_distribution(history) if history is not None else {}
    return out_df, {
        "verdict": global_verdict,
        "candidates": candidates,
        "rows": int(len(out_df)),
        "data_completeness_score": completeness,
        "imbalance_distribution": distribution,
    }
