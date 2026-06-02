"""Run cross-dispersion verdict for a SINGLE threshold and write the trade
parquet into an existing verdict directory. Used to complete partial runs
without re-doing already-finished thresholds.

Usage:
    python -m scripts.run_cross_dispersion_thr --thr 0.50 --out-dir outputs/runs/20260427_130903_cross_dispersion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.event_engine import EngineConfig, EventEngine
from src.events.triggers.cross_dex_dispersion import CrossDexDispersionTrigger
from src.routing.hedge_router import HedgeRouter
from src.routing.venue_capabilities import VenueCapabilities
from src.utils.io import project_root
from scripts.run_bitget_extreme_verdict import build_contracts_meta, load_archive


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--thr", type=float, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = project_root() / "data" / "normalized" / "funding_history"
    cadence_csv = project_root() / "data" / "static" / "funding_cadence.csv"
    panels = load_archive(archive_dir)
    contracts = build_contracts_meta(panels, cadence_csv)
    coins_by_venue: dict[str, set[str]] = {}
    for (v, sym), df in panels.items():
        if df.empty:
            continue
        b = df.iloc[0].get("base_asset")
        if b:
            coins_by_venue.setdefault(v, set()).add(b)
    coins_multi = sorted({c for cs in coins_by_venue.values() for c in cs
                          if sum(c in s for s in coins_by_venue.values()) >= 2})
    start = min(df["timestamp_utc"].min() for df in panels.values())
    end = max(df["timestamp_utc"].max() for df in panels.values())
    print(f"[thr={args.thr:.2f}] coins={len(coins_multi)} timespan={start}->{end}")

    cfg = EngineConfig(
        start=start, end=end, capital_usd=1_000_000,
        per_event_max_pct=0.10, exit_threshold_apr=0.10,
        max_hold_hours=72, exit_check_interval_minutes=60, role="taker",
    )
    caps = VenueCapabilities()
    router = HedgeRouter(caps=caps,
                         allowed_hedge_venues=["bitget", "binance", "bybit", "okx", "hyperliquid"])
    costs = CostModel.from_yaml()
    trig = CrossDexDispersionTrigger(coins=coins_multi, min_carry_apr=args.thr)
    engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                         funding_panels=panels, contracts_meta=contracts)
    res = engine.run()
    df = res.to_dataframe()
    out = args.out_dir / f"trades_thr_{int(args.thr*100)}.parquet"
    df.to_parquet(out, index=False)
    if not df.empty:
        pnl = df["realized_pnl_usd"].sum()
        days = max(1.0, (df["closed"].max() - df["opened"].min()).total_seconds()/86400)
        print(f"[thr={args.thr:.2f}] wrote {out}  trades={len(df)}  "
              f"netPnL=${pnl:,.0f}  netAPR={(pnl/1e6)*(365/days):.2%}  "
              f"WR={(df.realized_pnl_usd>0).mean():.1%}  hold={df.hold_hours.mean():.1f}h  "
              f"skipped={res.skipped_events}")
    else:
        print(f"[thr={args.thr:.2f}] empty result")


if __name__ == "__main__":
    main()
