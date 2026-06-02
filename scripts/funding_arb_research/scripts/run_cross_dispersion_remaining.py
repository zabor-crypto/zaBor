"""Run thresholds [0.50, 1.00, 2.00] sequentially in one process.

The parallel-process approach (subprocess per threshold) had pathological
disk + memory contention; serial in-process is materially faster because
the 131-panel archive is loaded exactly once.
"""
from __future__ import annotations

import sys
import time
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
    out_dir = project_root() / "outputs" / "runs" / "20260427_130903_cross_dispersion"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = project_root() / "data" / "normalized" / "funding_history"
    cadence_csv = project_root() / "data" / "static" / "funding_cadence.csv"

    print(f"[setup] loading panels", flush=True)
    panels = load_archive(archive_dir)
    contracts = build_contracts_meta(panels, cadence_csv)
    coins_by_venue: dict[str, set[str]] = {}
    for (v, sym), df in panels.items():
        if df.empty: continue
        b = df.iloc[0].get("base_asset")
        if b: coins_by_venue.setdefault(v, set()).add(b)
    coins_multi = sorted({c for cs in coins_by_venue.values() for c in cs
                          if sum(c in s for s in coins_by_venue.values()) >= 2})
    start = min(df["timestamp_utc"].min() for df in panels.values())
    end = max(df["timestamp_utc"].max() for df in panels.values())
    print(f"[setup] panels={len(panels)}, coins_multi={len(coins_multi)}, "
          f"timespan={start}->{end}", flush=True)

    cfg = EngineConfig(
        start=start, end=end, capital_usd=1_000_000,
        per_event_max_pct=0.10, exit_threshold_apr=0.10,
        max_hold_hours=72, exit_check_interval_minutes=60, role="taker",
    )
    caps = VenueCapabilities()
    router = HedgeRouter(caps=caps,
                         allowed_hedge_venues=["bitget", "binance", "bybit", "okx", "hyperliquid"])
    costs = CostModel.from_yaml()

    for thr in [0.50, 1.00, 2.00]:
        print(f"[thr={thr:.2f}] starting", flush=True)
        t0 = time.monotonic()
        trig = CrossDexDispersionTrigger(coins=coins_multi, min_carry_apr=thr)
        engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                             funding_panels=panels, contracts_meta=contracts)
        res = engine.run()
        df = res.to_dataframe()
        out = out_dir / f"trades_thr_{int(thr*100)}.parquet"
        df.to_parquet(out, index=False)
        elapsed = time.monotonic() - t0
        if not df.empty:
            pnl = df["realized_pnl_usd"].sum()
            days = max(1.0, (df["closed"].max() - df["opened"].min()).total_seconds()/86400)
            print(f"[thr={thr:.2f}] DONE in {elapsed:.0f}s  trades={len(df)}  "
                  f"netPnL=${pnl:,.0f}  netAPR={(pnl/1e6)*(365/days):.2%}  "
                  f"WR={(df.realized_pnl_usd>0).mean():.1%}  hold={df.hold_hours.mean():.1f}h",
                  flush=True)
        else:
            print(f"[thr={thr:.2f}] DONE in {elapsed:.0f}s  empty", flush=True)


if __name__ == "__main__":
    main()
