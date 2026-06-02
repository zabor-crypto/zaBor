"""Step 6 verdict: new-listing spike strategy.

Universe = the union of base_assets that have a launch_ts_utc within
the lookback window of the funding archive AND have funding panels on
at least 2 venues. Threshold grid is for ``abs_threshold_apr``;
``max_age_days`` is fixed (the listing-age is bounded by the trigger
itself; the engine just iterates time).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.event_engine import EngineConfig, EventEngine
from src.backtest.recorder_slippage import RecorderSlippage
from src.events.triggers.new_listing import NewListingTrigger
from src.routing.hedge_router import HedgeRouter
from src.routing.venue_capabilities import VenueCapabilities
from src.utils.io import dump_yaml, project_root
from scripts.run_bitget_extreme_verdict import (
    build_contracts_meta, load_archive, summarize, verdict_for_config,
)


def main() -> None:
    archive_dir = project_root() / "data" / "normalized" / "funding_history"
    cadence_csv = project_root() / "data" / "static" / "funding_cadence.csv"
    listing_csv = project_root() / "data" / "static" / "listing_archive.csv"

    panels = load_archive(archive_dir)
    if not panels:
        raise SystemExit("no funding panels loaded")
    contracts = build_contracts_meta(panels, cadence_csv)
    listings = pd.read_csv(listing_csv)
    listings["launch_ts_utc"] = pd.to_datetime(listings["launch_ts_utc"], utc=True, errors="coerce")

    # Universe: coins with a known launch ts AND funding data on ≥ 2 venues
    coins_with_data: dict[str, set[str]] = {}
    for (v, sym), df in panels.items():
        if df.empty: continue
        b = df.iloc[0].get("base_asset")
        if b: coins_with_data.setdefault(str(b), set()).add(v)
    multi_venue = {c for c, vs in coins_with_data.items() if len(vs) >= 2}
    coins = sorted(set(listings.dropna(subset=["launch_ts_utc"])["base_asset"]) & multi_venue)
    print(f"[verdict] panels={len(panels)}  coins (multi-venue + dated)={len(coins)}", flush=True)

    start = min(df["timestamp_utc"].min() for df in panels.values())
    end = max(df["timestamp_utc"].max() for df in panels.values())
    print(f"[verdict] timespan: {start} -> {end}", flush=True)

    capital_usd = 1_000_000
    cfg = EngineConfig(
        start=start, end=end, capital_usd=capital_usd,
        per_event_max_pct=0.10, exit_threshold_apr=0.10,
        max_hold_hours=72, exit_check_interval_minutes=60, role="taker",
    )
    caps = VenueCapabilities()
    router = HedgeRouter(caps=caps,
                         allowed_hedge_venues=["bitget", "binance", "bybit", "okx", "hyperliquid"])
    costs = CostModel.from_yaml()
    rec_slip = RecorderSlippage.from_project_root()
    cov = rec_slip.coverage_summary()
    print(f"[verdict] recorder slippage coverage: { {v: len(cs) for v,cs in cov.items()} }")

    out_dir = project_root() / "outputs" / "runs" / (
        datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_new_listing"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"coin": coins}).to_csv(out_dir / "universe.csv", index=False)

    grid = [(0.50, 30), (1.00, 30), (2.00, 30), (1.00, 14), (1.00, 7)]
    summaries = []
    for thr, age_days in grid:
        trig = NewListingTrigger(coins=coins, listing_archive=listings,
                                  max_age_days=age_days, abs_threshold_apr=thr)
        engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                             funding_panels=panels, contracts_meta=contracts,
                             recorder_slippage=rec_slip)
        res = engine.run()
        df = res.to_dataframe()
        df.to_parquet(out_dir / f"trades_thr_{int(thr*100)}_age_{age_days}d.parquet", index=False)
        summary = summarize(df, capital_usd)
        summary["abs_threshold_apr"] = thr
        summary["max_age_days"] = age_days
        summary["skipped"] = res.skipped_events
        summary["verdict"] = verdict_for_config(summary)
        if not df.empty:
            top = df.groupby(["long_venue","short_venue"]).size().sort_values(ascending=False).head(5)
            summary["top_pairs"] = ", ".join(f"{lv}-{sv}({n})" for (lv,sv), n in top.items())
        else:
            summary["top_pairs"] = ""
        if not df.empty and "long_slippage_source" in df.columns:
            src_counts = (
                df[["long_slippage_source", "short_slippage_source"]]
                .stack().value_counts().to_dict()
            )
            summary["slip_src_recorder"] = int(src_counts.get("recorder", 0))
            summary["slip_src_fallback"] = int(src_counts.get("fallback", 0))
        summaries.append(summary)
        slip_info = (f"  slip_src=rec:{summary.get('slip_src_recorder',0)}/"
                     f"fb:{summary.get('slip_src_fallback',0)}")
        print(f"[thr={thr:.2f} age={age_days}d] trades={summary['n_trades']:>4d}  "
              f"netAPR={summary['net_apr']:>7.2%}  WR={summary['win_rate']:>5.1%}  "
              f"holdH={summary['avg_hold_h']:>5.1f}  DD={summary['max_dd']:>6.1%}  "
              f"verdict={summary['verdict']}{slip_info}", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "threshold_grid_summary.csv", index=False)

    config_snapshot = {
        "start": str(start), "end": str(end),
        "capital_usd": capital_usd,
        "per_event_max_pct": 0.10,
        "exit_threshold_apr": 0.10,
        "max_hold_hours": 72, "role": "taker",
        "venues": ["bitget", "binance", "bybit", "okx", "hyperliquid"],
        "grid": [{"abs_threshold_apr": t, "max_age_days": a} for t, a in grid],
        "n_coins": len(coins), "n_panels": len(panels),
    }
    dump_yaml(config_snapshot, out_dir / "config_snapshot.yaml")

    lines = [
        "# New-Listing Spike — Step 6 Tier-1 Verdict\n",
        f"- timespan: {start} → {end}",
        f"- universe: {len(coins)} coins with known launch_ts AND ≥ 2 venue funding panels",
        f"- capital: ${capital_usd:,}  taker fees  fallback slippage 5bp",
        "\n## Grid (threshold APR × max listing age days)\n",
        "| thr_apr | age_days | trades | netAPR | WR | hold_h | maxDD | verdict | top pairs |",
        "|--------:|---------:|-------:|-------:|---:|-------:|------:|:--------|:----------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['abs_threshold_apr']:.2f} | {s['max_age_days']} | "
            f"{s['n_trades']} | {s['net_apr']:.2%} | "
            f"{s['win_rate']:.1%} | {s['avg_hold_h']:.1f} | "
            f"{s['max_dd']:.1%} | {s['verdict']} | {s['top_pairs']} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"\n[verdict] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
