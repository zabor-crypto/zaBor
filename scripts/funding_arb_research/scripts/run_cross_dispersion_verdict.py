"""Step 5 verdict: cross-DEX dispersion strategy.

Same archive as the Step 4 Bitget-extreme run; different trigger. Runs
across a ``min_carry_apr`` grid and writes a verdict.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.event_engine import EngineConfig, EventEngine
from src.backtest.recorder_slippage import RecorderSlippage
from src.events.triggers.cross_dex_dispersion import CrossDexDispersionTrigger
from src.routing.hedge_router import HedgeRouter
from src.routing.venue_capabilities import VenueCapabilities
from src.utils.io import dump_yaml, project_root
from scripts.run_bitget_extreme_verdict import (
    build_contracts_meta, load_archive, summarize, verdict_for_config,
)


def main() -> None:
    archive_dir = project_root() / "data" / "normalized" / "funding_history"
    cadence_csv = project_root() / "data" / "static" / "funding_cadence.csv"

    panels = load_archive(archive_dir)
    if not panels:
        raise SystemExit("no funding panels loaded")
    contracts = build_contracts_meta(panels, cadence_csv)

    # Universe = coins listed on at least 2 venues (otherwise no spread).
    coins_by_venue: dict[str, set[str]] = {}
    for (v, sym), df in panels.items():
        if df.empty:
            continue
        b = df.iloc[0].get("base_asset")
        if b:
            coins_by_venue.setdefault(v, set()).add(b)
    coins_multi = sorted({c for cs in coins_by_venue.values() for c in cs
                          if sum(c in s for s in coins_by_venue.values()) >= 2})
    print(f"[verdict] panels: {len(panels)}, multi-venue coins: {len(coins_multi)}")

    # Timespan: union of all panels.
    start = min(df["timestamp_utc"].min() for df in panels.values())
    end = max(df["timestamp_utc"].max() for df in panels.values())
    print(f"[verdict] timespan: {start} -> {end}")

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
        datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_cross_dispersion"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"coin": coins_multi}).to_csv(out_dir / "universe.csv", index=False)

    grid = [0.20, 0.30, 0.50, 1.00, 2.00]
    summaries = []
    for thr in grid:
        trig = CrossDexDispersionTrigger(coins=coins_multi, min_carry_apr=thr)
        engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                             funding_panels=panels, contracts_meta=contracts,
                             recorder_slippage=rec_slip)
        res = engine.run()
        df = res.to_dataframe()
        df.to_parquet(out_dir / f"trades_thr_{int(thr*100)}.parquet", index=False)
        summary = summarize(df, capital_usd)
        summary["min_carry_apr"] = thr
        summary["skipped_events"] = res.skipped_events
        summary["verdict"] = verdict_for_config(summary)
        # Pair distribution
        if not df.empty:
            pair_counts = (
                df.groupby(["long_venue", "short_venue"]).size()
                  .sort_values(ascending=False).head(5).to_dict()
            )
            summary["top_pairs"] = ", ".join(
                f"{lv}-{sv}({n})" for (lv, sv), n in pair_counts.items()
            )
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
        print(f"[thr={thr:.2f}] trades={summary['n_trades']:>4d}  "
              f"netAPR={summary['net_apr']:>7.2%}  "
              f"WR={summary['win_rate']:>5.1%}  "
              f"holdH={summary['avg_hold_h']:>5.1f}  "
              f"DD={summary['max_dd']:>6.1%}  "
              f"skipped={summary['skipped_events']:>4d}  "
              f"verdict={summary['verdict']}  "
              f"top_pairs=[{summary['top_pairs']}]{slip_info}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "threshold_grid_summary.csv", index=False)

    config_snapshot = {
        "start": str(start), "end": str(end),
        "capital_usd": capital_usd,
        "per_event_max_pct": 0.10,
        "exit_threshold_apr": 0.10,
        "max_hold_hours": 72, "role": "taker",
        "venues": ["bitget", "binance", "bybit", "okx", "hyperliquid"],
        "carry_grid_apr": grid,
        "n_coins": len(coins_multi),
        "n_panels": len(panels),
    }
    dump_yaml(config_snapshot, out_dir / "config_snapshot.yaml")

    lines = [
        "# Cross-DEX Dispersion — Step 5 Tier-1 Verdict\n",
        f"- timespan: {start} → {end}",
        f"- universe: {len(coins_multi)} coins listed on ≥ 2 venues "
        f"(bitget/binance/bybit/okx/hyperliquid)",
        f"- capital: ${capital_usd:,}",
        f"- engine: event-driven, exit on carry < 10% APR or hold > 72h, taker fees",
        "\n## Threshold grid\n",
        "| min_carry_apr | trades | netAPR | win_rate | hold_h | maxDD | verdict | top venue pairs |",
        "|--------------:|-------:|-------:|---------:|-------:|------:|:--------|:----------------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['min_carry_apr']:.2f} | {s['n_trades']} | "
            f"{s['net_apr']:.2%} | {s['win_rate']:.1%} | "
            f"{s['avg_hold_h']:.1f} | {s['max_dd']:.1%} | {s['verdict']} | {s['top_pairs']} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"\n[verdict] wrote {out_dir}")


if __name__ == "__main__":
    main()
