"""Step 4 verdict: Bitget extreme-funding strategy on top-30 universe.

Loads the multi-venue funding-history archive, runs the event-engine
across a threshold grid {50%, 100%, 200%} APR, and writes per-config
results + an aggregate verdict.

Usage (from funding_arb/):

    python -m scripts.run_bitget_extreme_verdict

Inputs:
  data/normalized/funding_history/<venue>/<symbol>.parquet
  data/static/funding_cadence.csv
  data/static/listing_archive.csv

Outputs:
  outputs/runs/<ts>_bitget_extreme/
    config_snapshot.yaml
    universe.csv
    threshold_grid_summary.csv
    trades_thr_<X>.parquet
    verdict.md
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.event_engine import EngineConfig, EventEngine
from src.backtest.recorder_slippage import RecorderSlippage
from src.events.triggers.bitget_extreme import BitgetExtremeTrigger
from src.routing.hedge_router import HedgeRouter
from src.routing.venue_capabilities import VenueCapabilities
from src.utils.io import dump_yaml, project_root

_HOURS_PER_YEAR = 365.0 * 24.0
_OUT_BASE = project_root() / "outputs" / "runs"


def load_archive(base: Path) -> dict[tuple[str, str], pd.DataFrame]:
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    for venue_dir in sorted(base.iterdir()):
        if not venue_dir.is_dir():
            continue
        venue = venue_dir.name
        for f in sorted(venue_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            if df.empty:
                continue
            sym = f.stem
            panels[(venue, sym)] = df.sort_values("timestamp_utc").reset_index(drop=True)
    return panels


def build_contracts_meta(panels: dict[tuple[str, str], pd.DataFrame],
                          cadence_csv: Path) -> pd.DataFrame:
    cadence = pd.read_csv(cadence_csv) if cadence_csv.exists() else pd.DataFrame()
    rows = []
    for (venue, sym), df in panels.items():
        base = df.iloc[0].get("base_asset")
        cad = None
        if not cadence.empty:
            m = cadence[(cadence["venue"] == venue) & (cadence["symbol"] == sym)]
            if not m.empty and pd.notna(m.iloc[0].get("fund_interval_hours")):
                cad = int(m.iloc[0]["fund_interval_hours"])
        if cad is None:
            cad = 1 if venue == "hyperliquid" else 8
        rows.append({"venue": venue, "symbol": sym, "base_asset": base,
                     "fund_interval_hours": cad})
    return pd.DataFrame(rows)


def coins_with_bitget_data(panels: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    out = []
    for (v, sym), df in panels.items():
        if v != "bitget" or df.empty:
            continue
        out.append(sym.removesuffix("USDT"))
    return sorted(set(out))


def summarize(trades: pd.DataFrame, capital_usd: float) -> dict:
    if trades.empty:
        return {"n_trades": 0, "net_pnl_usd": 0.0, "net_apr": 0.0,
                "win_rate": 0.0, "avg_hold_h": 0.0, "max_dd": 0.0}
    pnl = trades["realized_pnl_usd"].sum()
    days_span = max(1.0, (trades["closed"].max() - trades["opened"].min()).total_seconds() / 86400.0)
    net_apr = (pnl / capital_usd) * (365.0 / days_span)
    eq = capital_usd + trades.sort_values("closed")["realized_pnl_usd"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return {
        "n_trades": int(len(trades)),
        "net_pnl_usd": float(pnl),
        "net_apr": float(net_apr),
        "win_rate": float((trades["realized_pnl_usd"] > 0).mean()),
        "avg_hold_h": float(trades["hold_hours"].mean()),
        "max_dd": float(dd.min() if len(dd) else 0.0),
        "days_span": float(days_span),
    }


def verdict_for_config(summary: dict) -> str:
    """Apply the standing verdict semantics."""
    if summary["n_trades"] < 30:
        return "NEEDS MORE DATA"
    if summary["net_apr"] >= 0.20 and summary["max_dd"] >= -0.15:
        return "PROMOTE TO PAPER"
    if summary["net_apr"] <= 0.05:
        return "REJECT"
    return "MARGINAL"


def main() -> None:
    archive_dir = project_root() / "data" / "normalized" / "funding_history"
    cadence_csv = project_root() / "data" / "static" / "funding_cadence.csv"

    if not archive_dir.exists():
        raise SystemExit(f"missing archive at {archive_dir} — run "
                         "src.collectors.funding_history_archive first")

    panels = load_archive(archive_dir)
    if not panels:
        raise SystemExit("no funding panels loaded")
    contracts = build_contracts_meta(panels, cadence_csv)
    coins = coins_with_bitget_data(panels)
    print(f"[verdict] panels: {len(panels)}, coins (Bitget side): {len(coins)}")

    # Use the union timespan covered by Bitget panels (the trigger-side data).
    bitget_panels = [df for (v, _), df in panels.items() if v == "bitget"]
    start = min(df["timestamp_utc"].min() for df in bitget_panels)
    end = max(df["timestamp_utc"].max() for df in bitget_panels)
    print(f"[verdict] timespan: {start} -> {end}")

    capital_usd = 1_000_000
    cfg = EngineConfig(
        start=start, end=end, capital_usd=capital_usd,
        per_event_max_pct=0.10, exit_threshold_apr=0.10,
        max_hold_hours=72, exit_check_interval_minutes=60, role="taker",
    )
    caps = VenueCapabilities()
    router = HedgeRouter(caps=caps,
                         allowed_hedge_venues=["binance", "bybit", "okx", "hyperliquid"])
    costs = CostModel.from_yaml()
    rec_slip = RecorderSlippage.from_project_root()
    cov = rec_slip.coverage_summary()
    print(f"[verdict] recorder slippage coverage: { {v: len(cs) for v,cs in cov.items()} }")

    out_dir = _OUT_BASE / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_bitget_extreme"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"coin": coins}).to_csv(out_dir / "universe.csv", index=False)

    grid = [0.50, 1.00, 2.00]
    summaries = []
    for thr in grid:
        trig = BitgetExtremeTrigger(coins=coins, abs_threshold_apr=thr,
                                     hedge_venues=["binance", "bybit", "okx", "hyperliquid"])
        engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                             funding_panels=panels, contracts_meta=contracts,
                             recorder_slippage=rec_slip)
        res = engine.run()
        df = res.to_dataframe()
        df.to_parquet(out_dir / f"trades_thr_{int(thr*100)}.parquet", index=False)
        summary = summarize(df, capital_usd)
        summary["abs_threshold_apr"] = thr
        summary["skipped_events"] = res.skipped_events
        summary["verdict"] = verdict_for_config(summary)
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
              f"verdict={summary['verdict']}{slip_info}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "threshold_grid_summary.csv", index=False)

    config_snapshot = {
        "start": str(start), "end": str(end),
        "capital_usd": capital_usd,
        "per_event_max_pct": 0.10,
        "exit_threshold_apr": 0.10,
        "max_hold_hours": 72,
        "role": "taker",
        "hedge_venues": ["binance", "bybit", "okx", "hyperliquid"],
        "threshold_grid_apr": grid,
        "n_coins": len(coins),
        "n_panels": len(panels),
    }
    dump_yaml(config_snapshot, out_dir / "config_snapshot.yaml")

    # Verdict markdown
    lines = ["# Bitget Extreme — Step 4 Tier-1 Verdict\n",
             f"- timespan: {start} → {end}",
             f"- universe: top-{len(coins)} Bitget USDT-FUTURES by 24h volume",
             f"- hedge venues: binance, bybit, okx, hyperliquid",
             f"- capital: ${capital_usd:,}",
             "\n## Threshold grid\n",
             "| thr_apr | trades | netAPR | win_rate | avg_hold_h | maxDD | verdict |",
             "|--------:|-------:|-------:|---------:|-----------:|------:|:--------|"]
    for s in summaries:
        lines.append(
            f"| {s['abs_threshold_apr']:.2f} | {s['n_trades']:d} | "
            f"{s['net_apr']:.2%} | {s['win_rate']:.1%} | "
            f"{s['avg_hold_h']:.1f} | {s['max_dd']:.1%} | {s['verdict']} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"\n[verdict] wrote {out_dir}")


if __name__ == "__main__":
    main()
