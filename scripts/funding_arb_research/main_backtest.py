"""Top-level backtest entry point.

Usage:

    python -m funding_arb.main_backtest --strategy hl_binance_interval
    python -m funding_arb.main_backtest --strategy cross_cex_persistence
    python -m funding_arb.main_backtest --strategy gmx_imbalance_feasibility
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestEngine, EngineConfig
from src.backtest.metrics import summarize_run
from src.normalization.schema import empty_funding_frame, empty_price_frame
from src.reports.report_builder import ReportInputs, write_report
from src.strategies.cross_cex_persistence import (
    CrossCEXPersistenceStrategy, cross_cex_evaluate,
)
from src.strategies.cross_cex_residual import CrossCEXResidualStrategy
from src.strategies.hl_cross_venue_disp import HLCrossVenueDispersionStrategy
from src.strategies.gmx_imbalance_feasibility import run_feasibility
from src.strategies.hl_binance_interval import HLBinanceIntervalStrategy
from src.utils.io import (
    config_dir,
    load_yaml,
    normalized_dir,
    read_parquet,
    project_root,
)
from src.utils.logging import attach_run_logfile, get_logger


LOG = get_logger("funding_arb.main_backtest")


def _safe_read(path: Path, *, kind: str = "funding") -> pd.DataFrame:
    if not path.exists():
        LOG.warning("missing %s data: %s", kind, path)
        return empty_funding_frame() if kind == "funding" else empty_price_frame()
    return read_parquet(path)


def _build_depth_panels() -> dict[tuple[str, str], pd.DataFrame]:
    """Load ``depth_snapshot.parquet`` (if present) into per-(venue, symbol) panels."""
    path = normalized_dir() / "depth_snapshot.parquet"
    if not path.exists():
        return {}
    df = read_parquet(path)
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    if df.empty:
        return panels
    for (v, s), sub in df.groupby(["venue", "symbol"]):
        sub = sub.sort_values("timestamp_utc").reset_index(drop=True)
        panels[(str(v), str(s))] = sub
    return panels


def _build_panels(
    strategy: str, universe: list[str],
) -> tuple[dict, dict]:
    """Load whatever normalized parquets exist and build panels keyed by (venue, symbol).

    Funding panels are required for every leg the strategy will trade. Price
    panels are best-effort: when missing, the engine will fall back to
    funding-rate marks where possible (mark_price is included in the
    funding schema).
    """
    nd = normalized_dir()
    funding_panels: dict[tuple[str, str], pd.DataFrame] = {}
    price_panels: dict[tuple[str, str], pd.DataFrame] = {}

    venue_files = {
        "binance": nd / "funding_binance.parquet",
        "bybit": nd / "funding_bybit.parquet",
        "okx": nd / "funding_okx.parquet",
        "hyperliquid": nd / "funding_hyperliquid.parquet",
    }
    predicted_files = {
        "binance": nd / "predicted_funding_binance.parquet",
        "hyperliquid": nd / "predicted_funding_hyperliquid.parquet",
    }

    # Load realized funding first
    by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for venue, fpath in venue_files.items():
        df = _safe_read(fpath, kind="funding")
        if df.empty:
            continue
        for sym, sub in df.groupby("symbol"):
            sub = sub.sort_values("timestamp_utc").reset_index(drop=True)
            by_key[(venue, str(sym))] = sub

    # Layer in predicted funding rows. The HL predictedFundings payload is
    # cross-venue (it surfaces predicted rates for Binance/Bybit/OKX as
    # well), so we route each row to its own (venue, symbol) panel.
    for src_venue, fpath in predicted_files.items():
        pf = _safe_read(fpath, kind="funding")
        if pf.empty:
            continue
        for (venue, sym), sub in pf.groupby(["venue", "symbol"]):
            key = (str(venue), str(sym))
            sub = sub.sort_values("timestamp_utc").reset_index(drop=True)
            existing = by_key.get(key)
            if existing is None or existing.empty:
                by_key[key] = sub
            else:
                merged = pd.concat([existing, sub], ignore_index=True)
                merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
                by_key[key] = merged

    for key, sub in by_key.items():
        funding_panels[key] = sub
        if "mark_price" in sub.columns and sub["mark_price"].notna().any():
            price_panels[key] = sub.rename(
                columns={"mark_price": "mid_price"}
            )[["timestamp_utc", "mid_price"]].copy()

    # Hyperliquid funding rows do not carry markPrice. For research-grade
    # backtests we proxy HL mid with the matching Binance USDT-perp mid for
    # the same coin — they track within ~1 bp, and we trade the *funding*
    # spread, not the price. Without this proxy, every HL leg fails to
    # open and the backtest returns 1-legged directional positions.
    for (venue, sym), fp in list(funding_panels.items()):
        if venue != "hyperliquid":
            continue
        if (venue, sym) in price_panels:
            continue
        bin_panel = price_panels.get(("binance", f"{sym}USDT"))
        if bin_panel is None or bin_panel.empty:
            continue
        price_panels[(venue, sym)] = bin_panel.copy()
        LOG.info("synthesized HL price panel for %s from binance/%sUSDT", sym, sym)

    return funding_panels, price_panels


def _data_window(funding_panels: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the overlap of *realized* funding coverage across panels.

    Predicted-funding rows (where ``funding_rate`` is null) carry future
    settlement timestamps and would otherwise force the data window into
    the future; we exclude them from the window calculation.
    """
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for df in funding_panels.values():
        if df.empty:
            continue
        if "funding_rate" in df.columns:
            realized = df[df["funding_rate"].notna()]
        else:
            realized = df
        if realized.empty:
            continue
        starts.append(realized["timestamp_utc"].min())
        ends.append(realized["timestamp_utc"].max())
    if not starts:
        now = pd.Timestamp.now(tz="UTC")
        return now - pd.Timedelta(days=30), now
    return max(starts), min(ends)


# --------------------------------------------------------------------------- #
# Strategy runners
# --------------------------------------------------------------------------- #

def run_hl_binance(args) -> Path:
    strat_cfg_all = load_yaml(config_dir() / "strategy_params.yaml")
    cfg = strat_cfg_all["hl_binance_interval"]
    cfg["per_trade_capital_pct"] = strat_cfg_all["global"]["per_trade_capital_pct"]
    universe = cfg["universe"]

    funding_panels, price_panels = _build_panels("hl_binance_interval", universe)
    start, end = _data_window(funding_panels)
    LOG.info("data window: %s -> %s", start, end)

    engine_cfg = EngineConfig(
        start=start,
        end=end,
        step=timedelta(minutes=int(cfg.get("rebalance_minutes", 60))),
        capital_usd=float(strat_cfg_all["global"]["capital_usd"]),
        liquidation_buffer_min=float(strat_cfg_all["global"]["liquidation_buffer_min"]),
        max_strategy_dd=float(strat_cfg_all["global"]["max_strategy_dd"]),
        data_staleness_seconds=int(
            cfg.get("data_staleness_seconds",
                    strat_cfg_all["global"]["data_staleness_seconds"])
        ),
    )
    cost = CostModel.from_yaml()
    engine = BacktestEngine(engine_cfg, cost, cfg)
    strategy = HLBinanceIntervalStrategy()
    result = engine.run(strategy, funding_panels, price_panels, _build_depth_panels())
    metrics = summarize_run(result.equity, result.trades,
                            starting_capital=engine_cfg.capital_usd)

    inputs = ReportInputs(
        strategy_name="hl_binance_interval",
        config_snapshot={"engine": vars(engine_cfg), "strategy": cfg},
        metrics=metrics,
        equity=result.equity,
        trades=result.trades,
    )
    run_dir = write_report(inputs)
    attach_run_logfile(LOG, run_dir / "run.log")
    LOG.info("wrote run dir: %s", run_dir)
    return run_dir


def run_hl_cross_venue_disp(args) -> Path:
    strat_cfg_all = load_yaml(config_dir() / "strategy_params.yaml")
    cfg = strat_cfg_all.get("hl_cross_venue_disp") or strat_cfg_all["hl_binance_interval"].copy()
    cfg["per_trade_capital_pct"] = strat_cfg_all["global"]["per_trade_capital_pct"]
    funding_panels, price_panels = _build_panels("hl_cross_venue_disp",
                                                  cfg.get("universe", []))
    start, end = _data_window(funding_panels)
    LOG.info("data window: %s -> %s", start, end)
    engine_cfg = EngineConfig(
        start=start, end=end,
        step=timedelta(minutes=int(cfg.get("rebalance_minutes", 60))),
        capital_usd=float(strat_cfg_all["global"]["capital_usd"]),
        liquidation_buffer_min=float(strat_cfg_all["global"]["liquidation_buffer_min"]),
        max_strategy_dd=float(strat_cfg_all["global"]["max_strategy_dd"]),
        data_staleness_seconds=int(cfg.get("data_staleness_seconds",
                                            strat_cfg_all["global"]["data_staleness_seconds"])),
    )
    cost = CostModel.from_yaml()
    engine = BacktestEngine(engine_cfg, cost, cfg)
    strategy = HLCrossVenueDispersionStrategy()
    result = engine.run(strategy, funding_panels, price_panels, _build_depth_panels())
    metrics = summarize_run(result.equity, result.trades,
                            starting_capital=engine_cfg.capital_usd)
    inputs = ReportInputs(
        strategy_name="hl_cross_venue_disp",
        config_snapshot={"engine": vars(engine_cfg), "strategy": cfg},
        metrics=metrics, equity=result.equity, trades=result.trades,
    )
    run_dir = write_report(inputs)
    attach_run_logfile(LOG, run_dir / "run.log")
    LOG.info("wrote run dir: %s", run_dir)
    return run_dir


def run_cross_cex_residual(args) -> Path:
    strat_cfg_all = load_yaml(config_dir() / "strategy_params.yaml")
    cfg = strat_cfg_all["cross_cex_persistence"].copy()
    cfg["per_trade_capital_pct"] = strat_cfg_all["global"]["per_trade_capital_pct"]
    funding_panels, price_panels = _build_panels("cross_cex_residual", cfg["universe"])
    start, end = _data_window(funding_panels)
    LOG.info("data window: %s -> %s", start, end)

    engine_cfg = EngineConfig(
        start=start, end=end, step=timedelta(hours=8),
        capital_usd=float(strat_cfg_all["global"]["capital_usd"]),
        liquidation_buffer_min=float(strat_cfg_all["global"]["liquidation_buffer_min"]),
        max_strategy_dd=float(strat_cfg_all["global"]["max_strategy_dd"]),
        data_staleness_seconds=86400,
    )
    cost = CostModel.from_yaml()
    engine = BacktestEngine(engine_cfg, cost, cfg)
    strategy = CrossCEXResidualStrategy()
    result = engine.run(strategy, funding_panels, price_panels, _build_depth_panels())
    metrics = summarize_run(result.equity, result.trades,
                            starting_capital=engine_cfg.capital_usd)
    metrics["model"] = "residual"
    inputs = ReportInputs(
        strategy_name="cross_cex_residual",
        config_snapshot={"engine": vars(engine_cfg), "strategy": cfg},
        metrics=metrics, equity=result.equity, trades=result.trades,
    )
    run_dir = write_report(inputs)
    attach_run_logfile(LOG, run_dir / "run.log")
    LOG.info("wrote run dir: %s", run_dir)
    return run_dir


def run_cross_cex(args) -> Path:
    strat_cfg_all = load_yaml(config_dir() / "strategy_params.yaml")
    cfg = strat_cfg_all["cross_cex_persistence"]
    cfg["per_trade_capital_pct"] = strat_cfg_all["global"]["per_trade_capital_pct"]
    universe = cfg["universe"]

    funding_panels, price_panels = _build_panels("cross_cex_persistence", universe)
    start, end = _data_window(funding_panels)
    LOG.info("data window: %s -> %s", start, end)

    engine_cfg = EngineConfig(
        start=start, end=end,
        step=timedelta(hours=8),  # one decision per 8h funding cycle
        capital_usd=float(strat_cfg_all["global"]["capital_usd"]),
        liquidation_buffer_min=float(strat_cfg_all["global"]["liquidation_buffer_min"]),
        max_strategy_dd=float(strat_cfg_all["global"]["max_strategy_dd"]),
        data_staleness_seconds=86400,   # 8h cycle: more lenient
    )
    cost = CostModel.from_yaml()
    engine = BacktestEngine(engine_cfg, cost, cfg)
    strategy = CrossCEXPersistenceStrategy()
    result = engine.run(strategy, funding_panels, price_panels, _build_depth_panels())
    metrics = summarize_run(result.equity, result.trades,
                            starting_capital=engine_cfg.capital_usd)
    verdict = cross_cex_evaluate(metrics, min_net_apr=float(cfg["min_net_apr"]))
    metrics["verdict"] = verdict

    rejection: Optional[str] = None
    if verdict == "REJECTED":
        rejection = (
            f"net_apr={metrics['net_apr']:.4f} max_dd={metrics['max_drawdown']:.4f} "
            f"-> baseline cannot survive realistic costs"
        )

    inputs = ReportInputs(
        strategy_name="cross_cex_persistence",
        config_snapshot={"engine": vars(engine_cfg), "strategy": cfg},
        metrics=metrics,
        equity=result.equity,
        trades=result.trades,
        rejection_reason=rejection,
    )
    run_dir = write_report(inputs)
    attach_run_logfile(LOG, run_dir / "run.log")
    LOG.info("wrote run dir: %s", run_dir)
    return run_dir


def run_gmx_feasibility(args) -> Path:
    strat_cfg_all = load_yaml(config_dir() / "strategy_params.yaml")
    cfg = strat_cfg_all["gmx_imbalance"]
    cost = CostModel.from_yaml()

    snap_path = normalized_dir() / "gmx_v2_snapshot.parquet"
    bin_path = normalized_dir() / "funding_binance.parquet"
    history_path = normalized_dir() / "gmx_v2_imbalance_history.parquet"

    snapshots = _safe_read(snap_path, kind="funding")  # same dtypes are fine
    bin_funding = _safe_read(bin_path, kind="funding")
    history = read_parquet(history_path) if history_path.exists() else None

    # Group binance funding by base asset (e.g. BTCUSDT -> BTC)
    cex_panels: dict[str, pd.DataFrame] = {}
    if not bin_funding.empty:
        for base, sub in bin_funding.groupby("base_asset"):
            cex_panels[str(base)] = sub.sort_values("timestamp_utc").reset_index(drop=True)

    table, verdict = run_feasibility(
        snapshots, cex_panels, cost_model=cost, config=cfg, history=history,
    )

    metrics = {
        "verdict": verdict["verdict"],
        "candidates": verdict.get("candidates", 0),
        "rows": verdict.get("rows", 0),
        "data_completeness_score": verdict.get("data_completeness_score", 0.0),
        "imbalance_distribution": verdict.get("imbalance_distribution", {}),
        "per_symbol": {},
    }
    if not table.empty:
        metrics["net_apr_distribution"] = {
            "min": float(table["gmx_net_apr"].min()),
            "p25": float(table["gmx_net_apr"].quantile(0.25)),
            "median": float(table["gmx_net_apr"].median()),
            "p75": float(table["gmx_net_apr"].quantile(0.75)),
            "max": float(table["gmx_net_apr"].max()),
        }

    inputs = ReportInputs(
        strategy_name="gmx_imbalance_feasibility",
        config_snapshot={"strategy": cfg},
        metrics=metrics,
        equity=pd.DataFrame(),
        trades=pd.DataFrame(),
        rejection_reason=("verdict=REJECT" if verdict["verdict"] == "REJECT" else None),
    )
    run_dir = write_report(inputs)
    if not table.empty:
        table.to_csv(run_dir / "feasibility_table.csv", index=False)
    attach_run_logfile(LOG, run_dir / "run.log")
    LOG.info("wrote run dir: %s", run_dir)
    return run_dir


def _parse_args():
    p = argparse.ArgumentParser(description="Funding-arb backtest runner")
    p.add_argument("--strategy", required=True,
                   choices=["hl_binance_interval",
                            "hl_cross_venue_disp",
                            "cross_cex_persistence",
                            "cross_cex_residual",
                            "gmx_imbalance_feasibility"])
    return p.parse_args()


def main():
    args = _parse_args()
    if args.strategy == "hl_binance_interval":
        run_hl_binance(args)
    elif args.strategy == "hl_cross_venue_disp":
        run_hl_cross_venue_disp(args)
    elif args.strategy == "cross_cex_persistence":
        run_cross_cex(args)
    elif args.strategy == "cross_cex_residual":
        run_cross_cex_residual(args)
    elif args.strategy == "gmx_imbalance_feasibility":
        run_gmx_feasibility(args)


if __name__ == "__main__":
    main()
