"""Report builder: writes a per-run folder with metrics, charts, and CSVs.

Layout:

    outputs/runs/<YYYYMMDD_HHMMSS>_<strategy>/
        config_snapshot.yaml
        metrics.json
        equity_curve.csv
        equity_curve.parquet
        trades.parquet
        funding_spread.csv         (optional, strategy-specific)
        fee_drag.json
        run.log
        charts/
            equity.png
            drawdown.png
            funding_spread.png     (if data present)
            fee_drag.png
        rejection_report.md        (if applicable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.io import dump_json, dump_yaml, outputs_dir, write_parquet


@dataclass
class ReportInputs:
    strategy_name: str
    config_snapshot: dict
    metrics: dict
    equity: pd.DataFrame
    trades: pd.DataFrame
    funding_spread: Optional[pd.DataFrame] = None
    rejection_reason: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


def make_run_dir(strategy_name: str) -> Path:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir() / "runs" / f"{ts}_{strategy_name}"
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    return run_dir


def _plot_equity(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(equity["timestamp_utc"]), equity["equity_usd"])
    ax.set_title("Equity curve")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Equity (USD)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_drawdown(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    eq = equity["equity_usd"]
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(pd.to_datetime(equity["timestamp_utc"]), dd, 0, color="tab:red", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_spread(spread_df: pd.DataFrame, path: Path) -> None:
    if spread_df is None or spread_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ts = pd.to_datetime(spread_df["timestamp_utc"])
    for col in [c for c in spread_df.columns if c != "timestamp_utc"]:
        ax.plot(ts, spread_df[col], label=col, linewidth=0.8)
    ax.set_title("Funding spread")
    ax.set_xlabel("UTC")
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_fee_drag(metrics: dict, path: Path) -> None:
    per_sym = metrics.get("per_symbol") or {}
    if not per_sym:
        return
    syms = list(per_sym.keys())
    fees = [per_sym[s].get("fees_usd", 0.0) for s in syms]
    funding = [per_sym[s].get("funding_pnl_usd", 0.0) for s in syms]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(syms))
    ax.bar([i - 0.2 for i in x], funding, width=0.4, label="funding pnl")
    ax.bar([i + 0.2 for i in x], [-f for f in fees], width=0.4, label="-fees")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(syms, rotation=45, ha="right")
    ax.set_ylabel("USD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(inputs: ReportInputs) -> Path:
    run_dir = make_run_dir(inputs.strategy_name)

    dump_yaml(inputs.config_snapshot, run_dir / "config_snapshot.yaml")
    dump_json(inputs.metrics, run_dir / "metrics.json")

    if not inputs.equity.empty:
        inputs.equity.to_csv(run_dir / "equity_curve.csv", index=False)
        write_parquet(inputs.equity, run_dir / "equity_curve.parquet")
    if not inputs.trades.empty:
        write_parquet(inputs.trades, run_dir / "trades.parquet")
        inputs.trades.to_csv(run_dir / "trades.csv", index=False)

    if inputs.funding_spread is not None and not inputs.funding_spread.empty:
        inputs.funding_spread.to_csv(run_dir / "funding_spread.csv", index=False)

    # Charts
    _plot_equity(inputs.equity, run_dir / "charts" / "equity.png")
    _plot_drawdown(inputs.equity, run_dir / "charts" / "drawdown.png")
    if inputs.funding_spread is not None:
        _plot_spread(inputs.funding_spread, run_dir / "charts" / "funding_spread.png")
    _plot_fee_drag(inputs.metrics, run_dir / "charts" / "fee_drag.png")

    if inputs.rejection_reason:
        (run_dir / "rejection_report.md").write_text(
            f"# Strategy rejection report\n\n"
            f"**Strategy:** {inputs.strategy_name}\n\n"
            f"**Reason:** {inputs.rejection_reason}\n\n"
            f"## Key metrics\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in inputs.metrics.items()
                        if k not in ("per_symbol",))
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "strategy": inputs.strategy_name,
        **{k: v for k, v in inputs.metrics.items() if k != "per_symbol"},
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    return run_dir
