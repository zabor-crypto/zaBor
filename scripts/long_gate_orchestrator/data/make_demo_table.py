#!/usr/bin/env python3
"""Generate a SYNTHETIC unified_signal_table.csv so the optimization / validation / replay
demos run out of the box.

This is NOT real signal data. The real research corpus (per-signal realized_R settled on the
live execution venue) is intentionally not published. This generator fabricates a schema-faithful
table whose realized_R carries a mild, deterministic regime dependence (so the regime gate has
something to find in the demo) plus noise. Use it only to exercise the code paths — the numbers
mean nothing.

Run:  python3 make_demo_table.py   ->  writes ./unified_signal_table.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE / "panel" / "regime_panel.csv"
OUT = HERE / "unified_signal_table.csv"

RNG = np.random.default_rng(20260617)
N_PER_STRATEGY = 1200
SYMS = [f"DEMO{i}USDT" for i in range(40)]

# synthetic edge: each strategy's realized_R drifts with a regime axis (deterministic + noise)
EDGE = {
    "reversal": dict(axis="btc_r3", k=4.0, base=0.05, atr=False),
    "meanrev":  dict(axis="btc_r3", k=6.0, base=-0.10, atr=False),
    "breakout": dict(axis="btc_ema55_up", k=0.8, base=-0.05, atr=True),
}


def main():
    p = pd.read_csv(PANEL)
    p["known_at"] = pd.to_datetime(p["known_at"], utc=True)
    p = p.dropna(subset=["btc_r3"]).reset_index(drop=True)
    rows = []
    for strat, cfg in EDGE.items():
        idx = RNG.integers(0, len(p), N_PER_STRATEGY)
        for j in idx:
            bar = p.iloc[j]
            ts = bar["known_at"]
            axis_val = float(bar[cfg["axis"]])
            atr_pct = float(np.clip(RNG.normal(0.025, 0.012), 0.005, 0.08))
            # synthetic expected R: tilt by the regime axis, penalise high ATR for breakout
            mu = cfg["base"] + cfg["k"] * axis_val * 0.1
            if cfg["atr"]:
                mu -= max(0.0, atr_pct - 0.023) * 8.0
            r = float(RNG.normal(mu, 1.0))
            def g(col, default=0.0):
                v = bar.get(col, default)
                return round(float(v), 6) if pd.notna(v) else default
            rows.append(dict(
                strategy=strat, symbol=SYMS[RNG.integers(0, len(SYMS))],
                ts=ts.isoformat(), realized_R=round(r, 4), entries_filled=1,
                atr_pct=round(atr_pct, 5),
                # lag0 panel features the gate configs read directly (mirror the real schema)
                F5b=g("F5b"), F5=g("F5"), F3=g("F3_btc_down_volexp"), abtr=g("abtr"),
                trend_strong=g("trend_strong"), riskoff_z=g("riskoff_z"),
                btc_up=g("btc_up"), btc_ema55_up=g("btc_ema55_up"),
                btc_r3_panel=g("btc_r3"), breadth_panel=g("breadth"),
                # native-feature columns used by the "existing gate" comparator
                btc_r3_native=g("btc_r3"), breadth_native=g("breadth"),
                btc_z_native=round(float(RNG.normal(0, 1)), 4),
            ))
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"[demo] wrote {len(df)} synthetic signals -> {OUT}")
    print(df.groupby("strategy")["realized_R"].agg(["count", "mean", "sum"]).round(3))


if __name__ == "__main__":
    main()
