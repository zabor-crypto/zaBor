#!/usr/bin/env python3
"""Multi-strategy forward-shadow for the LONG-GATE ORCHESTRATOR (P5).

Logs, per signal, the orchestrator's decision vs an UNGATED control arm — the clean A/B that decides
GO-live (per-tag size_mult) without capital. Two modes:
  --replay   : re-run over the backtest unified table -> ledger + A/B summary (what the shadow WILL show).
  (live)     : import shadow_signal(row) into each scanner's emit path / a VPS cron over the signal feeds;
               append a row to data/shadow/<strategy>_shadow_ledger.csv at every fired signal.

Ledger row: ts, strategy, symbol, gated_mult, reason, realized_R(if known), arm tags. The verdict is the
gated-vs-ungated Δ on accrued live signals (n>=~25-30/strategy), cross-checked vs the backtest expectation.
"""
from __future__ import annotations
import sys, argparse, csv
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gate_orchestrator import Orchestrator, CAPITAL_ELIGIBLE

LEDGER_DIR = ROOT / "data" / "shadow"
COLS = ["ts", "strategy", "symbol", "gated_mult", "ungated_mult", "capital_eligible", "reason",
        "atr_pct", "realized_R"]


def shadow_signal(o: Orchestrator, strategy: str, ts, symbol: str, atr_pct=None, realized_R=None) -> dict:
    d = o.decide(strategy, ts, atr_pct=atr_pct)
    return dict(ts=str(ts), strategy=strategy, symbol=symbol, gated_mult=d["size_mult"], ungated_mult=1.0,
                capital_eligible=CAPITAL_ELIGIBLE[strategy], reason=d["reason"], atr_pct=atr_pct,
                realized_R=realized_R)


def append_ledger(strategy: str, row: dict):
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    f = LEDGER_DIR / f"{strategy}_shadow_ledger.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in COLS})


def replay():
    o = Orchestrator()
    u = pd.read_csv(ROOT / "data" / "unified_signal_table.csv")
    u = u[u.entries_filled > 0].dropna(subset=["realized_R"])
    print(f"{'strategy':9} {'arm':9} {'n':>5} {'PF':>6} {'sumR':>8} {'avgR':>7}  Δ vs ungated")
    for s in ("reversal", "meanrev", "breakout"):
        d = u[u.strategy == s].copy()
        mult = np.array([o.decide(s, r.ts, atr_pct=(r.atr_pct if s == "breakout" else None))["size_mult"]
                         for r in d.itertuples()])
        R = d["realized_R"].values
        for arm, m in (("ungated", np.ones_like(mult)), ("gated", mult)):
            Rt = R[m > 0]
            p = Rt[Rt > 0].sum(); n = -Rt[Rt < 0].sum()
            pf = p / n if n > 0 else float("inf")
            tag = "" if arm == "ungated" else f"  ΔsumR={(R*mult).sum()-R.sum():+.1f}  ΔPF={pf - (R[R>0].sum()/-R[R<0].sum()):+.2f}"
            print(f"{s:9} {arm:9} {len(Rt):>5} {pf:>6.3f} {(R*m if arm=='gated' else R).sum():>+8.1f} "
                  f"{Rt.mean():>+7.3f}{tag}")
    print("\n(replay = the A/B the live shadow will accrue; capital_eligible=False -> breakout logged only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    if a.replay:
        replay()
    else:
        print("live mode: import shadow_signal() into the scanner emit path or a VPS cron. See P5_DEPLOY_TICKET.md")
