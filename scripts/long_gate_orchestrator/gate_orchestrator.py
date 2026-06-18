#!/usr/bin/env python3
"""LONG-GATE ORCHESTRATOR — production decide() API (P5).

  decide(strategy, ts, atr_pct=None) -> {size_mult, trade, reason, features}

Causal, market-state, indicator-agnostic. The gate's only inputs are the regime panel
(data/panel/regime_panel.csv, refreshable from Correlations/data_cache) + (breakout only) the signal's own
atr_pct. Validated configs (P4, 2026-06-17):
  reversal  GO            : trade if btc_r3 >= -0.02
  meanrev   GO (regime)   : trade if btc_r3(lag8) >= -0.02 AND riskoff_z < 0.5
  breakout  WATCH/shadow  : trade if btc_ema55_4h_up(lag8) AND atr_pct <= ATR_CALM   (zero capital until shadow OK)

Graded mode (optional, per BEAR_01 size-down>>block): returns 0.5 instead of 0.0 in the soft band.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd, numpy as np

PANEL = Path(__file__).resolve().parent / "data" / "panel" / "regime_panel.csv"

# validated thresholds (P4 / P4.x)
TH = {
    "reversal": dict(btc_r3_min=-0.02),
    "meanrev":  dict(btc_r3_min=-0.02, btc_lag_h=8, riskoff_max=0.5),
    "breakout": dict(ema55_lag_h=8, atr_calm=0.023),   # capital-eligible: NO (shadow). atr_calm ~= train q25
}
CAPITAL_ELIGIBLE = {"reversal": True, "meanrev": True, "breakout": False}


class Orchestrator:
    def __init__(self, panel_path: Path = PANEL):
        p = pd.read_csv(panel_path)
        p["known_at"] = pd.to_datetime(p["known_at"], utc=True)
        self.p = p.sort_values("known_at").reset_index(drop=True)
        self.ka = self.p["known_at"].values

    def _asof(self, ts, col, lag_h=0):
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if lag_h:
            ts = ts - pd.Timedelta(hours=lag_h)
        i = np.searchsorted(self.ka, np.datetime64(ts.to_datetime64()), side="right") - 1
        if i < 0:
            return None
        s = self.p[col].iloc[:i + 1]      # last NON-NaN at-or-before ts (causal; skips partial latest bar)
        s = s[s.notna()]
        return float(s.iloc[-1]) if len(s) else None

    def decide(self, strategy: str, ts, atr_pct: float | None = None, graded: bool = False) -> dict:
        def fmt(x, d=3):
            return "NA" if x is None else f"{x:+.{d}f}"
        th = TH[strategy]; feats = {}
        if strategy == "reversal":
            br = self._asof(ts, "btc_r3"); feats["btc_r3"] = br
            ok = (br is not None) and (br >= th["btc_r3_min"])
            reason = f"btc_r3={fmt(br)} {'>=' if ok else '<'} {th['btc_r3_min']}"
        elif strategy == "meanrev":
            br = self._asof(ts, "btc_r3", th["btc_lag_h"]); rz = self._asof(ts, "riskoff_z")
            feats.update(btc_r3_lag8=br, riskoff_z=rz)
            ok = (br is not None and br >= th["btc_r3_min"]) and (rz is not None and rz < th["riskoff_max"])
            reason = f"btc_r3(lag8)={fmt(br)}>=-0.02 & riskoff_z={fmt(rz,2)}<{th['riskoff_max']} -> {ok}"
        elif strategy == "breakout":
            up = self._asof(ts, "btc_ema55_up", th["ema55_lag_h"]); feats["ema55_up_lag8"] = up
            feats["atr_pct"] = atr_pct
            calm = (atr_pct is not None) and (atr_pct <= th["atr_calm"])
            ok = (up is not None and up > 0.5) and calm
            reason = f"ema55_up(lag8)={up} & atr_pct={atr_pct} <= {th['atr_calm']} -> {ok}"
        else:
            raise ValueError(strategy)
        if ok:
            mult = 1.0
        elif graded:
            mult = 0.5
        else:
            mult = 0.0
        return dict(strategy=strategy, ts=str(ts), size_mult=mult, trade=mult > 0,
                    capital_eligible=CAPITAL_ELIGIBLE[strategy], reason=reason, features=feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=None, help="ISO ts (default latest panel bar)")
    ap.add_argument("--atr-pct", type=float, default=0.02, help="breakout signal atr_pct for the demo")
    a = ap.parse_args()
    o = Orchestrator()
    ts = a.at or str(o.p["known_at"].iloc[-1])
    print(f"LONG-GATE ORCHESTRATOR decision @ {ts}")
    for s in ("reversal", "meanrev", "breakout"):
        d = o.decide(s, ts, atr_pct=a.atr_pct if s == "breakout" else None)
        flag = "TRADE" if d["trade"] else "SKIP "
        cap = "" if d["capital_eligible"] else " (shadow-only)"
        print(f"  {s:9} {flag} size={d['size_mult']}{cap}   {d['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
