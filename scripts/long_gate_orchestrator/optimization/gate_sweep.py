#!/usr/bin/env python3
"""P3 — two-stage gate sweep (Stage-A realized_R cached in unified table; Stage-B = vectorized lookups).

Evaluates a CURATED (not open-Optuna) set of gate architectures per strategy + shared/global, against
the operator's DUAL objective: maximize PF & sumR (ROI) WHILE retaining signal count. Reports pooled
PF/sumR/medR/win/retain% AND a time-ordered 6-fold MAXIMIN worst-fold sumR (P4 preview) so we never pick
a pooled-PF-only winner. Lagged panel joins included to test the reversal 8-12h lag prior + graded (off/
half/full) per BEAR_01's size-down>>hard-block.

Axes (all causal, market-state, from the P2 panel): F5b/F5/F3 risk-off, trend_strong=|btc_trend_z|>=1,
btc_up=btc_r3>=0, btc_ema55_up (breakout self-gate), breadth. Native meanrev breadth/btc_r3 also available.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
LAG_FEATS = ["F5b", "F5", "F3", "trend_strong", "btc_up", "btc_ema55_up", "abtr", "btc_r3", "breadth", "riskoff_z"]
LAGS = [4, 8, 12, 16, 24]
K = 6


def load():
    u = pd.read_csv(DATA / "unified_signal_table.csv")
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u = u[u.entries_filled > 0].dropna(subset=["realized_R"]).sort_values("ts").reset_index(drop=True)
    panel = pd.read_csv(DATA / "panel" / "regime_panel.csv")
    panel["known_at"] = pd.to_datetime(panel["known_at"], utc=True)
    panel = panel.rename(columns={"F3_btc_down_volexp": "F3", "btc_r3": "btc_r3p", "breadth": "breadthp"})
    pc = {"F5b": "F5b", "F5": "F5", "F3": "F3", "trend_strong": "trend_strong", "btc_up": "btc_up",
          "btc_ema55_up": "btc_ema55_up", "abtr": "abtr", "btc_r3": "btc_r3p", "breadth": "breadthp",
          "riskoff_z": "riskoff_z"}
    panel = panel[["known_at"] + list(pc.values())].sort_values("known_at")
    # lagged asof joins: panel value as of (ts - L)
    for L in LAGS:
        u[f"_t{L}"] = u["ts"] - pd.Timedelta(hours=L)
        j = pd.merge_asof(u[["_t{}".format(L)]].rename(columns={f"_t{L}": "k"}).sort_values("k"),
                          panel, left_on="k", right_on="known_at", direction="backward")
        for feat, col in pc.items():
            u[f"{feat}_lag{L}"] = j[col].values
    # lag0 aliases (unifier named the panel btc_r3/breadth as *_panel)
    u["btc_r3p"] = u["btc_r3_panel"]
    u["breadthp"] = u["breadth_panel"]
    return u


def metrics(R: pd.Series):
    R = R.dropna()
    if len(R) == 0:
        return dict(n=0, pf=np.nan, sumR=0.0, medR=np.nan, win=np.nan)
    pos = R[R > 0].sum(); neg = -R[R < 0].sum()
    return dict(n=len(R), pf=(pos / neg if neg > 0 else np.inf), sumR=R.sum(),
                medR=R.median(), win=100 * (R > 0).mean())


def maximin_worstfold(df: pd.DataFrame, mult: pd.Series):
    """Time-ordered K contiguous folds; per-fold sumR of GATED returns; return worst-fold sumR + #pos."""
    d = df.sort_values("ts")
    gR = (d["realized_R"].values * mult.loc[d.index].values)
    n = len(gR)
    if n < K:
        return np.nan, 0
    bounds = [int(n * x / K) for x in range(K + 1)]
    folds = [gR[bounds[i]:bounds[i + 1]] for i in range(K)]
    sums = [float(np.nansum(f)) for f in folds]
    return min(sums), sum(1 for s in sums if s > 0)


# ---- gate config registry: name -> fn(df)->size_mult (1.0 trade / 0.5 halve / 0.0 skip) ----
def g_ungated(d):
    return pd.Series(1.0, index=d.index)


def skip_if(col):
    return lambda d: pd.Series(np.where(d[col].fillna(0) > 0.5, 0.0, 1.0), index=d.index)


def halve_if(col):
    return lambda d: pd.Series(np.where(d[col].fillna(0) > 0.5, 0.5, 1.0), index=d.index)


def keep_if(col):  # skip unless col true (for ema55_up / btc_up positive gates)
    return lambda d: pd.Series(np.where(d[col].fillna(0) > 0.5, 1.0, 0.0), index=d.index)


def keep_if_up():  # btc_r3>=0 using panel btc_r3p
    return lambda d: pd.Series(np.where(d["btc_r3p"].fillna(-1) >= 0, 1.0, 0.0), index=d.index)


def halve_if_not(col):
    return lambda d: pd.Series(np.where(d[col].fillna(0) > 0.5, 1.0, 0.5), index=d.index)


CONFIGS = {
    "reversal": {
        "ungated": g_ungated,
        "f5b_skip": skip_if("F5b"),
        "f5b_skip_lag8": skip_if("F5b_lag8"),
        "f5b_skip_lag12": skip_if("F5b_lag12"),
        "f5b_halve": halve_if("F5b"),
        "f5b_halve_lag8": halve_if("F5b_lag8"),
        "trend_skip": skip_if("trend_strong"),
        "trend_riskoff_skip": lambda d: pd.Series(
            np.where((d["F5b"].fillna(0) > 0.5) | (d["trend_strong"].fillna(0) > 0.5), 0.0, 1.0), index=d.index),
        "trend_riskoff_halve": lambda d: pd.Series(
            np.where((d["F5b"].fillna(0) > 0.5) | (d["trend_strong"].fillna(0) > 0.5), 0.5, 1.0), index=d.index),
        "f5b_skip_btcdown_halve": lambda d: pd.Series(
            np.where(d["F5b"].fillna(0) > 0.5, 0.0, np.where(d["btc_r3p"].fillna(0) < 0, 0.5, 1.0)), index=d.index),
        "btc_up_keep": keep_if_up(),
    },
    "meanrev": {
        "ungated": g_ungated,
        "btc_up_keep": keep_if_up(),
        "btc_up_keep_lag8": lambda d: pd.Series(np.where(d["btc_r3_lag8"].fillna(-1) >= 0, 1.0, 0.0), index=d.index),
        "btc_up_halve_down": lambda d: pd.Series(np.where(d["btc_r3p"].fillna(0) >= 0, 1.0, 0.5), index=d.index),
        "deployed_btcup_breadth": lambda d: pd.Series(np.where(
            (d["btc_r3_native"].fillna(-1) >= 0) & (d["breadth_native"].between(0.05, 0.15)), 1.0, 0.0), index=d.index),
        "btc_up_and_notf5b": lambda d: pd.Series(np.where(
            (d["btc_r3p"].fillna(-1) >= 0) & (d["F5b"].fillna(1) < 0.5), 1.0, 0.0), index=d.index),
        "f5b_skip": skip_if("F5b"),
        "ema55_up_keep": keep_if("btc_ema55_up"),
    },
    "breakout": {
        "ungated": g_ungated,
        "ema55_up_keep": keep_if("btc_ema55_up"),
        "ema55_up_keep_lag8": keep_if("btc_ema55_up_lag8"),
        "ema55_up_lowatr": lambda d: pd.Series(np.where(
            (d["btc_ema55_up"].fillna(0) > 0.5) & (d["atr_pct"] <= d["atr_pct"].quantile(0.40)), 1.0, 0.0), index=d.index),
        "btc_up_keep": keep_if_up(),
        "ema55_up_halve_else": halve_if_not("btc_ema55_up"),
        "ema55_up_and_notf5b": lambda d: pd.Series(np.where(
            (d["btc_ema55_up"].fillna(0) > 0.5) & (d["F5b"].fillna(1) < 0.5), 1.0, 0.0), index=d.index),
        "f5b_skip": skip_if("F5b"),
    },
}


def run():
    u = load()
    print(f"[sweep] filled signals: {len(u)}  span {u.ts.min().date()}->{u.ts.max().date()}\n")
    summary = {}
    for strat in ("reversal", "meanrev", "breakout"):
        d = u[u.strategy == strat].copy()
        base_n = len(d)
        ung = metrics(d["realized_R"])
        base_worst, _ = maximin_worstfold(d, g_ungated(d))
        print(f"================ {strat.upper()}  (ungated n={base_n} PF={ung['pf']:.3f} "
              f"sumR={ung['sumR']:+.1f} worstfold={base_worst:+.1f}) ================")
        print(f"{'config':28} {'n':>5} {'ret%':>5} {'PF':>6} {'sumR':>8} {'medR':>7} {'win%':>5} "
              f"{'wfMin':>7} {'wfPos':>5}")
        rows = []
        for name, fn in CONFIGS[strat].items():
            mult = fn(d).reindex(d.index).fillna(1.0)
            kept = d[mult > 0]
            gR = d["realized_R"] * mult
            m = metrics(gR[mult > 0])  # metrics on traded signals (PF of what we actually trade)
            wf, wfpos = maximin_worstfold(d, mult)
            ret = 100 * (mult > 0).mean()
            rows.append((name, m, ret, wf, wfpos, gR.sum()))
            print(f"{name:28} {m['n']:>5} {ret:>5.0f} {m['pf']:>6.3f} {gR.sum():>+8.1f} "
                  f"{m['medR']:>+7.3f} {m['win']:>5.0f} {wf:>+7.1f} {wfpos:>4}/6")
        summary[strat] = rows
        print()
    return u, summary


if __name__ == "__main__":
    run()
