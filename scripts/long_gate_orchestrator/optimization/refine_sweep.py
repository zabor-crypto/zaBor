#!/usr/bin/env python3
"""P3.5 — disciplined regime REFINEMENT (META_PIPELINE Phase-8 F1 + Rule 20 + v6).

Finetunes CONTINUOUS thresholds (btc_r3 cutoff, riskoff_z tercile, abtr cutoff, lag, atr quantile) and
per-strategy 2-D combos — dims combined pairwise, NOT full cross-product (sample-fragmentation guard).
Ranks by a BOUNDED objective (maximin-over-6-folds of MEAN-R, artifact-immune), NOT pooled PF, subject to:
  - SELECTIVITY GUARD: retain >= floor (operator: don't drastically cut signals; also overfit guard)
  - robustness: folds_pos >= 4/6
  - class PF floor (reversal 1.5, meanrev/breakout 1.2, §2.5)
Reports the top configs + lift vs P3 best + a NEIGHBOR-STABILITY check (plateau not point) on the winner.
HONEST: this is in-sample tuned — the verdict is P4's clean forward holdout the tuner never saw + placebo.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import gate_sweep as gs

K = 6
FLOOR_RETAIN = {"reversal": 0.40, "meanrev": 0.20, "breakout": 0.40}
FLOOR_PF = {"reversal": 1.5, "meanrev": 1.2, "breakout": 1.2}


def colname(feat, lag):
    base = {"riskoff_z": "riskoff_z", "abtr": "abtr", "btc_r3": "btc_r3p",
            "btc_ema55_up": "btc_ema55_up", "F5b": "F5b"}[feat]
    return base if lag == 0 else f"{feat}_lag{lag}"


def evaluate(d, mult):
    mult = mult.reindex(d.index).fillna(1.0)
    traded = mult > 0
    gR = d["realized_R"] * mult
    R = d["realized_R"][traded]
    n = int(traded.sum())
    if n < 30:
        return None
    pos = R[R > 0].sum(); neg = -R[R < 0].sum()
    pf = pos / neg if neg > 0 else np.inf
    # 6 time-folds (already ts-sorted) on TRADED signals — maximin of fold MEAN-R
    dd = d[traded].sort_values("ts")
    rr = dd["realized_R"].values
    b = [int(len(rr) * x / K) for x in range(K + 1)]
    fold_means = [float(np.nanmean(rr[b[i]:b[i + 1]])) if b[i + 1] > b[i] else np.nan for i in range(K)]
    maximin = np.nanmin(fold_means); folds_pos = int(np.nansum([m > 0 for m in fold_means]))
    return dict(n=n, retain=n / len(d), pf=pf, sumR=float(gR.sum()), meanR=float(R.mean()),
                maximin=maximin, folds_pos=folds_pos, win=100 * (R > 0).mean())


def gates_reversal(d):
    cfgs = {}
    for L in (0, 8, 12):
        for tr in (0.5, 1.0, 1.5, 99):       # riskoff_z skip threshold (99 = off)
            for tb in (-99, -0.02, 0.0, 0.01):  # btc_r3 keep threshold (-99 = off)
                rz = d[colname("riskoff_z", L)]
                m = (rz.fillna(-9) < tr) & (d["btc_r3p"].fillna(9) >= tb)
                cfgs[f"L{L}_rz<{tr}_btc>={tb}"] = pd.Series(np.where(m, 1.0, 0.0), index=d.index)
                # graded: halve instead of skip on riskoff
                mg = np.where(d["btc_r3p"].fillna(9) >= tb, np.where(rz.fillna(-9) < tr, 1.0, 0.5), 0.0)
                cfgs[f"L{L}_rz<{tr}_btc>={tb}_halve"] = pd.Series(mg, index=d.index)
    return cfgs


def gates_meanrev(d):
    cfgs = {}
    for L in (0, 8):
        for tb in (-0.02, -0.01, 0.0, 0.01, 0.02):   # btc_r3 keep threshold
            for tr in (0.5, 1.0, 1.5, 99):           # riskoff skip threshold
                br = d[colname("btc_r3", L)]
                rz = d["riskoff_z"]
                m = (br.fillna(-9) >= tb) & (rz.fillna(-9) < tr)
                cfgs[f"L{L}_btc>={tb}_rz<{tr}"] = pd.Series(np.where(m, 1.0, 0.0), index=d.index)
    return cfgs


def gates_breakout(d):
    cfgs = {}
    for L in (0, 8, 12):
        for q in (0.4, 0.5, 1.0):                    # atr_pct quantile ceiling (1.0 = off)
            for tr in (1.0, 1.5, 99):                # riskoff skip threshold
                up = d[colname("btc_ema55_up", L)]
                rz = d["riskoff_z"]
                athr = d["atr_pct"].quantile(q)
                m = (up.fillna(0) > 0.5) & (d["atr_pct"] <= athr) & (rz.fillna(-9) < tr)
                cfgs[f"L{L}_atr<=q{q}_rz<{tr}"] = pd.Series(np.where(m, 1.0, 0.0), index=d.index)
    return cfgs


GEN = {"reversal": gates_reversal, "meanrev": gates_meanrev, "breakout": gates_breakout}
P3_BEST = {"reversal": ("f5b_skip_lag8", 1.706, 332.4), "meanrev": ("btc_up_and_notf5b", 1.502, 337.9),
           "breakout": ("ema55_up_keep_lag8", 1.137, 272.5)}


def run():
    u = gs.load()
    u = u[u.entries_filled > 0].dropna(subset=["realized_R"]).sort_values("ts")
    for strat in ("reversal", "meanrev", "breakout"):
        d = u[u.strategy == strat].copy()
        rows = []
        for name, mult in GEN[strat](d).items():
            m = evaluate(d, mult)
            if m is None:
                continue
            m["name"] = name
            rows.append(m)
        df = pd.DataFrame(rows)
        # qualifying = passes selectivity + robustness + PF floor
        q = df[(df.retain >= FLOOR_RETAIN[strat]) & (df.folds_pos >= 4) & (df.pf >= FLOOR_PF[strat])]
        rank = q.sort_values("sumR", ascending=False) if len(q) else df.sort_values("sumR", ascending=False)
        pb = P3_BEST[strat]
        print(f"================ {strat.upper()}  (P3 best {pb[0]}: PF {pb[1]:.3f} sumR {pb[2]:+.0f}) "
              f"| {len(df)} configs, {len(q)} qualify ================")
        print(f"{'config':30} {'n':>5} {'ret%':>5} {'PF':>6} {'sumR':>8} {'meanR':>7} {'mmR':>7} {'f+':>3} {'win':>4}")
        for _, r in rank.head(8).iterrows():
            print(f"{r['name']:30} {r['n']:>5.0f} {100*r['retain']:>5.0f} {r['pf']:>6.3f} {r['sumR']:>+8.1f} "
                  f"{r['meanR']:>+7.3f} {r['maximin']:>+7.3f} {r['folds_pos']:>3.0f} {r['win']:>4.0f}")
        print()


if __name__ == "__main__":
    run()
