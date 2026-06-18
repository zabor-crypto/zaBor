#!/usr/bin/env python3
"""P4 — the validation gate (GO/WATCH/KILL). Applies the project's anti-overfit law to the REFINED gates:

  A. WALK-FORWARD selection  — pick the threshold on train folds, score on the NEXT-UNSEEN fold (the real
     OOS test). Does WFO-selected beat ungated AND the existing per-strategy gate out-of-sample? Do the
     selected thresholds DRIFT (overfit tell) or stay stable?
  B. CLEAN FORWARD HOLDOUT    — tune on first 75% by time, evaluate ONCE on the last 25% the tuner never saw.
  C. LABEL-PERMUTATION PLACEBO — fix the gate, permute realized_R within strategy; does the gate's traded
     subset beat random subsets of the same size? (p-value).  rule 69.
  D. EPISODE HONESTY          — per-month gated vs ungated; remove-best-month; is the edge one-episode-carried?
                                (~4-8 BTC swings in 6mo → count episodes, not days.  rule 64).

Comparators throughout: ungated, existing per-strategy gate, refined gate.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "optimization"))
import gate_sweep as gs

K = 6
FLOOR = {"reversal": 0.40, "meanrev": 0.20, "breakout": 0.40}
RNG = np.random.default_rng(20260617)


def col(feat, lag):
    base = {"riskoff_z": "riskoff_z", "btc_r3": "btc_r3p", "btc_ema55_up": "btc_ema55_up"}[feat]
    return base if lag == 0 else f"{feat}_lag{lag}"


# ---- gate families (param -> bool-keep mask) ----
def m_reversal(d, p):                      # p = btc_r3 cutoff (-99 => ungated)
    return d["btc_r3p"].fillna(9) >= p


def m_meanrev(d, p):                       # p = (btc cutoff, riskoff ceiling)
    tb, tr = p
    return (d[col("btc_r3", 8)].fillna(-9) >= tb) & (d["riskoff_z"].fillna(-9) < tr)


def m_breakout(d, p):                      # p = riskoff ceiling (ema55_up lag8 fixed, no atr filter)
    return (d[col("btc_ema55_up", 8)].fillna(0) > 0.5) & (d["riskoff_z"].fillna(-9) < tr_of(p))


def tr_of(p):
    return p


FAM = {"reversal": m_reversal, "meanrev": m_meanrev, "breakout": m_breakout}
GRID = {"reversal": [-99, -0.04, -0.02, -0.01, 0.0],
        "meanrev": [(tb, tr) for tb in (-0.02, -0.01, 0.0) for tr in (0.5, 1.0, 99)],
        "breakout": [1.0, 1.5, 99]}
REFINED = {"reversal": -0.02, "meanrev": (-0.02, 0.5), "breakout": 1.0}
EXISTING = {  # current live per-strategy gate
    "reversal": lambda d: d["F5b"].fillna(0) < 0.5,
    "meanrev": lambda d: (d["btc_r3_native"].fillna(-9) >= 0) & (d["breadth_native"].between(0.05, 0.15)),
    "breakout": lambda d: d["btc_ema55_up"].fillna(0) > 0.5,
}


def sumR(d, mask):
    return float(d["realized_R"][mask].sum())


def pf(d, mask):
    R = d["realized_R"][mask]; p = R[R > 0].sum(); n = -R[R < 0].sum()
    return (p / n if n > 0 else np.inf), len(R), (100 * (R > 0).mean() if len(R) else np.nan)


def best_on(train, strat):
    """pick grid param maximizing train sumR subject to train retain >= floor."""
    best, bs = None, -1e18
    for p in GRID[strat]:
        mask = FAM[strat](train, p)
        if mask.mean() < FLOOR[strat]:
            continue
        s = sumR(train, mask)
        if s > bs:
            bs, best = s, p
    return best if best is not None else GRID[strat][0]


def walk_forward(d, strat):
    d = d.sort_values("ts").reset_index(drop=True)
    n = len(d); b = [int(n * x / K) for x in range(K + 1)]
    picks, oos_sel, oos_ung, oos_exist = [], 0.0, 0.0, 0.0
    rows = []
    for f in range(1, K):
        train = d.iloc[:b[f]]; test = d.iloc[b[f]:b[f + 1]]
        if len(test) < 10:
            continue
        p = best_on(train, strat)
        msel = FAM[strat](test, p)
        s_sel = sumR(test, msel); s_ung = sumR(test, pd.Series(True, index=test.index))
        s_ex = sumR(test, EXISTING[strat](test))
        picks.append(p); oos_sel += s_sel; oos_ung += s_ung; oos_exist += s_ex
        rows.append((f, p, s_sel, s_ung, s_ex))
    return picks, oos_sel, oos_ung, oos_exist, rows


def holdout(d, strat, frac=0.75):
    d = d.sort_values("ts").reset_index(drop=True)
    cut = int(len(d) * frac); train, test = d.iloc[:cut], d.iloc[cut:]
    p = best_on(train, strat)
    out = {}
    for lbl, mask in (("ungated", pd.Series(True, index=test.index)),
                      ("existing", EXISTING[strat](test)),
                      (f"selected({p})", FAM[strat](test, p)),
                      (f"refined({REFINED[strat]})", FAM[strat](test, REFINED[strat]))):
        f_, nn, w = pf(test, mask)
        out[lbl] = (sumR(test, mask), f_, nn, 100 * mask.mean(), w)
    return p, out, len(test)


def placebo(d, strat, n_perm=2000):
    """fix REFINED gate; permute realized_R; does traded subset's sumR beat random same-size subsets?"""
    mask = FAM[strat](d, REFINED[strat]).values
    R = d["realized_R"].values
    real = float(R[mask].sum())
    k = int(mask.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = RNG.choice(R, size=k, replace=False).sum()
    p = float((null >= real).mean())
    return real, k, float(null.mean()), float(null.std()), p


def episodes(d, strat):
    d = d.copy(); d["month"] = pd.to_datetime(d["ts"], utc=True).dt.strftime("%Y-%m")
    mask = FAM[strat](d, REFINED[strat])
    per = []
    for mth, g in d.groupby("month"):
        gm = FAM[strat](g, REFINED[strat])
        per.append((mth, sumR(g, gm), sumR(g, pd.Series(True, index=g.index)), int(gm.sum()), len(g)))
    lift_total = sumR(d, mask) - 0  # gated sumR (ungated comparison printed per-month)
    # remove best month for gated, check still beats ungated-removed
    gated_by_m = {m: gs_ for (m, gs_, us_, gn, un) in per}
    best_m = max(gated_by_m, key=gated_by_m.get)
    d2 = d[d.month != best_m]
    gated_excl = sumR(d2, FAM[strat](d2, REFINED[strat]))
    ung_excl = sumR(d2, pd.Series(True, index=d2.index))
    return per, best_m, gated_excl, ung_excl


def run():
    u = gs.load(); u = u[u.entries_filled > 0].dropna(subset=["realized_R"]).sort_values("ts")
    verdicts = {}
    for strat in ("reversal", "meanrev", "breakout"):
        d = u[u.strategy == strat].copy()
        print(f"\n################  {strat.upper()}  (n={len(d)})  refined={REFINED[strat]}  ################")

        picks, osel, oung, oexist, rows = walk_forward(d, strat)
        print("A. WALK-FORWARD (select on train, score next-unseen fold):")
        for (f, p, s, su, se) in rows:
            print(f"   fold{f}: pick={str(p):>14}  test sumR  sel={s:+7.1f}  ungated={su:+7.1f}  existing={se:+7.1f}")
        print(f"   OOS TOTAL: selected={osel:+.1f}  ungated={oung:+.1f}  existing={oexist:+.1f}  "
              f"| picks={picks}")

        p_ho, out, ntest = holdout(d, strat)
        print(f"B. CLEAN HOLDOUT (tune first75% -> score last25%, n={ntest}, tuner picked {p_ho}):")
        for lbl, (s, f_, nn, ret, w) in out.items():
            print(f"   {lbl:22} sumR={s:+7.1f}  PF={f_:>5.2f}  n={nn:>4} retain={ret:>3.0f}% win={w:>3.0f}%")

        real, k, nmean, nstd, pval = placebo(d, strat)
        print(f"C. PLACEBO (permute R, refined gate, 2000x): traded sumR={real:+.1f} (k={k})  "
              f"null={nmean:+.1f}±{nstd:.1f}  p={pval:.4f}  {'PASS' if pval<0.05 else 'FAIL'}")

        per, best_m, gex, uex = episodes(d, strat)
        pos = sum(1 for (m, g, us, gn, un) in per if g > 0)
        print(f"D. EPISODE HONESTY: {pos}/{len(per)} months gated-positive; "
              f"remove-best-month({best_m}): gated={gex:+.1f} vs ungated={uex:+.1f} "
              f"{'(still beats)' if gex>uex else '(FAILS w/o best month)'}")

        # verdict heuristic
        wf_ok = osel > oung and osel > oexist
        ho_ok = out[f"selected({p_ho})"][0] > out["ungated"][0] and out[f"selected({p_ho})"][0] > out["existing"][0]
        pl_ok = pval < 0.05
        ep_ok = gex > uex and pos >= len(per) * 0.6
        score = sum([wf_ok, ho_ok, pl_ok, ep_ok])
        verdict = "GO" if score == 4 else ("WATCH" if score >= 2 else "KILL")
        verdicts[strat] = (verdict, dict(wf=wf_ok, holdout=ho_ok, placebo=pl_ok, episode=ep_ok))
        print(f"   => {strat}: WF={wf_ok} HOLDOUT={ho_ok} PLACEBO={pl_ok} EPISODE={ep_ok}  ==> {verdict}")

    print("\n================ P4 VERDICTS ================")
    for s, (v, d_) in verdicts.items():
        print(f"  {s:9} {v:6}  {d_}")


if __name__ == "__main__":
    run()
