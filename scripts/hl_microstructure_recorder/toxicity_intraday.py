#!/usr/bin/env python3
"""
Intraday toxic-STATE persistence measurement for HL MM research
(HL market-making research).

The 07-06 finding: no coin has a stable *daily* maker_net sign (SPX +8.2bp was a
partial-day artifact; every mid-tier perp flips sign day-to-day). That kills
STATIC per-coin selection (S2). It does NOT answer whether toxicity is a
persistent *intraday* regime you could GATE on in real time (S2-regime / S3
one-sided). That is what this tool measures — and it is the binding precondition: **a regime gate is only viable
if the state persists LONGER than the lag needed to detect it.**

METHOD (per coin, reusing toxicity.stream_fills — same verified attribution):
  1. Bin the wall-clock into fixed windows (default 5 min). Per bin compute the
     STATE = volume-weighted maker_net@60s over the fills in that bin
     (maker_net = -taker_markout - fee; <0 = toxic bin, >=0 = benign bin).
  2. PERSISTENCE of the continuous state:
     - ACF at lags 1..K over defined (has-flow) bins, pairwise-complete (honest on
       gaps). half_life = first lag where ACF < 0.5 (in wall-clock = lag*bin).
       ACF(1) < ~0 ⇒ state flips within one bin ⇒ NO gate at this resolution.
  3. STATE regime structure (adjacent defined bins only):
     - P(benign|benign), P(toxic|toxic), mean run-lengths, benign base rate.
  4. GATE-VIABILITY (the payoff): conditional NEXT-bin maker_net given the PRIOR
     bin's sign. "Quote only after a benign bin" is viable ⇔
     E[next_net | prev benign] > 0 AND the lift over unconditional is real.
     Placebo: permute the sign sequence N times, p = frac(|lift_perm| >= |lift|).
  5. TIME-OF-DAY: mean maker_net by UTC hour — a persistent, learnable structure
     (a clock gate) vs a fast-flipping one.
  6. DETECTION-LAG proxies: median trades/bin & notional/bin (estimate noise),
     fraction of bins with flow, median gap between defined bins.

Verdict per coin: gate_viable = (ACF(1) shows persistence) AND
(E[next|benign] > 0) AND (placebo p < 0.10) AND (benign base rate in [0.2,0.8]).
Otherwise: no real-time toxicity gate on this coin at this bin size.

No orders, no capital. Tolerates truncated live gz. Import-safe (guarded main).

Usage:
  toxicity_intraday.py --root ./data --coins SPX,DYDX,PURR/USDC \
      --days 2026-07-03,2026-07-04,2026-07-05 --bin-min 5 --json
"""
import argparse, glob, json, math, os, random
from collections import defaultdict

import toxicity as tox  # reuse verified load_mid_series / stream_fills / maker_fee_bps / is_spot

H_STATE = tox.H1_MS       # 60s markout defines the state
ACF_MAX_LAG = 12
N_PERM = 500
SEED = 12345


def bin_states(root, coin, days, bin_ms):
    """Return per-bin state records: {idx, t0, net, mk, vol, n, ofi, hour} for bins with flow."""
    times, mids, halfs = tox.load_mid_series(root, coin, days)
    if len(times) < 50:
        return None, f"insufficient bbo ({len(times)} pts)"
    fee = tox.maker_fee_bps(coin)
    bins = {}
    for f in tox.stream_fills(times, mids, halfs, root, coin, days):
        mk = f["mk"].get(H_STATE)
        if mk is None or f["half_spread"] is None:
            continue
        idx = f["t"] // bin_ms
        d = bins.get(idx)
        if d is None:
            d = bins[idx] = {"idx": idx, "t0": idx * bin_ms, "vol": 0.0, "net_w": 0.0,
                             "mk_w": 0.0, "n": 0, "buy": 0.0, "sell": 0.0}
        d["vol"] += f["notional"]; d["net_w"] += (-mk - fee) * f["notional"]
        d["mk_w"] += mk * f["notional"]; d["n"] += 1
        if f["s"] > 0: d["buy"] += f["notional"]
        else: d["sell"] += f["notional"]
    recs = []
    for idx in sorted(bins):
        d = bins[idx]
        if d["vol"] <= 0:
            continue
        net = d["net_w"] / d["vol"]
        ofi = (d["buy"] - d["sell"]) / d["vol"]           # signed order-flow imbalance in bin
        hour = int((d["t0"] // 3_600_000) % 24)
        recs.append({"idx": idx, "t0": d["t0"], "net": net, "mk": d["mk_w"] / d["vol"],
                     "vol": d["vol"], "n": d["n"], "ofi": ofi, "hour": hour})
    return recs, None


def acf_pairwise(recs, max_lag):
    """Pairwise-complete ACF of the .net series indexed by bin idx (gaps respected)."""
    by_idx = {r["idx"]: r["net"] for r in recs}
    idxs = [r["idx"] for r in recs]
    mu = sum(by_idx.values()) / len(by_idx)
    var = sum((v - mu) ** 2 for v in by_idx.values()) / len(by_idx)
    out = {}
    if var <= 0:
        return out
    for lag in range(1, max_lag + 1):
        num = 0.0; npair = 0
        for i in idxs:
            j = i + lag
            if j in by_idx:
                num += (by_idx[i] - mu) * (by_idx[j] - mu); npair += 1
        out[lag] = {"acf": round(num / npair / var, 3) if npair else None, "npairs": npair}
    return out


def adjacency(recs):
    """Adjacent defined-bin pairs (prev, next) where next.idx == prev.idx+1."""
    by_idx = {r["idx"]: r for r in recs}
    pairs = []
    for r in recs:
        nxt = by_idx.get(r["idx"] + 1)
        if nxt is not None:
            pairs.append((r, nxt))
    return pairs


def analyze(root, coin, days, bin_ms):
    recs, err = bin_states(root, coin, days, bin_ms)
    if err:
        return {"coin": coin, "error": err}
    if len(recs) < 10:
        return {"coin": coin, "error": f"only {len(recs)} bins with flow"}

    nets = [r["net"] for r in recs]
    benign = [r for r in recs if r["net"] >= 0]
    base_benign = len(benign) / len(recs)
    uncond = sum(r["net"] * r["vol"] for r in recs) / sum(r["vol"] for r in recs)

    # persistence
    acf = acf_pairwise(recs, ACF_MAX_LAG)
    half_life = None
    for lag in range(1, ACF_MAX_LAG + 1):
        a = acf.get(lag, {}).get("acf")
        if a is not None and a < 0.5:
            half_life = lag; break
    acf1 = acf.get(1, {}).get("acf")

    # regime structure on adjacent pairs
    pairs = adjacency(recs)
    n_adj = len(pairs)
    p_bb = p_tt = None
    if n_adj:
        b_prev = [(p, n) for p, n in pairs if p["net"] >= 0]
        t_prev = [(p, n) for p, n in pairs if p["net"] < 0]
        p_bb = round(sum(1 for _, n in b_prev if n["net"] >= 0) / len(b_prev), 3) if b_prev else None
        p_tt = round(sum(1 for _, n in t_prev if n["net"] < 0) / len(t_prev), 3) if t_prev else None
        # conditional next-bin maker_net (vol-weighted by NEXT bin volume)
        def cond(prevset):
            if not prevset: return None, 0
            vw = sum(n["net"] * n["vol"] for _, n in prevset) / sum(n["vol"] for _, n in prevset)
            return round(vw, 3), len(prevset)
        next_if_benign, n_b = cond(b_prev)
        next_if_toxic, n_t = cond(t_prev)
    else:
        next_if_benign = next_if_toxic = None; n_b = n_t = 0

    lift = (next_if_benign - next_if_toxic) if (next_if_benign is not None and next_if_toxic is not None) else None

    # placebo: permute the sign sequence, recompute lift on adjacency
    p_value = None
    if lift is not None and n_adj >= 20:
        rng = random.Random(SEED)
        seq = [(r["net"], r["vol"]) for r in recs]
        idx_order = [r["idx"] for r in recs]
        pos = {ix: k for k, ix in enumerate(idx_order)}
        adj_k = [(pos[p["idx"]], pos[n["idx"]]) for p, n in pairs]
        hits = 0
        for _ in range(N_PERM):
            perm = seq[:]; rng.shuffle(perm)
            bset = [(perm[a], perm[c]) for a, c in adj_k if perm[a][0] >= 0]
            tset = [(perm[a], perm[c]) for a, c in adj_k if perm[a][0] < 0]
            if not bset or not tset: continue
            nb = sum(nx[0] * nx[1] for _, nx in bset) / sum(nx[1] for _, nx in bset)
            nt = sum(nx[0] * nx[1] for _, nx in tset) / sum(nx[1] for _, nx in tset)
            if abs(nb - nt) >= abs(lift): hits += 1
        p_value = round(hits / N_PERM, 3)

    # time-of-day
    hod = defaultdict(lambda: {"vol": 0.0, "net_w": 0.0, "n": 0})
    for r in recs:
        h = hod[r["hour"]]; h["vol"] += r["vol"]; h["net_w"] += r["net"] * r["vol"]; h["n"] += 1
    hod_profile = {h: {"maker_net": round(d["net_w"] / d["vol"], 2), "nbins": d["n"]}
                   for h, d in sorted(hod.items()) if d["vol"] > 0}

    # detection-lag proxies
    ns = sorted(r["n"] for r in recs)
    vols = sorted(r["vol"] for r in recs)
    span_bins = recs[-1]["idx"] - recs[0]["idx"] + 1
    gaps = sorted(recs[i + 1]["idx"] - recs[i]["idx"] for i in range(len(recs) - 1)) or [0]

    gate_viable = bool(
        acf1 is not None and acf1 >= 0.15
        and next_if_benign is not None and next_if_benign > 0
        and p_value is not None and p_value < 0.10
        and 0.2 <= base_benign <= 0.8
    )

    return {
        "coin": coin, "type": "spot" if tox.is_spot(coin) else "perp",
        "bin_min": bin_ms // 60000, "n_bins_with_flow": len(recs),
        "span_bins": span_bins, "flow_bin_frac": round(len(recs) / span_bins, 3),
        "median_trades_per_bin": ns[len(ns) // 2], "median_notional_per_bin": round(vols[len(vols) // 2]),
        "median_gap_bins": gaps[len(gaps) // 2],
        "uncond_maker_net": round(uncond, 3), "base_benign_share": round(base_benign, 3),
        "acf1": acf1, "half_life_bins": half_life,
        "acf": {str(k): v["acf"] for k, v in acf.items()},
        "regime": {"p_benign_given_benign": p_bb, "p_toxic_given_toxic": p_tt, "n_adjacent_pairs": n_adj},
        "gate": {"next_net_if_prev_benign": next_if_benign, "n_prev_benign": n_b,
                 "next_net_if_prev_toxic": next_if_toxic, "n_prev_toxic": n_t,
                 "lift_benign_minus_toxic": round(lift, 3) if lift is not None else None,
                 "placebo_p": p_value},
        "hour_of_day_maker_net": hod_profile,
        "gate_viable": gate_viable,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--coins", default="SPX,DYDX,XPL,FET,PUMP,LINK,PURR/USDC")
    ap.add_argument("--days", default="")
    ap.add_argument("--bin-min", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.days:
        days = args.days.split(",")
    else:
        days = sorted({os.path.basename(os.path.dirname(p)).replace("dt=", "")
                       for p in glob.glob(os.path.join(args.root, "bbo", "dt=*", "coin=*"))})
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    bin_ms = args.bin_min * 60_000
    results = [analyze(args.root, c, days, bin_ms) for c in coins]

    if args.json:
        print(json.dumps({"days": days, "bin_min": args.bin_min, "results": results}, indent=2))
        return

    print(f"# intraday toxic-state persistence — days={days} bin={args.bin_min}min "
          f"(state=maker_net@60s per bin)")
    print(f"{'coin':11s} {'nbin':>4s} {'uNet':>6s} {'benB%':>6s} {'ACF1':>5s} {'hlfLf':>5s} "
          f"{'nxt|B':>6s} {'nxt|T':>6s} {'lift':>6s} {'plcbo':>5s} {'gate':>4s}")
    for r in results:
        if r.get("error"):
            print(f"{r['coin']:11s} ERROR {r['error']}"); continue
        g = r["gate"]
        print(f"{r['coin']:11s} {r['n_bins_with_flow']:4d} {r['uncond_maker_net']:6.2f} "
              f"{r['base_benign_share']:6.2f} {str(r['acf1']):>5s} {str(r['half_life_bins']):>5s} "
              f"{str(g['next_net_if_prev_benign']):>6s} {str(g['next_net_if_prev_toxic']):>6s} "
              f"{str(g['lift_benign_minus_toxic']):>6s} {str(g['placebo_p']):>5s} "
              f"{'YES' if r['gate_viable'] else 'no':>4s}")
    print("\nuNet=uncond vol-wtd maker_net@60s (bps). benB%=share of bins benign. "
          "ACF1=lag-1 autocorr of the state (>0 = persists to next bin).")
    print("nxt|B / nxt|T = next-bin maker_net given prev bin benign / toxic. "
          "lift=nxt|B−nxt|T. plcbo=permutation p (want <0.10).")
    print("gate_viable = ACF1>=0.15 AND nxt|B>0 AND plcbo<0.10 AND benign-share in [0.2,0.8] "
          "(r85: state must persist past detection lag).")


if __name__ == "__main__":
    main()
