#!/usr/bin/env python3
"""
Taker-wallet toxicity PERSISTENCE harness (HL market-making research).

wallet-toxicity is in-sample-fragile — a toxic-avoidance quoting
rule is only viable if wallets flagged toxic on a TRAIN window stay toxic on the
next (TEST) window. This trains per-wallet taker markout on --train-days, evaluates
on --test-days, and answers:

  1. STAY-TOXIC (precision): of train-toxic wallets that reappear in test, what
     fraction are still toxic? Churn = fraction of train-toxic wallets that vanish.
  2. RANK STABILITY: Spearman corr of per-wallet markout between train and test
     (wallets active/eligible in both).
  3. AVOIDANCE VALUE (the decision metric): test-window maker_net@60s WITH vs
     WITHOUT filling train-toxic wallets. If refusing train-flagged wallets lifts
     net materially (ideally >= 0), a toxic-avoidance rule is worth building.
     Reported per coin and pooled across the coin set (wallets pooled venue-wide).

Reuses the VERIFIED attribution from toxicity.py (stream_fills): side A=sell/B=buy,
users=[buyer,seller] side-dependent taker, markout with spread embedded. Horizon 60s.
No orders, no capital.

Usage:
  toxicity_persistence.py --coins SPX,DYDX,XPL,LINK,FET --train-days 2026-07-03 --test-days 2026-07-04
  (with only ~2 days now this is a low-power smoke test; real power at G1 with 7 days —
   use e.g. --train-days d1,d2,d3,d4,d5 --test-days d6,d7)
"""
import argparse, glob, os
from collections import defaultdict

from toxicity import (load_mid_series, stream_fills, maker_fee_bps, H1_MS)

MIN_BBO = 50


def wallet_agg_train(root, coin, days, horizon=H1_MS):
    """taker -> [vol, mk_w, n] over days (train window)."""
    times, mids, halfs = load_mid_series(root, coin, days)
    W = defaultdict(lambda: [0.0, 0.0, 0])
    if len(times) < MIN_BBO:
        return W, len(times)
    for f in stream_fills(times, mids, halfs, root, coin, days):
        tk = f["taker"]; mk = f["mk"][horizon]
        if tk is None or mk is None or f["half_spread"] is None:
            continue
        w = W[tk]; w[0] += f["notional"]; w[1] += mk * f["notional"]; w[2] += 1
    return W, len(times)


def test_pass(root, coin, days, toxic_set, horizon=H1_MS):
    """One stream over the test window: builds We plus avoidance-value accumulators
    given a toxic_set to exclude. Returns (We, allv, exclv, toxv) where each v=[vol, net_w]
    (net_w uses per-coin maker fee); toxv accumulates [vol, mk_w] of the excluded flow."""
    fee = maker_fee_bps(coin)
    times, mids, halfs = load_mid_series(root, coin, days)
    We = defaultdict(lambda: [0.0, 0.0, 0])
    allv = [0.0, 0.0]; exclv = [0.0, 0.0]; toxv = [0.0, 0.0]
    if len(times) < MIN_BBO:
        return We, allv, exclv, toxv, len(times)
    for f in stream_fills(times, mids, halfs, root, coin, days):
        tk = f["taker"]; mk = f["mk"][horizon]
        if tk is None or mk is None or f["half_spread"] is None:
            continue
        notional = f["notional"]; net = -mk - fee
        w = We[tk]; w[0] += notional; w[1] += mk * notional; w[2] += 1
        allv[0] += notional; allv[1] += net * notional
        if tk in toxic_set:
            toxv[0] += notional; toxv[1] += mk * notional
        else:
            exclv[0] += notional; exclv[1] += net * notional
    return We, allv, exclv, toxv, len(times)


def eligible(W, min_fills, min_vol):
    """taker -> mean markout bps, for wallets with enough activity."""
    return {a: v[1] / v[0] for a, v in W.items() if v[2] >= min_fills and v[0] >= min_vol and v[0] > 0}


def spearman(pairs):
    n = len(pairs)
    if n < 3:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals); i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n; my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else None


def metrics(Wt, We, allv, exclv, toxv, tox_thresh, min_fills, min_vol):
    et = eligible(Wt, min_fills, min_vol)
    ee = eligible(We, min_fills, min_vol)
    toxic_train = {a for a, m in et.items() if m >= tox_thresh}
    reappear = [a for a in toxic_train if a in ee]
    stay = [a for a in reappear if ee[a] >= tox_thresh]
    both = [(et[a], ee[a]) for a in et if a in ee]
    base_net = allv[1] / allv[0] if allv[0] else None
    avoid_net = exclv[1] / exclv[0] if exclv[0] else None
    return {
        "n_eligible_train": len(et), "n_toxic_train": len(toxic_train),
        "reappear": len(reappear), "churn": round(1 - len(reappear) / len(toxic_train), 3) if toxic_train else None,
        "stay_toxic": len(stay),
        "stay_precision": round(len(stay) / len(reappear), 3) if reappear else None,
        "spearman_both": round(spearman(both), 3) if spearman(both) is not None else None,
        "n_both": len(both),
        "base_net_bps": round(base_net, 3) if base_net is not None else None,
        "avoid_net_bps": round(avoid_net, 3) if avoid_net is not None else None,
        "net_lift_bps": round(avoid_net - base_net, 3) if (base_net is not None and avoid_net is not None) else None,
        "toxic_share_test_vol": round(toxv[0] / allv[0], 3) if allv[0] else None,
        "toxic_test_mean_mk_bps": round(toxv[1] / toxv[0], 2) if toxv[0] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--coins", default="SPX,DYDX,XPL,LINK,FET,SOL,HYPE")
    ap.add_argument("--train-days", required=True, help="comma list, e.g. 2026-07-03")
    ap.add_argument("--test-days", required=True, help="comma list, e.g. 2026-07-04")
    ap.add_argument("--tox-thresh-bps", type=float, default=2.0, help="mean taker markout to flag a wallet toxic")
    ap.add_argument("--min-fills", type=int, default=5)
    ap.add_argument("--min-vol", type=float, default=1000.0)
    args = ap.parse_args()
    train = args.train_days.split(","); test = args.test_days.split(",")
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    P = dict(tox_thresh=args.tox_thresh_bps, min_fills=args.min_fills, min_vol=args.min_vol)

    Wt_pool = defaultdict(lambda: [0.0, 0.0, 0]); We_pool = defaultdict(lambda: [0.0, 0.0, 0])
    per = {}
    for coin in coins:
        Wt, nbt = wallet_agg_train(args.root, coin, train)
        et = eligible(Wt, args.min_fills, args.min_vol)
        tox_train = {a for a, m in et.items() if m >= args.tox_thresh_bps}
        We, allv, exclv, toxv, nbe = test_pass(args.root, coin, test, tox_train)
        if nbt < MIN_BBO or nbe < MIN_BBO:
            per[coin] = {"error": f"insufficient bbo (train {nbt}, test {nbe})"}
        else:
            per[coin] = metrics(Wt, We, allv, exclv, toxv, args.tox_thresh_bps, args.min_fills, args.min_vol)
        for a, v in Wt.items():
            p = Wt_pool[a]; p[0] += v[0]; p[1] += v[1]; p[2] += v[2]
        for a, v in We.items():
            p = We_pool[a]; p[0] += v[0]; p[1] += v[1]; p[2] += v[2]

    # pooled (wallets venue-wide): toxic set from pooled train, value re-streamed with pooled set
    et_pool = eligible(Wt_pool, args.min_fills, args.min_vol)
    tox_pool = {a for a, m in et_pool.items() if m >= args.tox_thresh_bps}
    allv = [0.0, 0.0]; exclv = [0.0, 0.0]; toxv = [0.0, 0.0]
    for coin in coins:
        _, a1, e1, t1, nbe = test_pass(args.root, coin, test, tox_pool)
        if nbe >= MIN_BBO:
            for dst, src in ((allv, a1), (exclv, e1), (toxv, t1)):
                dst[0] += src[0]; dst[1] += src[1]
    pooled = metrics(Wt_pool, We_pool, allv, exclv, toxv, args.tox_thresh_bps, args.min_fills, args.min_vol)

    print(f"# taker-wallet toxicity persistence — train={train} test={test}  "
          f"thr={args.tox_thresh_bps}bps min_fills={args.min_fills} min_vol=${args.min_vol:.0f}")
    hdr = (f"{'coin':10s} {'elig':>5s} {'tox':>4s} {'reapp':>5s} {'churn':>6s} {'stayP':>6s} "
           f"{'spear':>6s} {'baseNet':>8s} {'avoidNet':>9s} {'lift':>6s} {'toxSh':>6s} {'toxMk':>6s}")
    print(hdr)
    def row(name, m):
        if m.get("error"):
            print(f"{name:10s} {m['error']}"); return
        print(f"{name:10s} {m['n_eligible_train']:5d} {m['n_toxic_train']:4d} {m['reappear']:5d} "
              f"{str(m['churn']):>6s} {str(m['stay_precision']):>6s} {str(m['spearman_both']):>6s} "
              f"{str(m['base_net_bps']):>8s} {str(m['avoid_net_bps']):>9s} {str(m['net_lift_bps']):>6s} "
              f"{str(m['toxic_share_test_vol']):>6s} {str(m['toxic_test_mean_mk_bps']):>6s}")
    for coin in coins:
        row(coin, per[coin])
    print("-" * len(hdr))
    row("POOLED", pooled)
    print("\nstayP = of reappearing train-toxic wallets, share still toxic in test (want high).")
    print("lift = avoidNet - baseNet: net_bps gained by refusing train-flagged wallets (the decision metric).")
    print("spear = per-wallet markout rank stability train->test. Low eligible n => underpowered (need G1's 7 days).")


if __name__ == "__main__":
    main()
