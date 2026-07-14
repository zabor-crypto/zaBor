#!/usr/bin/env python3
"""
Tier-1 DEPTH-markout scan for HL MM research (HL market-making research —
"does quoting DEEPER than touch rescue the maker?" question) + the λ(δ) fill-
intensity fit that Avellaneda-Stoikov needs to pick the optimal depth.

WHY: every at-touch branch is dead (07-06: S1 fee wall, S4 toxic, S2 no stable/
gate-able edge). The one lever left is resting δ bps BEHIND the touch: you earn
more spread upfront (δ instead of half_spread) but fill rarer and only when the
market sweeps to you — which could be less toxic (noise sweeps) or MORE toxic
(informed sweeps). This scan answers it directly from the existing tape.

FILL MODEL (trade-through, tape-only — an UPPER BOUND; Phase-2 sim tightens it):
  maker rests a bid at B = mid_t·(1−δ/1e4) and an ask at A = mid_t·(1+δ/1e4).
    * a taker-SELL print (side "A") at px ≤ B fills the bid → maker LONG at B,
      pnl(h) = (mid(t+h) − B)/B·1e4.
    * a taker-BUY  print (side "B") at px ≥ A fills the ask → maker SHORT at A,
      pnl(h) = (A − mid(t+h))/A·1e4.
    * maker_net(h) = pnl(h) − maker_fee.  (δ is EARNED inside pnl via the fill px;
      at δ≈median half_spread this reproduces the at-touch toxicity number.)
  Reader locates "touch" by comparing δ to the coin's median half_spread (printed).
  Caveats (flagged UPPER BOUND): no queue position (we assume our full size fills
  when a sweep reaches our level); a single multi-print sweep can double-count
  fills at deep δ (inflates capacity, not per-fill markout); exogenous tape (our
  passive deep quote does not move the book — defensible on liquid perps, NOT at
  PURR touch). Only the Phase-2 sim + shadow upgrade the evidence class.

λ(δ) FIT (plan task 1.2): arrival intensity that reaches distance δ =
  N(≥δ)/T fills·s⁻¹, pooled bid+ask. Fit ln λ(δ) = ln A − k·δ (OLS on the δ grid,
  points with N>0). A-S optimal half-spread ∝ 1/k + inventory/vol terms, so k is
  the number that says whether deeper quoting even has room. R² reports fit quality.

OUTPUT per coin: δ-scan (maker_net@60/300s, fills/day, $/day per δ), the best δ
(max maker_net@60s with a capacity floor) and whether it clears 0, and (A,k,R²).

No orders, no capital. Import-safe. Tolerates truncated live gz.

Usage:
  toxicity_depth.py --root ./data --coins SPX,DYDX,PURR/USDC \
      --days 2026-07-03,2026-07-04,2026-07-05 --deltas 1,2,5,10,20,40 --json
"""
import argparse, glob, json, math, os

import toxicity as tox  # reuse verified load_mid_series / make_lookup / read_gz_lines / fees

H1_MS, H2_MS = tox.H1_MS, tox.H2_MS
MIN_FILLS_PER_DAY = 5.0   # capacity floor for the "best profitable δ" call


def scan_coin(root, coin, days, deltas):
    times, mids, halfs = tox.load_mid_series(root, coin, days)
    if len(times) < 50:
        return {"coin": coin, "error": f"insufficient bbo ({len(times)} pts)"}
    fee = tox.maker_fee_bps(coin)
    at = tox.make_lookup(times)
    t_end = times[-1]
    dur_days = (times[-1] - times[0]) / 86_400_000 or None
    dur_s = (times[-1] - times[0]) / 1000 or None

    # per-delta accumulators (bid+ask pooled for economics; count for λ)
    acc = {d: {"vol": 0.0, "net60_w": 0.0, "net300_w": 0.0, "n": 0} for d in deltas}
    hs_used = []

    for day in days:
        pat = os.path.join(root, "trades", f"dt={day}", f"coin={tox.coin_dir(coin)}", "*.jsonl.gz")
        for r in tox.read_gz_lines(glob.glob(pat)):
            for tr in (r.get("d") or []):
                try:
                    px = float(tr["px"]); sz = float(tr["sz"]); t = tr["time"]
                    side = tr["side"]
                except (KeyError, ValueError, TypeError):
                    continue
                i0 = at(t - 1)
                if i0 is None:
                    continue
                mid_t = mids[i0]
                if mid_t <= 0:
                    continue
                notional = px * sz
                # markout mids
                j1 = at(t + H1_MS) if t + H1_MS <= t_end else None
                j2 = at(t + H2_MS) if t + H2_MS <= t_end else None
                mid1 = mids[j1] if j1 is not None else None
                mid2 = mids[j2] if j2 is not None else None
                if mid1 is None:  # need at least the 60s horizon to score
                    continue
                hs_used.append(halfs[i0])
                for d in deltas:
                    if side == "A":                    # taker sell → could fill my bid
                        B = mid_t * (1 - d / 1e4)
                        if px > B:
                            continue                   # sweep didn't reach my bid
                        pnl60 = (mid1 - B) / B * 1e4
                        pnl300 = (mid2 - B) / B * 1e4 if mid2 is not None else None
                    elif side == "B":                  # taker buy → could fill my ask
                        A = mid_t * (1 + d / 1e4)
                        if px < A:
                            continue
                        pnl60 = (A - mid1) / A * 1e4
                        pnl300 = (A - mid2) / A * 1e4 if mid2 is not None else None
                    else:
                        continue
                    a = acc[d]
                    a["vol"] += notional; a["n"] += 1
                    a["net60_w"] += (pnl60 - fee) * notional
                    if pnl300 is not None:
                        a["net300_w"] += (pnl300 - fee) * notional
                    else:
                        a["net300_w"] += (pnl60 - fee) * notional  # fallback so weights align at tape tail

    rows = []
    lam_pts = []  # (delta, lambda_per_s) for the fit
    for d in deltas:
        a = acc[d]
        if a["vol"] <= 0:
            rows.append({"delta_bps": d, "fills": 0, "maker_net60": None, "maker_net300": None,
                         "fills_per_day": 0.0, "usd_per_day": 0.0})
            continue
        net60 = a["net60_w"] / a["vol"]; net300 = a["net300_w"] / a["vol"]
        fpd = a["n"] / dur_days if dur_days else None
        upd = a["vol"] / dur_days if dur_days else None
        rows.append({"delta_bps": d, "fills": a["n"], "maker_net60": round(net60, 3),
                     "maker_net300": round(net300, 3),
                     "fills_per_day": round(fpd, 1) if fpd is not None else None,
                     "usd_per_day": round(upd) if upd is not None else None})
        if dur_s:
            lam_pts.append((d, a["n"] / dur_s))

    # best profitable δ with capacity
    prof = [r for r in rows if r["maker_net60"] is not None and r["maker_net60"] > 0
            and (r["fills_per_day"] or 0) >= MIN_FILLS_PER_DAY]
    best = max(prof, key=lambda r: r["maker_net60"]) if prof else None
    # best by maker_net regardless of sign (for the curve shape)
    scored = [r for r in rows if r["maker_net60"] is not None]
    best_any = max(scored, key=lambda r: r["maker_net60"]) if scored else None

    # CREDIBILITY: a real A-S optimum is INTERIOR (concave peak), not a runaway at the
    # deepest grid point on a handful of fills. Require: best δ has a DEEPER δ that was
    # actually filled (capacity) and scores LOWER (curve turns over), and a decent total n.
    credible = None
    if best is not None:
        di = deltas.index(best["delta_bps"])
        deeper = [rows[k] for k in range(di + 1, len(deltas))
                  if rows[k]["maker_net60"] is not None and (rows[k]["fills_per_day"] or 0) >= 1]
        turns_over = bool(deeper) and all(r["maker_net60"] < best["maker_net60"] for r in deeper)
        is_deepest = (di == len(deltas) - 1) or not deeper  # no filled δ beyond it → can't confirm a peak
        enough_n = best["fills"] >= 30                       # ≥~10/day over 3d, not a tail handful
        credible = {**best, "is_interior_peak": turns_over, "is_deepest_or_unconfirmed": is_deepest,
                    "n_fills_total": best["fills"], "enough_n": enough_n,
                    "credible": bool(turns_over and enough_n)}

    # λ(δ) = A exp(-k δ) via OLS on ln λ
    lam_fit = {"A_per_s": None, "k_per_bp": None, "r2": None, "n_pts": len(lam_pts)}
    pts = [(d, l) for d, l in lam_pts if l > 0]
    if len(pts) >= 3:
        xs = [d for d, _ in pts]; ys = [math.log(l) for _, l in pts]
        n = len(xs); sx = sum(xs); sy = sum(ys)
        sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if denom != 0:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            yhat = [intercept + slope * x for x in xs]
            ybar = sy / n
            ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
            ss_tot = sum((y - ybar) ** 2 for y in ys) or 1e-12
            lam_fit = {"A_per_s": round(math.exp(intercept), 5), "k_per_bp": round(-slope, 4),
                       "r2": round(1 - ss_res / ss_tot, 3), "n_pts": n}

    return {
        "coin": coin, "type": "spot" if tox.is_spot(coin) else "perp", "maker_fee_bps": fee,
        "half_spread_bps_median": round(sorted(hs_used)[len(hs_used) // 2], 3) if hs_used else None,
        "span_days": round(dur_days, 2) if dur_days else None,
        "scan": rows,
        "best_profitable_delta": best,          # None ⇒ no δ clears 0 with capacity
        "credible_depth_edge": credible,        # interior-peak + enough-n gated (the real go/no-go)
        "best_maker_net_delta": best_any,       # peak of the curve regardless of sign
        "lambda_fit": lam_fit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--coins", default="SPX,DYDX,XPL,FET,PUMP,LINK,DOGE,ZEC,SOL,HYPE,PURR/USDC")
    ap.add_argument("--days", default="")
    ap.add_argument("--deltas", default="1,2,5,10,20,40", help="bps-from-mid grid")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.days:
        days = args.days.split(",")
    else:
        days = sorted({os.path.basename(os.path.dirname(p)).replace("dt=", "")
                       for p in glob.glob(os.path.join(args.root, "bbo", "dt=*", "coin=*"))})
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    results = [scan_coin(args.root, c, days, deltas) for c in coins]

    if args.json:
        print(json.dumps({"days": days, "deltas": deltas, "results": results}, indent=2))
        return

    print(f"# depth-markout scan — days={days} deltas(bps-from-mid)={deltas} "
          f"(maker_net@60s per δ; UPPER BOUND: no queue, exogenous tape)")
    print(f"{'coin':11s} {'hspr':>5s} " + " ".join(f"d{int(d):<2d}".rjust(7) for d in deltas)
          + "   bestδ  net@best  f/day  λk")
    for r in results:
        if r.get("error"):
            print(f"{r['coin']:11s} ERROR {r['error']}"); continue
        cells = []
        for row in r["scan"]:
            v = row["maker_net60"]
            cells.append((f"{v:+.1f}" if v is not None else "-").rjust(7))
        cr = r["credible_depth_edge"]
        if cr and cr["credible"]:
            bstr = f"{cr['delta_bps']:g}bp {cr['maker_net60']:+.1f} n{cr['n_fills_total']} CREDIBLE"
        elif r["best_profitable_delta"]:
            bp = r["best_profitable_delta"]
            tag = "tail?" if (not cr or cr["is_deepest_or_unconfirmed"]) else "thin-n"
            bstr = f"{bp['delta_bps']:g}bp {bp['maker_net60']:+.1f} n{bp['fills']} ({tag})"
        else:
            bstr = "none"
        k = r["lambda_fit"]["k_per_bp"]
        print(f"{r['coin']:11s} {str(r['half_spread_bps_median']):>5s} " + " ".join(cells)
              + f"   {bstr}  {('k='+str(k)) if k is not None else ''}")
    print("\nEach cell = vol-wtd maker_net@60s (bps) resting at that δ from mid. "
          "'touch' ≈ the δ nearest the coin's hspr.")
    print("best call = δ with maker_net>0 & ≥5 fills/day. CREDIBLE = INTERIOR peak (curve turns over "
          "at a deeper filled δ) AND ≥30 total fills; (tail?) = only the deepest δ / unconfirmed peak "
          "= likely small-sample artifact; 'none' = depth does not rescue this coin.")
    print("λk = decay of fill-intensity λ(δ)=A·exp(−kδ) (A-S input; bigger k ⇒ fills die fast with depth ⇒ less room).")


if __name__ == "__main__":
    main()
