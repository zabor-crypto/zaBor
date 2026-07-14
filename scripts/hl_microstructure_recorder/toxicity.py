#!/usr/bin/env python3
"""
Taker-toxicity classifier for HL MM research (HL market-making research).

Answers the binding G1 question for a coin: *against the flow that crosses our
spread, does a maker actually make money?* Uses HL's wallet-tagged trade tape
(each trade carries both counterparty addresses) to measure adverse selection
directly, per aggression and per taker wallet.

EMPIRICALLY VERIFIED SEMANTICS (PURR/USDC, 2026-07-04):
  * side "A" => taker SOLD (prints below mid); side "B" => taker BOUGHT (above).
    aggression sign s = +1 for "B", -1 for "A".
  * users = [BUYER, SELLER] (feed-level ordering, NOT [maker,taker]). Proven on
    PURR's clean one-sided MM: it is users[0] on 100% of side-A trades (taker
    sells => maker is the buyer => users[0]) and users[0] on 0% of side-B trades
    (taker buys => maker is the seller => users[1]). Therefore the TAKER is
    users[0] when side "B" (buy) and users[1] when side "A" (sell) — a side-
    dependent index, not a fixed slot. A per-run diagnostic reports the busiest
    address's by-side u0-rate so the ordering can be re-confirmed on new data.
  * Aggression-side markout depends ONLY on side+price (both verified) and is
    independent of the users ordering; wallet attribution uses the rule above.

METRICS (size-weighted, per coin):
  taker_markout(dt)  = s * (mid(t+dt) - fill_px) / fill_px            [>0 = informed taker = bad for maker]
  half_spread        = (ask - bid) / 2 / mid at trade time            [what the maker earns upfront]
  maker_gross(dt)    = half_spread - taker_markout                    [maker edge before fees]
  maker_net(dt)      = maker_gross - maker_fee                        [THE G1 number]
  non_toxic_share    = taker-volume fraction with maker_net(dt) >= 0  [G1 pass wants a real, profitable segment]

Horizons: +60s and +300s. Fees: spot maker 4bps, perp maker 1.5bps (fees.json).
Tolerates truncated live gz (EOFError). No orders, no capital.

Usage:
  toxicity.py --root ./data --coins PURR/USDC,@107,SOL --days 2026-07-04[,2026-07-05]
  (default root=./data, all days present, a sensible coin set)
"""
import argparse, bisect, glob, gzip, json, os, sys
from collections import defaultdict

SPOT_MAKER_BPS = 4.0
PERP_MAKER_BPS = 1.5
H1_MS, H2_MS = 60_000, 300_000


def is_spot(coin):
    return coin.startswith("@") or "/" in coin


def maker_fee_bps(coin):
    return SPOT_MAKER_BPS if is_spot(coin) else PERP_MAKER_BPS


def coin_dir(coin):
    return coin.replace("/", "_").replace("@", "at")


def read_gz_lines(paths):
    for p in sorted(paths):
        try:
            with gzip.open(p, "rt") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (EOFError, OSError):
            continue  # truncated live file: keep whatever decoded


def load_mid_series(root, coin, days):
    """Sorted (times[], mids[], halfspr_bps[]) from bbo for the coin over days."""
    t, mid, hs = [], [], []
    for day in days:
        pat = os.path.join(root, "bbo", f"dt={day}", f"coin={coin_dir(coin)}", "*.jsonl.gz")
        for r in read_gz_lines(glob.glob(pat)):
            d = r.get("d") or {}
            b = d.get("bbo")
            if not (b and len(b) == 2 and b[0] and b[1]):
                continue
            bid = float(b[0]["px"]); ask = float(b[1]["px"])
            if ask <= 0 or bid <= 0 or ask < bid:
                continue
            m = (bid + ask) / 2
            t.append(d["time"]); mid.append(m); hs.append((ask - bid) / 2 / m * 1e4)
    order = sorted(range(len(t)), key=lambda i: t[i])
    return [t[i] for i in order], [mid[i] for i in order], [hs[i] for i in order]


def make_lookup(times):
    def at(ts):
        i = bisect.bisect_right(times, ts) - 1
        return i if i >= 0 else None
    return at


def stream_fills(times, mids, halfs, root, coin, days):
    """Yield one dict per parseable trade with the VERIFIED attribution + markout —
    the single shared source of truth (analyze_coin AND the persistence harness use it).
      side "A"=taker sell / "B"=taker buy;  users=[buyer,seller] => taker = u0 on "B", u1 on "A";
      mk[h] = s*(mid(t+h) - fill_px)/fill_px * 1e4  (bps vs fill; spread embedded; None if unavailable).
    Fields: t, s, px, sz, notional, u0, u1, taker, half_spread (None if no pre-mid), mk{H1,H2}."""
    at = make_lookup(times)
    t_end = times[-1] if times else 0
    for day in days:
        pat = os.path.join(root, "trades", f"dt={day}", f"coin={coin_dir(coin)}", "*.jsonl.gz")
        for r in read_gz_lines(glob.glob(pat)):
            for tr in (r.get("d") or []):
                try:
                    px = float(tr["px"]); sz = float(tr["sz"]); t = tr["time"]
                    s = 1.0 if tr["side"] == "B" else -1.0
                    users = tr.get("users") or [None, None]
                    u0 = users[0] if len(users) > 0 else None
                    u1 = users[1] if len(users) > 1 else None
                    taker = u0 if s > 0 else u1
                except (KeyError, ValueError, TypeError):
                    continue
                i0 = at(t - 1)
                hspr = halfs[i0] if i0 is not None else None
                mk = {}
                for h in (H1_MS, H2_MS):
                    if i0 is None or t + h > t_end:
                        mk[h] = None; continue
                    j = at(t + h)
                    mk[h] = s * (mids[j] - px) / px * 1e4 if j is not None else None
                yield {"t": t, "s": s, "px": px, "sz": sz, "notional": px * sz,
                       "u0": u0, "u1": u1, "taker": taker, "half_spread": hspr, "mk": mk}


def analyze_coin(root, coin, days):
    times, mids, halfs = load_mid_series(root, coin, days)
    if len(times) < 50:
        return {"coin": coin, "error": f"insufficient bbo ({len(times)} pts)"}
    fee = maker_fee_bps(coin)
    agg = {h: {"vol": 0.0, "mk_w": 0.0, "gross_w": 0.0, "net_w": 0.0, "ntx_vol": 0.0, "mk_list": []}
           for h in (H1_MS, H2_MS)}
    maker_vol = 0.0
    wallet = defaultdict(lambda: {"vol": 0.0, "loss_w": 0.0, "n": 0})  # taker wallet -> adverse selection @H1
    # diagnostic to re-confirm users=[buyer,seller]: busiest one-sided MM shows u0-rate
    # high on side A (maker=buyer) and low on side B (maker=seller).
    freq = defaultdict(int)
    byside = defaultdict(lambda: [0, 0, 0, 0])  # addr -> [u0_on_A, tot_A, u0_on_B, tot_B]
    n_tr = n_used = 0
    hs_used = []

    for f in stream_fills(times, mids, halfs, root, coin, days):
        n_tr += 1
        u0, u1, s = f["u0"], f["u1"], f["s"]
        for a in (u0, u1):
            if a:
                freq[a] += 1
        if u0:
            bs = byside[u0]
            if s > 0: bs[2] += 1; bs[3] += 1
            else: bs[0] += 1; bs[1] += 1
        if u1:
            bs = byside[u1]
            if s > 0: bs[3] += 1
            else: bs[1] += 1
        if f["half_spread"] is None:
            continue
        notional = f["notional"]; taker = f["taker"]
        got_any = False
        for h in (H1_MS, H2_MS):
            tk_mk = f["mk"][h]
            if tk_mk is None:
                continue
            gross = -tk_mk; net = gross - fee
            a = agg[h]
            a["vol"] += notional
            a["mk_w"] += tk_mk * notional
            a["gross_w"] += gross * notional
            a["net_w"] += net * notional
            if net >= 0:
                a["ntx_vol"] += notional
            a["mk_list"].append((tk_mk, notional))
            got_any = True
            if h == H1_MS and taker:
                w = wallet[taker]
                w["vol"] += notional; w["n"] += 1
                w["loss_w"] += tk_mk * notional
        if got_any:
            n_used += 1; maker_vol += notional; hs_used.append(f["half_spread"])

    def wmean(num, den):
        return num / den if den else None

    def wmedian(pairs):
        if not pairs:
            return None
        pairs = sorted(pairs, key=lambda x: x[0])
        tot = sum(w for _, w in pairs); acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= tot / 2:
                return v
        return pairs[-1][0]

    # diagnostic: re-confirm users=[buyer,seller] via the busiest address's by-side u0-rate.
    # A one-sided MM (maker) shows u0-rate high on side A (maker=buyer) and low on side B
    # (maker=seller). "inconclusive" when no single address is one-sided (typical on majors).
    diag = {"busiest_addr": None, "u0rate_sideA": None, "u0rate_sideB": None, "buyer_seller": "inconclusive"}
    if freq:
        top = max(freq, key=freq.get)
        bs = byside[top]
        rA = bs[0] / bs[1] if bs[1] else None
        rB = bs[2] / bs[3] if bs[3] else None
        diag["busiest_addr"] = top[:12] + ".."
        diag["u0rate_sideA"] = round(rA, 3) if rA is not None else None
        diag["u0rate_sideB"] = round(rB, 3) if rB is not None else None
        if rA is not None and rB is not None and abs(rA - rB) >= 0.5:
            # clean one-sided address: high-A/low-B (maker) or low-A/high-B (taker) both confirm [buyer,seller]
            diag["buyer_seller"] = "confirmed"

    out = {
        "coin": coin, "type": "spot" if is_spot(coin) else "perp",
        "maker_fee_bps": fee, "n_trades": n_tr, "n_used": n_used,
        "taker_volume_usd": round(maker_vol, 2),
        "half_spread_bps_median": round(sorted(hs_used)[len(hs_used) // 2], 3) if hs_used else None,
        "users_order_diag": diag,
        "horizons": {},
    }
    for h in (H1_MS, H2_MS):
        a = agg[h]
        tk_mean = wmean(a["mk_w"], a["vol"])
        tk_med = wmedian(a["mk_list"])
        out["horizons"][f"{h//1000}s"] = {
            "taker_markout_bps_wmean": round(tk_mean, 3) if tk_mean is not None else None,
            "taker_markout_bps_wmedian": round(tk_med, 3) if tk_med is not None else None,
            "maker_gross_bps_wmean": round(wmean(a["gross_w"], a["vol"]), 3) if a["vol"] else None,
            "maker_net_bps_wmean": round(wmean(a["net_w"], a["vol"]), 3) if a["vol"] else None,
            "non_toxic_vol_share": round(a["ntx_vol"] / a["vol"], 3) if a["vol"] else None,
        }

    # toxic taker wallets by adverse-selection $ imposed on makers (@60s); needs a floor of activity
    tox = sorted((kv for kv in wallet.items() if kv[1]["n"] >= 5 and kv[1]["vol"] > 0),
                 key=lambda kv: -kv[1]["loss_w"])[:8]
    out["top_toxic_takers_60s"] = [
        {"wallet": w[:12] + "..", "vol_usd": round(d["vol"], 1), "n": d["n"],
         "mean_taker_markout_bps": round(d["loss_w"] / d["vol"], 2)}
        for w, d in tox if d["loss_w"] > 0]
    out["n_taker_wallets"] = len(wallet)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--coins", default="PURR/USDC,@107,@301,@198,@1,SOL,LINK,HYPE")
    ap.add_argument("--days", default="")
    ap.add_argument("--json", action="store_true", help="emit full JSON")
    args = ap.parse_args()

    if args.days:
        days = args.days.split(",")
    else:
        days = sorted({os.path.basename(os.path.dirname(p)).replace("dt=", "")
                       for p in glob.glob(os.path.join(args.root, "bbo", "dt=*", "coin=*"))})
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    results = [analyze_coin(args.root, c, days) for c in coins]

    if args.json:
        print(json.dumps({"days": days, "results": results}, indent=2))
        return

    print(f"# taker-toxicity — days={days}")
    print(f"{'coin':12s} {'type':4s} {'n_use':>6s} {'vol$':>10s} {'hspr':>6s} "
          f"{'tkMk60':>7s} {'mkNet60':>8s} {'ntx60':>6s} {'mkNet300':>9s} {'ntx300':>7s} chk")
    for r in results:
        if r.get("error"):
            print(f"{r['coin']:12s} ERROR {r['error']}"); continue
        h1 = r["horizons"]["60s"]; h2 = r["horizons"]["300s"]
        bs = r["users_order_diag"]["buyer_seller"]
        chk = {"confirmed": "bs✓", "inconclusive": "bs?"}.get(bs, bs)
        print(f"{r['coin']:12s} {r['type']:4s} {r['n_used']:6d} {r['taker_volume_usd']:10.0f} "
              f"{str(r['half_spread_bps_median']):>6s} {str(h1['taker_markout_bps_wmean']):>7s} "
              f"{str(h1['maker_net_bps_wmean']):>8s} {str(h1['non_toxic_vol_share']):>6s} "
              f"{str(h2['maker_net_bps_wmean']):>9s} {str(h2['non_toxic_vol_share']):>7s} {chk}")
    print("\nmkNet>0 = maker profitable net of fee against this flow; ntx = non-toxic taker vol share.")
    print("bs✓ = users=[buyer,seller] re-confirmed on this coin; bs? = inconclusive (no one-sided MM, majors).")


if __name__ == "__main__":
    main()
