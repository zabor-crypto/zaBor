#!/usr/bin/env python3
"""First-look microstructure snapshot from the fresh HL capture (~15 min).
Per coin: median/p25/p75 spread (bps), mid volatility (bps/min), trade
arrival rate, avg trade size, top-of-book depth ($), spread/(2*fee) ratio.
Output: JSON to stdout."""
import gzip, json, glob, math, statistics as st, sys, os

ROOT = sys.argv[1]
MAKER_FEE_BPS = 1.0   # HL base-tier perp maker assumption; VERIFY via userFees
TAKER_FEE_BPS = 3.5

def load(stream, coin):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{stream}/dt=*/coin={coin}/*.jsonl.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try: out.append(json.loads(line))
                    except Exception: pass
        except EOFError:
            pass  # live file still being written; keep what decoded
    return out

coins = sorted({os.path.basename(p).split("=",1)[1] for p in glob.glob(f"{ROOT}/bbo/dt=*/coin=*")})
res = {}
for coin in coins:
    bbo = load("bbo", coin)
    trades = load("trades", coin)
    spreads, mids, depths, ts = [], [], [], []
    for r in bbo:
        d = r["d"].get("bbo")
        if not d or not d[0] or not d[1]: continue
        b, a = float(d[0]["px"]), float(d[1]["px"])
        if a <= b: continue
        mid = (a+b)/2
        spreads.append((a-b)/mid*1e4)
        mids.append(mid); ts.append(r["d"].get("time", r["recv_ts_ms"]))
        depths.append((float(d[0]["sz"])*b + float(d[1]["sz"])*a)/2)
    ntr, vol_usd, sizes = 0, 0.0, []
    t_first = t_last = None
    for r in trades:
        for t in r["d"]:
            px, sz = float(t["px"]), float(t["sz"])
            ntr += 1; vol_usd += px*sz; sizes.append(px*sz)
            tm = t.get("time")
            if tm:
                t_first = tm if t_first is None else min(t_first, tm)
                t_last = tm if t_last is None else max(t_last, tm)
    # 10s-sampled mid returns -> bps/min vol
    vol_bpm = None
    if len(mids) > 10 and ts:
        samp, last_t = [], 0
        for m, t in zip(mids, ts):
            if t - last_t >= 10_000:
                samp.append(m); last_t = t
        rets = [ (samp[i+1]/samp[i]-1)*1e4 for i in range(len(samp)-1) ]
        if len(rets) > 3:
            vol_bpm = st.pstdev(rets) * math.sqrt(6)  # 10s -> 1min
    dur_min = (t_last - t_first)/60000 if (t_first and t_last and t_last > t_first) else None
    med_spread = st.median(spreads) if spreads else None
    res[coin] = {
        "n_bbo": len(bbo),
        "spread_bps": {"p25": round(st.quantiles(spreads, n=4)[0],2) if len(spreads)>4 else None,
                        "med": round(med_spread,2) if med_spread else None,
                        "p75": round(st.quantiles(spreads, n=4)[2],2) if len(spreads)>4 else None},
        "top_depth_usd_med": round(st.median(depths)) if depths else None,
        "vol_bps_per_min": round(vol_bpm,2) if vol_bpm else None,
        "trades_per_min": round(ntr/dur_min,2) if dur_min else None,
        "avg_trade_usd": round(vol_usd/ntr) if ntr else None,
        "spread_over_2maker": round(med_spread/(2*MAKER_FEE_BPS),2) if med_spread else None,
        "spread_minus_2maker_bps": round(med_spread-2*MAKER_FEE_BPS,2) if med_spread else None,
    }
print(json.dumps(res, indent=1))
