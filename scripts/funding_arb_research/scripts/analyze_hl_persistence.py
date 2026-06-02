"""HL-vs-CEX funding divergence persistence analysis.

Question: do HL-vs-Binance/Bybit divergences last long enough to be
tradeable — specifically, do they persist longer than CEX-vs-CEX spreads
(which we know fail because carry mean-reverts in ~21h)?

Methodology:
  1. Load 540d funding archive for HL + Binance + Bybit.
  2. Normalize all rates to APR; align to hourly grid (forward-fill).
  3. For each coin × pair-type (HL-Binance, HL-Bybit, Binance-Bybit):
       a. Identify episodes where |spread| > THRESHOLD_APR.
       b. For each episode measure duration_h and accumulated carry.
  4. Compare distributions across pair types.

Break-even: $150K notional, HL(3.5bp)+CEX(5bp) taker fees → $255 round-trip.
Need episodes delivering >$255 carry to be profitable (ignoring slippage).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────
THRESHOLD_APRS = [0.20, 0.50, 1.00, 2.00]
HALF_EXIT_FACTOR = 0.5
NOTIONAL_USD = 150_000
HL_FEE_BPS = 3.5
CEX_FEE_BPS = 5.0
BASE = Path("data/normalized/funding_history")
CADENCE_CSV = Path("data/static/funding_cadence.csv")

# ── cadence lookup ────────────────────────────────────────────────────────

def _build_cadence() -> dict[tuple[str, str], int]:
    if not CADENCE_CSV.exists():
        return {}
    df = pd.read_csv(CADENCE_CSV)
    out: dict[tuple[str, str], int] = {}
    for _, r in df.iterrows():
        out[(str(r["venue"]), str(r["symbol"]))] = int(r["fund_interval_hours"])
    return out

_CADENCE = _build_cadence()

def _interval(venue: str, symbol: str) -> int:
    """Return funding interval in hours."""
    if venue == "hyperliquid":
        return 1   # HL is always 1h (documented in funding_cadence.csv)
    return _CADENCE.get((venue, symbol), 8)


# ── data loading ──────────────────────────────────────────────────────────

def _load_venue(venue: str) -> dict[str, pd.Series]:
    """Return {base_asset: Series(APR, hourly UTC index)}."""
    out: dict[str, pd.Series] = {}
    vdir = BASE / venue
    if not vdir.exists():
        return out
    for f in sorted(vdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty or "funding_rate" not in df.columns:
            continue
        sym = f.stem        # e.g. "BTCUSDT" or "BTC" for HL
        base = str(df.iloc[0].get("base_asset", "") or sym.replace("USDT", ""))
        iv_h = _interval(venue, sym)
        df2 = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        raw = df2.set_index("timestamp_utc")["funding_rate"]
        # Annualise: raw_rate is the per-period rate; APR = rate * periods_per_year
        apr = raw * (8760.0 / iv_h)
        out[base] = apr
    return out


def _to_hourly(s: pd.Series) -> pd.Series:
    """Resample to 1h grid.

    Archive timestamps have sub-second jitter (e.g. 08:00:00.102) so a plain
    reindex finds no exact matches.  We floor to the hour first, then
    resample to get a clean UTC grid, then forward-fill gaps.
    """
    if s.empty:
        return s
    s = s.copy()
    s.index = s.index.floor("1h")
    return s.resample("1h").last().ffill()


# ── episode detection ─────────────────────────────────────────────────────

def _episodes(spread: pd.Series, threshold: float) -> list[dict]:
    """Identify divergence episodes and measure their statistics.

    Episode starts when |spread| >= threshold.
    Episode ends when |spread| < threshold * HALF_EXIT_FACTOR OR spread reverses.
    """
    eps = []
    vals = spread.values
    times = spread.index
    n = len(vals)

    in_ep = False
    start_i = 0
    direction = 0

    for i in range(n):
        v = float(vals[i]) if not np.isnan(vals[i]) else np.nan
        if np.isnan(v):
            if in_ep:
                # Gap in data — close episode
                _finalize(eps, vals, times, start_i, i)
                in_ep = False
            continue

        if not in_ep:
            if abs(v) >= threshold:
                in_ep = True
                start_i = i
                direction = int(np.sign(v))
        else:
            exit_cond = (
                abs(v) < threshold * HALF_EXIT_FACTOR
                or (direction != 0 and int(np.sign(v)) != direction and abs(v) > 0.02)
            )
            if exit_cond or i == n - 1:
                _finalize(eps, vals, times, start_i, i)
                in_ep = False
                # Could immediately re-enter on same bar
                if abs(v) >= threshold:
                    in_ep = True
                    start_i = i
                    direction = int(np.sign(v))
    return eps


def _finalize(eps, vals, times, start_i, end_i):
    ep_vals = np.array([v for v in vals[start_i:end_i] if not np.isnan(v)])
    if len(ep_vals) < 1:
        return
    dur_h = float(end_i - start_i)
    init = float(vals[start_i])
    # Carry per hour = |APR| / 8760 * notional
    carry_usd = float(np.nansum(np.abs(ep_vals) / 8760.0 * NOTIONAL_USD))
    eps.append({
        "start": times[start_i],
        "duration_h": dur_h,
        "init_spread_apr": abs(init),
        "carry_usd": carry_usd,
    })


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading funding archives...")
    hl      = _load_venue("hyperliquid")
    binance = _load_venue("binance")
    bybit   = _load_venue("bybit")

    coins = sorted(set(hl) & set(binance) & set(bybit))
    print(f"Coins with HL + Binance + Bybit: {len(coins)}  {coins}\n")

    # ── spot-check APR magnitudes ─────────────────────────────────────────
    print("APR spot-check (last 5 rows):")
    for coin in ["TRUMP", "BTC", "ETH", "HYPE", "SOL"]:
        if coin not in hl or coin not in binance:
            continue
        h = _to_hourly(hl[coin])
        b = _to_hourly(binance[coin])
        print(f"  HL/{coin:6s}: mean={hl[coin].mean():.1%}  max={hl[coin].max():.1%}  "
              f"Binance mean={binance[coin].mean():.1%}  max={binance[coin].max():.1%}")
    print()

    # ── build hourly spread series ────────────────────────────────────────
    pair_types = ["HL-Binance", "HL-Bybit", "Binance-Bybit"]
    spreads: dict[tuple[str, str], pd.Series] = {}

    for coin in coins:
        hl_h  = _to_hourly(hl[coin])
        bin_h = _to_hourly(binance[coin])
        byt_h = _to_hourly(bybit[coin])
        idx   = hl_h.index
        bin_r = bin_h.reindex(idx).ffill()
        byt_r = byt_h.reindex(idx).ffill()
        spreads[("HL-Binance",   coin)] = hl_h - bin_r
        spreads[("HL-Bybit",     coin)] = hl_h - byt_r
        spreads[("Binance-Bybit",coin)] = bin_r - byt_r

    # ── spread distribution overview ─────────────────────────────────────
    print("Spread |APR| distribution (across all coins, hourly samples):")
    for pt in pair_types:
        all_vals = pd.concat([s for (p, _), s in spreads.items() if p == pt]).dropna()
        abs_vals = all_vals.abs()
        print(f"  {pt:15s}: p50={abs_vals.quantile(.5):.1%}  "
              f"p90={abs_vals.quantile(.9):.1%}  "
              f"p99={abs_vals.quantile(.99):.1%}  "
              f"max={abs_vals.max():.1%}  "
              f">200%: {(abs_vals>2.0).mean():.2%} of hours")
    print()

    rt_hl  = 2 * (HL_FEE_BPS + CEX_FEE_BPS) * 1e-4 * NOTIONAL_USD
    rt_cex = 2 * (CEX_FEE_BPS + CEX_FEE_BPS) * 1e-4 * NOTIONAL_USD
    print(f"Round-trip cost: HL-CEX ${rt_hl:.0f}  CEX-CEX ${rt_cex:.0f}\n")

    # ── episode analysis by threshold ─────────────────────────────────────
    for threshold in THRESHOLD_APRS:
        print(f"{'='*70}")
        print(f"ENTRY THRESHOLD = {threshold:.0%} APR  (exit at {threshold*HALF_EXIT_FACTOR:.0%})")
        print(f"{'='*70}")

        results: dict[str, list[dict]] = {pt: [] for pt in pair_types}
        for (pt, coin), spr in spreads.items():
            eps = _episodes(spr, threshold)
            results[pt].extend(eps)

        for pt in pair_types:
            eps = results[pt]
            if not eps:
                print(f"  {pt:15s}: 0 episodes")
                continue
            rt = rt_hl if pt.startswith("HL") else rt_cex
            dur    = np.array([e["duration_h"] for e in eps])
            carry  = np.array([e["carry_usd"]  for e in eps])
            profit = carry > rt
            n_settles_med = np.median(dur) * (1 if "HL" in pt else (1/4 if "Bybit" in pt else 1/8))
            print(f"  {pt:15s}: {len(eps):4d} eps | "
                  f"dur p50={np.median(dur):5.1f}h p75={np.percentile(dur,75):5.1f}h p95={np.percentile(dur,95):5.1f}h | "
                  f"carry p50=${np.median(carry):6.0f} p75=${np.percentile(carry,75):6.0f} | "
                  f"settles_p50={n_settles_med:.1f} | "
                  f"profitable={profit.mean():.0%} ({profit.sum()}/{len(eps)})")

    # ── HL-Binance per-coin detail at 200% ────────────────────────────────
    thr = 2.0
    print(f"\n{'='*70}")
    print("HL-Binance per-coin breakdown at thr=200% APR")
    print(f"{'='*70}")
    all_hl_bin_eps: dict[str, list] = {}
    for (pt, coin), spr in spreads.items():
        if pt != "HL-Binance":
            continue
        eps = _episodes(spr, thr)
        all_hl_bin_eps[coin] = eps

    print(f"  {'coin':8s}  {'eps':>5s}  {'dur_p50':>9s}  {'carry_p50':>10s}  "
          f"{'carry_p75':>10s}  {'profit%':>7s}  {'HL_settles_p50':>14s}")
    for coin in sorted(all_hl_bin_eps, key=lambda c: -len(all_hl_bin_eps[c])):
        eps = all_hl_bin_eps[coin]
        if not eps:
            print(f"  {coin:8s}  {'0':>5s}")
            continue
        dur   = np.array([e["duration_h"] for e in eps])
        carry = np.array([e["carry_usd"]  for e in eps])
        prof  = (carry > rt_hl).mean()
        print(f"  {coin:8s}  {len(eps):5d}  {np.median(dur):8.1f}h  "
              f"${np.median(carry):9,.0f}  ${np.percentile(carry,75):9,.0f}  "
              f"{prof:6.0%}  {np.median(dur):13.1f}")

    # ── key comparison summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("KEY COMPARISON: HL-Binance vs Binance-Bybit at thr=200%")
    print(f"(Binance-Bybit = the CEX-vs-CEX baseline that already FAILED)")
    print(f"{'='*70}")
    for pt in ["HL-Binance", "Binance-Bybit"]:
        eps = []
        for (p, coin), spr in spreads.items():
            if p != pt: continue
            eps.extend(_episodes(spr, 2.0))
        if not eps:
            print(f"  {pt}: 0 episodes"); continue
        rt   = rt_hl if pt.startswith("HL") else rt_cex
        dur  = np.array([e["duration_h"] for e in eps])
        carry = np.array([e["carry_usd"] for e in eps])
        print(f"  {pt:15s}: {len(eps):4d} eps  "
              f"dur_mean={dur.mean():.1f}h  dur_med={np.median(dur):.1f}h  "
              f"carry_mean=${carry.mean():,.0f}  carry_med=${np.median(carry):,.0f}  "
              f"RT=${rt:.0f}  profitable={( carry>rt).mean():.0%}")
        # Distribution of episode durations
        buckets = [(1,4,'<4h'), (4,12,'4-12h'), (12,24,'12-24h'), (24,72,'24-72h'), (72,99999,'>72h')]
        for lo, hi, label in buckets:
            n = ((dur >= lo) & (dur < hi)).sum()
            print(f"    {label:8s}: {n:4d} eps ({n/len(eps):.0%})")


if __name__ == "__main__":
    main()
