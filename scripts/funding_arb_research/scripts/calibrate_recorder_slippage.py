"""Build a slippage calibration table from recorder book snapshots.

Runs on the machine that holds the recorder data (VPS). Samples up to
SAMPLES_PER_COIN book snapshots per (venue, coin), computes mean slippage
at a grid of notional sizes, and writes a small calibration CSV.

Usage (from /opt/funding_arb/):
    python -m scripts.calibrate_recorder_slippage \
        --recorder-root /opt/funding_arb/data/recorder \
        --out data/static/recorder_slippage_calibration.csv

Output columns:
    venue, coin, notional_usd, mean_slippage_bps, p50_slippage_bps,
    p75_slippage_bps, p95_slippage_bps, n_snapshots
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

NOTIONAL_GRID_USD = [10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000]
SAMPLES_PER_COIN = 200
RANDOM_SEED = 42


def _slippage_from_snapshot(
    snap: pd.DataFrame,
    notional_usd: float,
    side: str = "buy",
) -> float | None:
    """Walk one side of a snapshot, return slippage_bps or None if insufficient depth."""
    book_side = "ask" if side == "buy" else "bid"
    levels = snap[snap["side"] == book_side].copy()
    if levels.empty:
        return None
    levels = levels.sort_values("price", ascending=(book_side == "ask"))
    best_price = float(levels.iloc[0]["price"])
    if best_price <= 0:
        return None

    filled_cost = 0.0
    filled_qty = 0.0
    for _, row in levels.iterrows():
        remaining = notional_usd - filled_cost
        if remaining <= 0:
            break
        lev_notional = float(row["price"]) * float(row["size"])
        take = min(lev_notional, remaining)
        filled_cost += take
        filled_qty += take / float(row["price"])

    if filled_qty <= 0 or filled_cost < notional_usd * 0.99:
        # Insufficient depth to fill this notional
        return None

    eff_price = filled_cost / filled_qty
    if book_side == "ask":
        slip_bps = (eff_price / best_price - 1.0) * 1e4
    else:
        slip_bps = (best_price / eff_price - 1.0) * 1e4
    return max(0.0, slip_bps)


def calibrate_pair(recorder_root: Path, venue: str, coin: str) -> list[dict]:
    book_base = recorder_root / "book"
    # Collect all parquet files for this (venue, coin)
    pattern = f"dt=*/venue={venue}/coin={coin}/*.parquet"
    files = sorted(book_base.glob(pattern))
    if not files:
        return []

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(files, min(SAMPLES_PER_COIN, len(files)))

    slip_by_notional: dict[float, list[float]] = {n: [] for n in NOTIONAL_GRID_USD}

    for fpath in sampled:
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        if df.empty:
            continue
        # Take one snapshot from this file: the last complete snapshot.
        # Group by timestamp_utc (works for both bitget[seq] and binance[seq_first/seq_last]).
        snap_col = "is_snapshot" if "is_snapshot" in df.columns else None
        if snap_col:
            snap_rows = df[df[snap_col]]
        else:
            snap_rows = df
        if snap_rows.empty:
            continue
        last_ts = snap_rows["timestamp_utc"].max()
        snap = snap_rows[snap_rows["timestamp_utc"] == last_ts]
        if snap.empty:
            continue
        for notional in NOTIONAL_GRID_USD:
            s = _slippage_from_snapshot(snap, notional, side="buy")
            if s is not None:
                slip_by_notional[notional].append(s)

    rows = []
    n_snapshots = min(len(v) for v in slip_by_notional.values()) if slip_by_notional else 0
    for notional, values in slip_by_notional.items():
        if not values:
            continue
        arr = np.array(values)
        rows.append({
            "venue": venue,
            "coin": coin,
            "notional_usd": notional,
            "mean_slippage_bps": float(arr.mean()),
            "p50_slippage_bps": float(np.percentile(arr, 50)),
            "p75_slippage_bps": float(np.percentile(arr, 75)),
            "p95_slippage_bps": float(np.percentile(arr, 95)),
            "n_snapshots": len(values),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder-root", default="/opt/funding_arb/data/recorder",
                    help="Path to recorder data root")
    ap.add_argument("--out", default="data/static/recorder_slippage_calibration.csv")
    args = ap.parse_args()

    recorder_root = Path(args.recorder_root)
    book_root = recorder_root / "book"

    # Discover all (venue, coin) pairs with book data
    pairs: list[tuple[str, str]] = []
    for venue_dir in sorted(book_root.glob("dt=*/venue=*")):
        venue = venue_dir.name.replace("venue=", "")
        for coin_dir in sorted(venue_dir.glob("coin=*")):
            coin = coin_dir.name.replace("coin=", "")
            pairs.append((venue, coin))
    pairs = sorted(set(pairs))
    print(f"[calibrate] discovered {len(pairs)} (venue, coin) pairs")

    all_rows: list[dict] = []
    for venue, coin in pairs:
        rows = calibrate_pair(recorder_root, venue, coin)
        if rows:
            # Report mean slippage at 150K as a spot check
            r150 = next((r for r in rows if r["notional_usd"] == 150_000), None)
            if r150:
                print(f"  {venue}/{coin}: n={r150['n_snapshots']:3d}  "
                      f"150K_slip={r150['mean_slippage_bps']:.3f}bp (p95={r150['p95_slippage_bps']:.3f}bp)")
        all_rows.extend(rows)

    if not all_rows:
        print("[calibrate] no data collected — exiting")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    print(f"\n[calibrate] wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
