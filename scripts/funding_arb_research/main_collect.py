"""Top-level data-collection entry point.

Usage:

    python -m funding_arb.main_collect \
        --venues binance bybit okx hyperliquid \
        --symbols BTC ETH SOL \
        --start 2025-01-01 --end 2025-04-01 \
        --out data/normalized

    python -m funding_arb.main_collect --venues gmx_v2 --gmx-snapshot
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.collectors.binance_collector import BinanceCollector
from src.collectors.bybit_collector import BybitCollector
from src.collectors.depth import DepthCollector, to_dataframe as depth_to_dataframe
from src.collectors.gmx_collector import GMXCollector
from src.collectors.hyperliquid_collector import HyperliquidCollector
from src.collectors.okx_collector import OKXCollector
from src.normalization.normalize_funding import stitch_funding
from src.utils.io import load_yaml, normalized_dir, project_root, write_parquet
from src.utils.logging import get_logger
from src.utils.time import to_utc


LOG = get_logger("funding_arb.main_collect")


def _venue_symbol(venue: str, base: str) -> str:
    if venue == "okx":
        return f"{base}-USDT-SWAP"
    if venue == "hyperliquid":
        return base
    return f"{base}USDT"


async def _collect_cex(
    venue: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    out_dir: Path,
) -> Path:
    if venue == "binance":
        async with BinanceCollector().ctx as ctx:
            coll = BinanceCollector(ctx=ctx)
            frames = [await coll.fetch_funding_history(_venue_symbol(venue, s), start, end)
                      for s in symbols]
            # Snapshot predicted funding (premiumIndex) for each symbol
            predicted_frames = []
            for s in symbols:
                try:
                    predicted_frames.append(
                        await coll.fetch_premium_index(_venue_symbol(venue, s))
                    )
                except Exception as e:
                    LOG.warning("binance premiumIndex %s failed: %s", s, e)
            if predicted_frames:
                pf = stitch_funding(predicted_frames)
                if not pf.empty:
                    write_parquet(pf, out_dir / "predicted_funding_binance.parquet")
                    LOG.info("wrote %d binance predicted-funding rows", len(pf))
    elif venue == "bybit":
        async with BybitCollector().ctx as ctx:
            coll = BybitCollector(ctx=ctx)
            frames = [await coll.fetch_funding_history(_venue_symbol(venue, s), start, end)
                      for s in symbols]
    elif venue == "okx":
        async with OKXCollector().ctx as ctx:
            coll = OKXCollector(ctx=ctx)
            frames = [await coll.fetch_funding_history(_venue_symbol(venue, s), start, end)
                      for s in symbols]
    elif venue == "hyperliquid":
        async with HyperliquidCollector().ctx as ctx:
            coll = HyperliquidCollector(ctx=ctx)
            frames = [await coll.fetch_funding_history(s, start, end) for s in symbols]
            # Snapshot predictedFundings (covers HL + major CEX-perp predicted rates)
            try:
                pf = await coll.fetch_predicted_fundings()
                if pf is not None and not pf.empty:
                    write_parquet(pf, out_dir / "predicted_funding_hyperliquid.parquet")
                    LOG.info("wrote %d HL-aggregated predicted-funding rows", len(pf))
            except Exception as e:
                LOG.warning("hyperliquid predictedFundings failed: %s", e)
    else:
        raise ValueError(f"unknown venue: {venue}")

    df = stitch_funding(frames)
    out_path = out_dir / f"funding_{venue}.parquet"
    write_parquet(df, out_path)
    LOG.info("wrote %s rows -> %s", len(df), out_path)
    return out_path


async def _collect_gmx(
    out_dir: Path, *, prefer_rpc: bool = False, history_intervals: int = 14 * 24,
) -> Path:
    async with GMXCollector().ctx as ctx:
        coll = GMXCollector(ctx=ctx)
        df = await coll.snapshot_to_frame(prefer_rpc=prefer_rpc)
        history = await coll.fetch_imbalance_distribution(intervals=history_intervals)
    out_path = out_dir / "gmx_v2_snapshot.parquet"
    write_parquet(df, out_path)
    LOG.info("wrote %s snapshot rows -> %s", len(df), out_path)
    if history is not None and not history.empty:
        hist_path = out_dir / "gmx_v2_imbalance_history.parquet"
        write_parquet(history, hist_path)
        LOG.info("wrote %s history rows -> %s", len(history), hist_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Funding-arb data collector")
    p.add_argument("--venues", nargs="+", required=True,
                   choices=["binance", "bybit", "okx", "hyperliquid", "gmx_v2"])
    p.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "SOL"])
    p.add_argument("--start", default=None,
                   help="UTC start date (YYYY-MM-DD). Default: 90d ago")
    p.add_argument("--end", default=None,
                   help="UTC end date (YYYY-MM-DD). Default: now")
    p.add_argument("--out", default=None,
                   help="Output directory. Default: data/normalized")
    p.add_argument("--gmx-snapshot", action="store_true",
                   help="Capture GMX v2 snapshot instead of historical funding")
    p.add_argument("--gmx-rpc", action="store_true",
                   help="Use Arbitrum RPC + Reader contract for GMX (requires env vars)")
    p.add_argument("--gmx-history-intervals", type=int, default=14 * 24,
                   help="Hours of GMX hourly imbalance history to pull (default 14d)")
    p.add_argument("--depth", action="store_true",
                   help="Snapshot L2 depth on each requested CEX venue")
    p.add_argument("--depth-limit", type=int, default=100,
                   help="Levels per side to request (default 100)")
    p.add_argument("--depth-band-bps", type=float, default=10.0,
                   help="Band around inside price counted as available liquidity")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    end = to_utc(args.end) if args.end else pd.Timestamp.now(tz=timezone.utc)
    start = to_utc(args.start) if args.start else end - timedelta(days=90)
    out_dir = Path(args.out) if args.out else normalized_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow venue-config override (loaded for visibility/logging only).
    venues_cfg = load_yaml(project_root() / "config" / "venues.yaml")
    LOG.info("venues_cfg keys: %s", list(venues_cfg.keys()))

    for venue in args.venues:
        if venue == "gmx_v2":
            await _collect_gmx(out_dir, prefer_rpc=args.gmx_rpc,
                                history_intervals=args.gmx_history_intervals)
            continue
        await _collect_cex(venue, args.symbols, start, end, out_dir)

    if args.depth:
        await _collect_depth(args.venues, args.symbols, out_dir,
                             limit=args.depth_limit, band_bps=args.depth_band_bps)


async def _collect_depth(
    venues: list[str], symbols: list[str], out_dir: Path,
    *, limit: int, band_bps: float,
) -> Path | None:
    cex_venues = [v for v in venues if v in ("binance", "bybit", "okx", "hyperliquid")]
    if not cex_venues:
        return None
    requests = [(v, _venue_symbol(v, s)) for v in cex_venues for s in symbols]
    async with DepthCollector() as dc:
        snaps = await dc.fetch_many(requests, limit=limit)
    df = depth_to_dataframe(snaps, band_bps=band_bps)
    out_path = out_dir / "depth_snapshot.parquet"
    write_parquet(df, out_path)
    LOG.info("wrote %d depth rows -> %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    asyncio.run(_main())
