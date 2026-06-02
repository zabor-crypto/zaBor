"""Recorder CLI entrypoint.

Usage (from the funding_arb directory)::

    python -m src.recorder.main --venues bitget binance \
        --coins BTC ETH SOL --quota-gb 50

The recorder is a long-lived process. Press Ctrl-C (SIGINT) for clean
shutdown; in-flight buffers are flushed to parquet before exit.

Output layout:
  data/recorder/<channel>/dt=YYYY-MM-DD/venue=<v>/coin=<c>/<HHMMSSZ>_<n>.parquet
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .rest_pollers.binance_mark import BinanceMarkPoller, BinanceMarkPollerConfig
from .supervisor import RecorderSupervisor
from .venue_clients.binance import BinanceClient
from .venue_clients.bitget import BitgetClient
from .writer import ParquetWriter, WriterConfig
from ..utils.io import data_dir
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.recorder.main")

_VENUE_FACTORIES = {
    "bitget": BitgetClient.build,
    "binance": BinanceClient.build,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-hosted WS market-data recorder.")
    p.add_argument("--venues", nargs="+", required=True,
                   choices=sorted(_VENUE_FACTORIES.keys()))
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"])
    p.add_argument("--depth-levels", type=str, default="20",
                   help="Order-book depth subscription per venue. "
                        "'full' = full L2 diff (heaviest, ~270 MB/day/coin). "
                        "'20' = top-20 partial book (default, ~30 MB/day/coin). "
                        "'15' / '10' / '5' supported by some venues.")
    p.add_argument("--quota-gb", type=float, default=50.0,
                   help="Max disk usage under data/recorder/ before oldest day-partitions are deleted.")
    p.add_argument("--flush-rows", type=int, default=5_000)
    p.add_argument("--flush-interval-s", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=None,
                   help="Override output base dir (default: data/recorder).")
    p.add_argument("--max-runtime-s", type=float, default=None,
                   help="Optional auto-shutdown after this many seconds (for smoke tests).")
    p.add_argument("--binance-mark-poll-s", type=float, default=10.0,
                   help="REST polling cadence for Binance mark/funding (the WS gateway "
                        "drops @markPrice from Tokyo region). Set 0 to disable.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    base = args.out or (data_dir() / "recorder")
    writer = ParquetWriter(WriterConfig(
        flush_rows=args.flush_rows,
        flush_interval_s=args.flush_interval_s,
        base_dir=base,
    ))
    clients = [_VENUE_FACTORIES[v](args.coins, writer, depth_levels=args.depth_levels)
               for v in args.venues]
    # Binance mark+funding poller fills the WS-gateway gap (no @markPrice on
    # Tokyo region). Skip if Binance isn't selected or polling is disabled.
    if "binance" in args.venues and args.binance_mark_poll_s > 0:
        clients.append(BinanceMarkPoller(
            BinanceMarkPollerConfig(coins=args.coins, poll_interval_s=args.binance_mark_poll_s),
            writer,
        ))
    sup = RecorderSupervisor(clients, writer, quota_gb=args.quota_gb)
    _LOG.info("recorder starting venues=%s coins=%s out=%s",
              args.venues, args.coins, base)
    runner = asyncio.create_task(sup.run(), name="recorder-supervisor")
    if args.max_runtime_s is not None:
        async def _auto_stop():
            await asyncio.sleep(args.max_runtime_s)
            _LOG.info("max-runtime reached; signalling stop")
            sup._stop.set()
        asyncio.create_task(_auto_stop(), name="recorder-auto-stop")
    await runner


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
