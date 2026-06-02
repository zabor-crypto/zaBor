"""One-shot collector that snapshots Bitget MMR tier tables for every
USDT-FUTURES symbol and writes a single parquet for the engine to load.

Run:
    python -m src.collectors.bitget_mmr_archive

Output: ``data/static/bitget_mmr.parquet``  (one row per (symbol, tier))

The MMR ladder rarely changes; re-run weekly or after a notable Bitget
risk-limit announcement. The engine's ``BitgetMMR`` loader reads this
parquet at init and falls back to ``config/bitget_mmr_fallback.yaml``
only for symbols missing here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from ..utils.io import write_parquet, project_root
from ..utils.logging import get_logger
from .base import CollectorContext
from .bitget_collector import BitgetCollector

_LOG = get_logger("funding_arb.collectors.bitget_mmr_archive")


async def build(out_path: Path | None = None) -> pd.DataFrame:
    out_path = out_path or (project_root() / "data" / "static" / "bitget_mmr.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with CollectorContext(venue="bitget", rest_base="https://api.bitget.com",
                                 rate_limit_rps=10.0) as ctx:
        bc = BitgetCollector(ctx=ctx)
        contracts = await bc.fetch_contracts()
        symbols = contracts["symbol"].dropna().tolist() if not contracts.empty else []
        _LOG.info("snapshotting MMR ladders for %d symbols", len(symbols))
        frames = []
        for i, sym in enumerate(symbols, 1):
            try:
                df = await bc.fetch_mmr_tiers(sym)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("MMR fetch failed for %s: %s", sym, e)
                continue
            if not df.empty:
                frames.append(df)
            if i % 50 == 0:
                _LOG.info("  progress: %d/%d", i, len(symbols))

    if not frames:
        _LOG.error("no MMR ladders fetched")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    write_parquet(out, out_path)
    out.to_csv(out_path.with_suffix(".csv"), index=False)
    _LOG.info("wrote %d MMR rows for %d symbols to %s",
              len(out), out["symbol"].nunique(), out_path)
    return out


if __name__ == "__main__":
    asyncio.run(build())
