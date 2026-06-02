"""GMX v2 (Synthetics) feasibility-only collector.

This collector is a *research probe*, not a live execution feed. It pulls
market state (open interest, pool liquidity, funding/borrowing parameters)
either from the public stats subgraph or directly from an Arbitrum RPC via
the Reader contract, and produces a normalized snapshot that the
``gmx_imbalance_feasibility`` module can consume.

Two modes:

1. ``subgraph`` (default): hits the Synthetics Stats GraphQL endpoint. Fast
   to set up but does not include funding/borrowing factors.
2. ``rpc``: reads on-chain state via web3 + the Reader contract. Authoritative
   and includes funding/borrowing factors. Requires:
       - ``ARBITRUM_RPC_URL``: HTTPS RPC endpoint
       - ``GMX_READER_ADDRESS``: GMX v2 Reader address on Arbitrum
       - ``GMX_DATA_STORE_ADDRESS``: GMX v2 DataStore address on Arbitrum
   Addresses are intentionally not hardcoded — they are deployment-specific
   and the canonical record lives in the GMX deploy registry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..utils.io import hash_payload
from ..utils.logging import get_logger
from ..utils.time import to_utc
from .base import CollectorContext, request_json


_RPC_LOG = get_logger("funding_arb.collectors.gmx_v2.rpc")


@dataclass
class GMXMarketSnapshot:
    timestamp_utc: pd.Timestamp
    market: str
    long_oi_usd: float
    short_oi_usd: float
    pool_value_usd: float
    long_token: str
    short_token: str
    funding_factor_per_second: float | None
    borrowing_factor_long_per_second: float | None
    borrowing_factor_short_per_second: float | None
    raw_hash: str

    def imbalance_pct(self) -> float:
        total = self.long_oi_usd + self.short_oi_usd
        if total <= 0:
            return 0.0
        return (self.long_oi_usd - self.short_oi_usd) / total


class GMXCollector:
    """GMX v2 collector. Default mode = subgraph stats."""

    def __init__(
        self,
        ctx: Optional[CollectorContext] = None,
        subgraph_url: str = "https://subgraph.satsuma-prod.com/3b2ced13c8d9/gmx/synthetics-arbitrum-stats/api",
        chain: str = "arbitrum",
    ):
        self.subgraph_url = subgraph_url
        self.chain = chain
        self.ctx = ctx or CollectorContext(
            venue="gmx_v2",
            rest_base=subgraph_url,
            rate_limit_rps=2,
        )

    # ----- Public stats API path (gmxinfra.io) --------------------------- #

    async def fetch_market_snapshots_stats_api(self) -> list[GMXMarketSnapshot]:
        """Pull live market info from the gmxinfra.io public API.

        This replaces the legacy Satsuma subgraph (whose host has been
        deprecated). It returns one snapshot per market with OI in USD
        (scaled 1e30) and available liquidity per side.
        """
        # Use a fresh httpx client because our context is bound to the subgraph URL
        import httpx
        try:
            async with httpx.AsyncClient(timeout=20.0) as cl:
                resp = await cl.get("https://arbitrum-api.gmxinfra.io/markets/info")
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:
            self.ctx.logger.warning("gmx stats API fetch failed (%s)", e)
            return []

        markets = (payload or {}).get("markets") or []
        snaps: list[GMXMarketSnapshot] = []
        ts = pd.Timestamp.now(tz="UTC")
        for m in markets:
            try:
                long_oi = float(m.get("openInterestLong") or 0) / 1e30
                short_oi = float(m.get("openInterestShort") or 0) / 1e30
                avail_long = float(m.get("availableLiquidityLong") or 0) / 1e30
                avail_short = float(m.get("availableLiquidityShort") or 0) / 1e30
                pool_value = avail_long + avail_short  # rough proxy
            except (TypeError, ValueError):
                continue
            snaps.append(GMXMarketSnapshot(
                timestamp_utc=ts,
                market=m.get("marketToken", ""),
                long_oi_usd=long_oi, short_oi_usd=short_oi,
                pool_value_usd=pool_value,
                long_token=str(m.get("longToken", "")),
                short_token=str(m.get("shortToken", "")),
                funding_factor_per_second=None,
                borrowing_factor_long_per_second=None,
                borrowing_factor_short_per_second=None,
                raw_hash=hash_payload(m),
            ))
        return snaps

    # ----- Subgraph path (kept as opportunistic fallback) ---------------- #

    async def fetch_market_snapshots_subgraph(self, limit: int = 100) -> list[GMXMarketSnapshot]:
        """Pull a snapshot of all v2 markets via the Stats subgraph.

        Note: the Synthetics Stats subgraph schema evolves; we run a
        minimal query and gracefully tolerate missing fields.
        """
        query = """
        {
          marketInfos(first: %d) {
            id
            marketToken
            indexToken { symbol address }
            longToken  { symbol address }
            shortToken { symbol address }
          }
          poolValueInfos(first: %d, orderBy: timestamp, orderDirection: desc) {
            marketAddress
            poolValue
            longOpenInterestUsd
            shortOpenInterestUsd
            timestamp
          }
        }
        """ % (limit, limit)
        try:
            payload = await request_json(
                self.ctx, "POST", "",
                json_body={"query": query},
            )
        except Exception as e:
            self.ctx.logger.warning(
                "gmx subgraph fetch failed (%s); returning empty snapshot list", e,
            )
            return []

        if not payload or "data" not in payload:
            self.ctx.logger.warning("gmx subgraph returned no data; payload=%s",
                                     str(payload)[:200])
            return []

        market_meta = {
            m["id"]: m for m in (payload["data"].get("marketInfos") or [])
        }
        snaps: list[GMXMarketSnapshot] = []
        for entry in payload["data"].get("poolValueInfos") or []:
            mkt = market_meta.get(entry.get("marketAddress")) or {}
            ts = to_utc(int(entry["timestamp"]))
            snaps.append(GMXMarketSnapshot(
                timestamp_utc=ts,
                market=entry.get("marketAddress", ""),
                long_oi_usd=float(entry.get("longOpenInterestUsd", 0)) / 1e30,
                short_oi_usd=float(entry.get("shortOpenInterestUsd", 0)) / 1e30,
                pool_value_usd=float(entry.get("poolValue", 0)) / 1e30,
                long_token=(mkt.get("longToken") or {}).get("symbol", ""),
                short_token=(mkt.get("shortToken") or {}).get("symbol", ""),
                funding_factor_per_second=None,      # not available in this query
                borrowing_factor_long_per_second=None,
                borrowing_factor_short_per_second=None,
                raw_hash=hash_payload(entry),
            ))
        return snaps

    # ----- RPC / Reader-contract path ------------------------------------ #

    @staticmethod
    def _required_env(*names: str) -> dict[str, str]:
        out = {}
        missing = []
        for n in names:
            v = os.environ.get(n)
            if not v:
                missing.append(n)
            else:
                out[n] = v
        if missing:
            raise RuntimeError(
                f"GMX RPC mode requires environment variables: {', '.join(missing)}"
            )
        return out

    def fetch_rpc_snapshots(self) -> list[GMXMarketSnapshot]:
        """Read market state directly from the Arbitrum Reader contract.

        What we read on-chain (price-independent):
          * ``Reader.getMarkets`` → market enumeration with the four token
            addresses (marketToken, indexToken, longToken, shortToken).
          * ``DataStore.getUint(fundingFactorKey(market))`` →
            ``fundingFactor`` for the market (per second, scaled by 1e30).
          * ``DataStore.getUint(borrowingFactorKey(market, isLong))`` →
            per-side borrowing factor (per second, scaled by 1e30).
          * ``DataStore.getUint(openInterestKey(market, collateralToken, isLong))``
            for both collateral tokens, summed → token-denominated OI.

        ERC-20 ``symbol()`` is called per token to surface human-readable
        labels.

        We deliberately do *not* call ``getMarketInfo`` / ``getOpenInterestWithPnl``
        here: they require real index/collateral prices to return correct
        results, and wiring an on-chain oracle path is out of scope for the
        feasibility MVP. OI in USD comes from the subgraph (which already
        applies live prices) and is merged in by ``snapshot_to_frame``.
        """
        try:
            from web3 import Web3
        except ImportError as e:
            raise RuntimeError("web3 is required for GMX RPC mode") from e

        env = self._required_env(
            "ARBITRUM_RPC_URL", "GMX_READER_ADDRESS", "GMX_DATA_STORE_ADDRESS",
        )
        w3 = Web3(Web3.HTTPProvider(env["ARBITRUM_RPC_URL"]))
        if not w3.is_connected():
            raise RuntimeError("could not connect to ARBITRUM_RPC_URL")

        reader_addr = Web3.to_checksum_address(env["GMX_READER_ADDRESS"])
        data_store_addr = Web3.to_checksum_address(env["GMX_DATA_STORE_ADDRESS"])
        reader = w3.eth.contract(address=reader_addr, abi=_GMX_READER_MIN_ABI)
        data_store = w3.eth.contract(address=data_store_addr, abi=_GMX_DATA_STORE_MIN_ABI)

        markets_raw = reader.functions.getMarkets(data_store_addr, 0, 1000).call()
        _RPC_LOG.info("GMX RPC: enumerated %d markets", len(markets_raw))
        snaps: list[GMXMarketSnapshot] = []
        ts = pd.Timestamp.now(tz="UTC")

        symbol_cache: dict[str, str] = {}

        def _symbol(addr: str) -> str:
            if addr in symbol_cache:
                return symbol_cache[addr]
            try:
                erc20 = w3.eth.contract(
                    address=Web3.to_checksum_address(addr), abi=_ERC20_MIN_ABI,
                )
                sym = erc20.functions.symbol().call()
            except Exception as e:
                _RPC_LOG.debug("symbol() failed for %s: %s", addr, e)
                sym = addr[:8]
            symbol_cache[addr] = sym
            return sym

        for m in markets_raw:
            try:
                market_token, index_token, long_token, short_token = m
            except (TypeError, ValueError):
                _RPC_LOG.warning("skipping malformed market entry: %s", m)
                continue

            try:
                ff_key = _saved_funding_factor_per_second_key(market_token)
                bf_long_key = _borrowing_factor_per_second_key(market_token, True)
                bf_short_key = _borrowing_factor_per_second_key(market_token, False)
                # SAVED_FUNDING_FACTOR_PER_SECOND is int256 (sign carries
                # direction); borrowing factors are uint256.
                funding_factor = float(data_store.functions.getInt(ff_key).call()) / 1e30
                borrowing_long = float(data_store.functions.getUint(bf_long_key).call()) / 1e30
                borrowing_short = float(data_store.functions.getUint(bf_short_key).call()) / 1e30
            except Exception as e:
                _RPC_LOG.warning("DataStore reads failed for %s: %s", market_token, e)
                funding_factor = borrowing_long = borrowing_short = None

            snaps.append(GMXMarketSnapshot(
                timestamp_utc=ts, market=market_token,
                long_oi_usd=0.0, short_oi_usd=0.0,    # filled from subgraph
                pool_value_usd=0.0,                    # filled from subgraph
                long_token=_symbol(long_token),
                short_token=_symbol(short_token),
                funding_factor_per_second=funding_factor,
                borrowing_factor_long_per_second=borrowing_long,
                borrowing_factor_short_per_second=borrowing_short,
                raw_hash=hash_payload({"market": market_token, "ts": str(ts)}),
            ))
        return snaps

    async def fetch_imbalance_distribution(
        self,
        *,
        intervals: int = 14 * 24,
        interval_hours: int = 1,
    ) -> pd.DataFrame:
        """Multi-week imbalance + carry-APR distribution.

        Pulls historical hourly snapshots from the Synthetics Stats subgraph
        (one query per page) and returns a long-format frame the
        feasibility module can aggregate over multiple weeks.
        """
        try:
            payload = await request_json(
                self.ctx, "POST", "",
                json_body={"query": _SUBGRAPH_HOURLY_QUERY % {"first": intervals}},
            )
        except Exception as e:
            self.ctx.logger.warning("gmx hourly subgraph failed: %s", e)
            return pd.DataFrame()
        rows = (payload or {}).get("data", {}).get("marketInfoHourlyReports") or []
        out = []
        for r in rows:
            try:
                ts = to_utc(int(r["timestamp"]))
                long_oi = float(r["longOpenInterestUsd"]) / 1e30
                short_oi = float(r["shortOpenInterestUsd"]) / 1e30
            except (KeyError, ValueError, TypeError):
                continue
            total = long_oi + short_oi
            out.append({
                "timestamp_utc": ts,
                "market": r.get("marketAddress", ""),
                "long_oi_usd": long_oi,
                "short_oi_usd": short_oi,
                "imbalance_pct": (long_oi - short_oi) / total if total > 0 else 0.0,
            })
        return pd.DataFrame(out)

    async def snapshot_to_frame(self, *, prefer_rpc: bool = False) -> pd.DataFrame:
        rpc_snaps: list[GMXMarketSnapshot] = []
        if prefer_rpc or os.environ.get("GMX_PREFER_RPC") == "1":
            try:
                rpc_snaps = self.fetch_rpc_snapshots()
            except Exception as e:
                _RPC_LOG.warning("GMX RPC mode failed (%s); falling back to subgraph", e)

        # Stats API is the live OI source. Subgraph is kept only as a
        # last-resort fallback for environments where it's still up.
        stats_snaps = await self.fetch_market_snapshots_stats_api()
        subgraph_snaps = stats_snaps or await self.fetch_market_snapshots_subgraph()

        # If we have both, merge: RPC supplies factors, stats/subgraph supplies OI/pool.
        if rpc_snaps and subgraph_snaps:
            sg_by_market = {s.market.lower(): s for s in subgraph_snaps}
            merged: list[GMXMarketSnapshot] = []
            for r in rpc_snaps:
                sg = sg_by_market.get(r.market.lower())
                if sg is None:
                    merged.append(r)
                    continue
                merged.append(GMXMarketSnapshot(
                    timestamp_utc=r.timestamp_utc, market=r.market,
                    long_oi_usd=sg.long_oi_usd, short_oi_usd=sg.short_oi_usd,
                    pool_value_usd=sg.pool_value_usd,
                    long_token=r.long_token or sg.long_token,
                    short_token=r.short_token or sg.short_token,
                    funding_factor_per_second=r.funding_factor_per_second,
                    borrowing_factor_long_per_second=r.borrowing_factor_long_per_second,
                    borrowing_factor_short_per_second=r.borrowing_factor_short_per_second,
                    raw_hash=r.raw_hash,
                ))
            snaps = merged
        else:
            snaps = rpc_snaps or subgraph_snaps

        if not snaps:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for s in snaps:
            rows.append({
                "timestamp_utc": s.timestamp_utc,
                "venue": "gmx_v2",
                "symbol": f"{s.long_token}-{s.short_token}",
                "base_asset": s.long_token,
                "quote_asset": s.short_token,
                "instrument_type": "perp_amm",
                "long_oi": s.long_oi_usd,
                "short_oi": s.short_oi_usd,
                "open_interest": s.long_oi_usd + s.short_oi_usd,
                "pool_value_usd": s.pool_value_usd,
                "imbalance_pct": s.imbalance_pct(),
                "funding_factor_per_second": s.funding_factor_per_second,
                "borrowing_factor_long_per_second": s.borrowing_factor_long_per_second,
                "borrowing_factor_short_per_second": s.borrowing_factor_short_per_second,
                "source": "gmx_v2/subgraph",
                "raw_payload_hash": s.raw_hash,
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Subgraph + ABI constants
# --------------------------------------------------------------------------- #

_SUBGRAPH_HOURLY_QUERY = """
{
  marketInfoHourlyReports(first: %(first)d, orderBy: timestamp, orderDirection: desc) {
    marketAddress
    timestamp
    longOpenInterestUsd
    shortOpenInterestUsd
  }
}
"""


# --------------------------------------------------------------------------- #
# DataStore key helpers
# --------------------------------------------------------------------------- #
#
# GMX v2 stores config values in a single ``DataStore`` keyed by
# ``keccak256(abi.encode(BASE_KEY, ...args))``. ``BASE_KEY`` is itself a
# keccak hash of the human-readable label. The labels are stable across
# deploys; the addresses change.

def _hash_string(label: str) -> bytes:
    """GMX wraps every label in ``keccak256(abi.encode("LABEL"))`` — note
    the abi.encode, not raw bytes. The two produce different digests."""
    from eth_utils import keccak
    from eth_abi import encode
    return keccak(encode(["string"], [label]))


def _abi_encode_bytes32_address(b32: bytes, addr: str) -> bytes:
    from eth_abi import encode
    return encode(["bytes32", "address"], [b32, addr])


def _abi_encode_bytes32_address_bool(b32: bytes, addr: str, b: bool) -> bytes:
    from eth_abi import encode
    return encode(["bytes32", "address", "bool"], [b32, addr, b])


def _saved_funding_factor_per_second_key(market: str) -> bytes:
    """Live (dynamic) funding factor for the market, int256, scaled 1e30.

    GMX v2 uses adaptive funding; the canonical "current" rate is the
    saved int — positive => longs pay shorts, negative => shorts pay longs.
    """
    from eth_utils import keccak
    return keccak(_abi_encode_bytes32_address(
        _hash_string("SAVED_FUNDING_FACTOR_PER_SECOND"), market,
    ))


def _borrowing_factor_per_second_key(market: str, is_long: bool) -> bytes:
    """Per-side per-second borrowing factor, uint256, scaled 1e30."""
    from eth_utils import keccak
    return keccak(_abi_encode_bytes32_address_bool(
        _hash_string("BORROWING_FACTOR_PER_SECOND"), market, is_long,
    ))


# Minimal ABIs.

_GMX_READER_MIN_ABI = [
    {
        "name": "getMarkets",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "dataStore", "type": "address"},
            {"name": "start", "type": "uint256"},
            {"name": "end", "type": "uint256"},
        ],
        "outputs": [{
            "name": "markets",
            "type": "tuple[]",
            "components": [
                {"name": "marketToken", "type": "address"},
                {"name": "indexToken", "type": "address"},
                {"name": "longToken", "type": "address"},
                {"name": "shortToken", "type": "address"},
            ],
        }],
    },
]


_GMX_DATA_STORE_MIN_ABI = [
    {
        "name": "getUint",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "key", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "getInt",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "key", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "int256"}],
    },
]


_ERC20_MIN_ABI = [
    {
        "name": "symbol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    },
]
