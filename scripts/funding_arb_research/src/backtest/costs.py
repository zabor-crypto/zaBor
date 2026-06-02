"""Trading-cost model: fees + slippage. Configurable per venue via fees.yaml."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.io import config_dir, load_yaml


@dataclass(frozen=True)
class FeeSchedule:
    maker_bps: float
    taker_bps: float
    default_role: str = "taker"

    def per_side_bps(self, role: Optional[str] = None) -> float:
        r = role or self.default_role
        return self.maker_bps if r == "maker" else self.taker_bps


@dataclass(frozen=True)
class SlippageConfig:
    min_slippage_bps: float = 1.0
    k: float = 0.5
    fallback_bps: float = 5.0


class CostModel:
    """Costs in basis points. All conversions to PnL are done by the caller.

    bps unit: 1 bp = 0.0001 of notional.
    """

    def __init__(
        self,
        fees: dict[str, FeeSchedule],
        slippage: SlippageConfig,
        gmx_open_close_bps: float = 7.0,
    ):
        self._fees = fees
        self._slip = slippage
        self.gmx_open_close_bps = gmx_open_close_bps

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "CostModel":
        path = path or (config_dir() / "fees.yaml")
        cfg = load_yaml(path)
        fees: dict[str, FeeSchedule] = {}
        for venue in ("bitget", "binance", "bybit", "okx", "hyperliquid"):
            v = cfg.get(venue, {}) or {}
            fees[venue] = FeeSchedule(
                maker_bps=float(v.get("maker_bps", 2.0)),
                taker_bps=float(v.get("taker_bps", 5.0)),
                default_role=str(v.get("default_role", "taker")),
            )
        gmx_oc = float((cfg.get("gmx") or {}).get("open_close_bps", 7.0))
        slip_cfg = cfg.get("slippage") or {}
        slip = SlippageConfig(
            min_slippage_bps=float(slip_cfg.get("min_slippage_bps", 1.0)),
            k=float(slip_cfg.get("k", 0.5)),
            fallback_bps=float(slip_cfg.get("fallback_bps", 5.0)),
        )
        return cls(fees=fees, slippage=slip, gmx_open_close_bps=gmx_oc)

    # --------------------------------------------------------------------- #
    # Fees
    # --------------------------------------------------------------------- #

    def fee_bps(self, venue: str, role: Optional[str] = None) -> float:
        if venue == "gmx" or venue == "gmx_v2":
            return self.gmx_open_close_bps
        if venue not in self._fees:
            raise KeyError(f"unknown venue '{venue}' in cost model")
        return self._fees[venue].per_side_bps(role)

    def round_trip_fee_bps(self, venue: str, role: Optional[str] = None) -> float:
        return 2.0 * self.fee_bps(venue, role)

    # --------------------------------------------------------------------- #
    # Slippage
    # --------------------------------------------------------------------- #

    def slippage_bps(
        self,
        order_notional: float,
        depth_notional: Optional[float] = None,
    ) -> float:
        """Conservative slippage estimate.

        slippage_bps = max(min, k * sqrt(order / depth)) * 1e4
        Where k is a unitless constant. Yields bps, not fraction.
        """
        if depth_notional is None or depth_notional <= 0 or order_notional <= 0:
            return self._slip.fallback_bps
        ratio = order_notional / depth_notional
        if ratio <= 0:
            return self._slip.min_slippage_bps
        slip = self._slip.k * math.sqrt(ratio) * 1e4
        return max(self._slip.min_slippage_bps, slip)

    # --------------------------------------------------------------------- #
    # Helpers used by strategies
    # --------------------------------------------------------------------- #

    def entry_cost_bps(self, venue: str, *, role: Optional[str] = None,
                       order_notional: float = 0.0,
                       depth_notional: Optional[float] = None) -> float:
        return self.fee_bps(venue, role) + self.slippage_bps(order_notional, depth_notional)

    def exit_cost_bps(self, venue: str, *, role: Optional[str] = None,
                      order_notional: float = 0.0,
                      depth_notional: Optional[float] = None) -> float:
        return self.fee_bps(venue, role) + self.slippage_bps(order_notional, depth_notional)

    def two_leg_round_trip_bps(
        self,
        venue_a: str,
        venue_b: str,
        *,
        notional: float = 0.0,
        depth_a: Optional[float] = None,
        depth_b: Optional[float] = None,
    ) -> float:
        return (
            self.entry_cost_bps(venue_a, order_notional=notional, depth_notional=depth_a)
            + self.exit_cost_bps(venue_a, order_notional=notional, depth_notional=depth_a)
            + self.entry_cost_bps(venue_b, order_notional=notional, depth_notional=depth_b)
            + self.exit_cost_bps(venue_b, order_notional=notional, depth_notional=depth_b)
        )
