"""Per-venue capability registry.

The hedge router needs to know, for each venue + side (long/short),
whether a delta-neutral leg is fundable, what fee tier we assume, and
what the venue's funding-publication cadence looks like. Defaults here
match the public-rate side of each venue (no VIP tier assumed). Bump
in ``config/venue_capabilities.yaml`` if the user is on a tighter tier.

Bitget Mode-2 (negative-funding hedged with spot short) is **disabled**
by design — see ``event_funding_capture_v1_design.md``. Negative
Bitget funding routes to a perp short on a different venue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..utils.io import config_dir
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.routing.venue_capabilities")


@dataclass(frozen=True)
class VenueCap:
    venue: str
    can_long_perp: bool = True
    can_short_perp: bool = True
    is_dex: bool = False
    maker_fee_bps: float = 2.0                # round-trip per leg uses 2× this
    taker_fee_bps: float = 5.0
    funding_pub_latency_ms: int = 5_000        # premium-index update cadence
    fund_interval_hours_default: int = 8
    free_margin_proxy_buffer: float = 0.30     # used by margin proxy (hedge venues)
    notes: str = ""


# Documented defaults; loader can override these from a YAML config.
_DEFAULTS: dict[str, VenueCap] = {
    "bitget":      VenueCap("bitget",      maker_fee_bps=2.0, taker_fee_bps=6.0,
                            fund_interval_hours_default=4),
    "binance":     VenueCap("binance",     maker_fee_bps=2.0, taker_fee_bps=5.0),
    "bybit":       VenueCap("bybit",       maker_fee_bps=2.0, taker_fee_bps=5.5),
    "okx":         VenueCap("okx",         maker_fee_bps=2.0, taker_fee_bps=5.0),
    "hyperliquid": VenueCap("hyperliquid", maker_fee_bps=1.5, taker_fee_bps=3.5,
                            fund_interval_hours_default=1, is_dex=True,
                            funding_pub_latency_ms=1_000),
    "dydx":        VenueCap("dydx",        maker_fee_bps=0.0, taker_fee_bps=5.0,
                            is_dex=True, fund_interval_hours_default=1),
    "drift":       VenueCap("drift",       maker_fee_bps=0.0, taker_fee_bps=10.0,
                            is_dex=True, fund_interval_hours_default=1),
    "aevo":        VenueCap("aevo",        maker_fee_bps=3.0, taker_fee_bps=5.0,
                            is_dex=True),
}


class VenueCapabilities:
    """Registry that the router and engine query. Loadable from YAML."""

    def __init__(self, by_venue: Optional[dict[str, VenueCap]] = None) -> None:
        self._by_venue = by_venue or dict(_DEFAULTS)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VenueCapabilities":
        path = path or (config_dir() / "venue_capabilities.yaml")
        if not path.exists():
            _LOG.info("no %s — using built-in defaults", path)
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        merged = dict(_DEFAULTS)
        for venue, override in (doc.get("venues") or {}).items():
            base = merged.get(venue) or VenueCap(venue=venue)
            merged[venue] = VenueCap(**{**base.__dict__, "venue": venue, **(override or {})})
        return cls(merged)

    def get(self, venue: str) -> VenueCap:
        cap = self._by_venue.get(venue)
        if cap is None:
            _LOG.warning("unknown venue %r — using permissive defaults", venue)
            return VenueCap(venue=venue)
        return cap

    def __contains__(self, venue: str) -> bool:
        return venue in self._by_venue

    def all_venues(self) -> list[str]:
        return list(self._by_venue.keys())
