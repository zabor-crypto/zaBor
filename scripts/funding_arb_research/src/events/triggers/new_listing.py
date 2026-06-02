"""New-listing funding-spike trigger.

Fires when:
  1. The coin has at least one perp listed within ``max_age_days`` of
     ``t`` on any venue in the listing archive, AND
  2. The annualized funding |apr| on at least one available venue
     exceeds ``abs_threshold_apr``.

The hypothesis: in the first hours-to-weeks of a new perp listing,
order-flow imbalance creates wider and more persistent funding than
on mature contracts (market makers haven't tightened spreads;
directional retail flow dominates). The cross-DEX dispersion verdict
already showed funding spreads exist but mean-revert too fast on
mature pairs — new listings should mean-revert slower.

Look-ahead discipline: trigger reads listing_archive (a static
"as-of-now" snapshot) but only counts listings whose
``launch_ts_utc < t``. Funding panels are sliced strictly < t by the
engine.

Anchor for routing: the **listing venue** that's carrying the extreme
funding is the receiver leg. If multiple venues list the coin and
multiple are above threshold, the highest-|apr| listing-venue wins.
The router's `_route_anchored` reads ``metadata["listing_venue"]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..event import FundingEvent
from .base import MarketSnapshot, Trigger

_HOURS_PER_YEAR = 365.0 * 24.0


@dataclass
class NewListingTrigger(Trigger):
    coins: list[str]
    listing_archive: pd.DataFrame                      # cols: venue, base_asset, launch_ts_utc
    max_age_days: int = 30
    abs_threshold_apr: float = 0.50
    hedge_venues: list[str] = field(default_factory=lambda: ["binance", "bybit", "okx", "hyperliquid"])
    name: str = "new_listing_spike"
    _next_check_cache: Optional[pd.Timestamp] = None
    # Pre-bucket the listing archive once for cheap per-coin lookups.
    _listings_by_coin: dict[str, list[tuple[str, pd.Timestamp]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.listing_archive is None or self.listing_archive.empty:
            return
        df = self.listing_archive.copy()
        df["launch_ts_utc"] = pd.to_datetime(df["launch_ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["launch_ts_utc"])
        for coin, sub in df.groupby("base_asset"):
            self._listings_by_coin[str(coin)] = [
                (str(r.venue), r.launch_ts_utc) for r in sub.itertuples(index=False)
            ]

    def next_check_at(self, t: pd.Timestamp) -> pd.Timestamp:
        if self._next_check_cache is not None and self._next_check_cache > t:
            return self._next_check_cache
        return t + pd.Timedelta(hours=1)

    def evaluate(self, snap: MarketSnapshot) -> list[FundingEvent]:
        events: list[FundingEvent] = []
        next_settle_candidates: list[pd.Timestamp] = []
        max_age = pd.Timedelta(days=self.max_age_days)

        for coin in self.coins:
            listings = self._listings_by_coin.get(coin, [])
            recent = [(v, ts) for v, ts in listings if ts < snap.t and (snap.t - ts) <= max_age]
            if not recent:
                continue

            # Per-venue latest realized APR.
            apr_by_venue: dict[str, float] = {}
            for v in {*[lv for lv, _ in recent], *self.hedge_venues}:
                sym = self._symbol(v, coin, snap)
                panel = snap.funding_panels.get((v, sym))
                if panel is None or panel.empty:
                    continue
                row = panel.iloc[-1]
                rate = row.get("funding_rate")
                if pd.isna(rate):
                    continue
                cad = self._cadence(v, sym, snap)
                apr_by_venue[v] = float(rate) * (_HOURS_PER_YEAR / cad)
                next_settle_candidates.append(row["timestamp_utc"] + pd.Timedelta(hours=cad))

            if not apr_by_venue:
                continue

            # Listing venue with the largest |apr| above threshold becomes the anchor.
            listing_venues_with_data = {v for v, _ in recent if v in apr_by_venue}
            qualifying = [(v, apr_by_venue[v]) for v in listing_venues_with_data
                          if abs(apr_by_venue[v]) >= self.abs_threshold_apr]
            if not qualifying:
                continue
            listing_venue, listing_apr = max(qualifying, key=lambda x: abs(x[1]))

            # Listing age in hours (newest listing among qualifying)
            age_h = min(
                (snap.t - ts).total_seconds() / 3600.0 for v, ts in recent if v == listing_venue
            )
            confidence = min(1.0, abs(listing_apr) / (3.0 * self.abs_threshold_apr))

            events.append(FundingEvent(
                coin=coin,
                timestamp_utc=snap.t,
                trigger_type="new_listing_spike",
                venue_signal=apr_by_venue,
                expected_funding_pnl_apr=abs(listing_apr),
                confidence=confidence,
                metadata={
                    "listing_venue": listing_venue,
                    "listing_age_hours": age_h,
                    "abs_threshold_apr": self.abs_threshold_apr,
                    "max_age_days": self.max_age_days,
                },
            ))

        if next_settle_candidates:
            self._next_check_cache = min(next_settle_candidates)
        return events

    # ---- helpers ------------------------------------------------------ #

    def _cadence(self, venue: str, symbol: str, snap: MarketSnapshot) -> int:
        cm = snap.contracts_meta
        if cm is not None and not cm.empty:
            row = cm[(cm["venue"] == venue) & (cm["symbol"] == symbol)]
            if not row.empty and pd.notna(row.iloc[0].get("fund_interval_hours")):
                return int(row.iloc[0]["fund_interval_hours"])
        return 1 if venue == "hyperliquid" else 8

    def _symbol(self, venue: str, coin: str, snap: MarketSnapshot) -> str:
        cm = snap.contracts_meta
        if cm is not None and not cm.empty:
            row = cm[(cm["venue"] == venue) & (cm["base_asset"] == coin)]
            if not row.empty:
                return row.iloc[0]["symbol"]
        if venue == "hyperliquid":
            return coin
        if venue == "okx":
            return f"{coin}-USDT-SWAP"
        return f"{coin}USDT"
