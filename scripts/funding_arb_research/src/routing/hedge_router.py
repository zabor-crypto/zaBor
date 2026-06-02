"""Hedge router: ``FundingEvent`` → (long_venue, short_venue, hedge_mode).

The trigger tells us *what* funding edge to chase; the router decides
*how* to construct the delta-neutral pair. It picks the venue pair
that:

  1. Is fundable on both legs (``can_long_perp`` / ``can_short_perp``).
  2. Has the venue carrying the extreme funding on the side that
     *receives* funding (long when funding<0, short when funding>0).
  3. Hedges on the cheapest available venue that lists the same coin
     and is in the strategy's allowed venue universe.

Rule recap from the locked design:
  - Bitget Mode-2 (spot short) DISABLED. A negative-funding Bitget event
    → Bitget *long* (receives funding) hedged by a perp *short* on
    Binance/Bybit/OKX/HL — never spot.
  - Cross-DEX dispersion events are already a venue pair from the
    trigger; the router just rubber-stamps them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..events.event import FundingEvent
from ..utils.logging import get_logger
from .venue_capabilities import VenueCapabilities

HedgeMode = Literal["perp_perp", "cross_dex_perp_perp"]

_LOG = get_logger("funding_arb.routing.hedge_router")


@dataclass(frozen=True)
class RoutedPair:
    """Output of the router. Both legs are perps."""
    long_venue: str
    short_venue: str
    hedge_mode: HedgeMode
    long_funding_apr: float
    short_funding_apr: float
    expected_carry_apr: float                 # long.f - short.f (signed correctly)


@dataclass
class HedgeRouter:
    caps: VenueCapabilities
    allowed_hedge_venues: list[str]           # e.g. ["binance","bybit","okx","hyperliquid"]

    def route(self, event: FundingEvent) -> Optional[RoutedPair]:
        if event.trigger_type == "cross_dex_dispersion":
            return self._route_dispersion(event)
        # Anchor venue determined by trigger type. The trigger picked it for
        # a reason (the extreme funding lives there); the router must not
        # pivot to a different venue just because its apr is larger in magnitude.
        anchor = {
            "bitget_extreme": "bitget",
            "new_listing_spike": event.metadata.get("listing_venue"),
        }.get(event.trigger_type)
        return self._route_anchored(event, anchor)

    # ------------------------------------------------------------------ #

    def _route_anchored(self, event: FundingEvent, anchor: Optional[str]) -> Optional[RoutedPair]:
        """Receiver leg = trigger's anchor venue. Hedge leg = whichever allowed
        venue maximizes *signed expected carry* net of fees.

        Earned funding pnl on the position over time = ``short_apr − long_apr``.
        A hedge venue picked solely on cheap fees is strictly worse than one
        that delivers positive funding contribution. We therefore enumerate
        candidates and pick by ``carry_apr − round_trip_fee_apr_proxy``.
        """
        sigs = event.venue_signal
        if not sigs:
            return None
        if anchor and anchor in sigs:
            recv_venue = anchor
        else:
            recv_venue = max(sigs.keys(), key=lambda v: abs(sigs[v]))
        recv_apr = sigs[recv_venue]
        # Receiver direction: long when its apr is negative, short when positive.
        recv_dir = +1 if recv_apr < 0 else -1

        candidates = [v for v in self.allowed_hedge_venues
                      if v != recv_venue and v in self.caps and v in sigs]
        candidates = [v for v in candidates
                      if self.caps.get(v).can_long_perp and self.caps.get(v).can_short_perp]
        if not candidates:
            return None

        # Score each candidate by net carry minus a rough fee proxy.
        # The hedge leg sign is opposite to the receiver, so:
        #   if recv long  (recv_apr<0): hedge is short → short_apr=hedge_apr, long_apr=recv_apr
        #   if recv short (recv_apr>0): hedge is long  → long_apr=hedge_apr,  short_apr=recv_apr
        best: Optional[tuple[float, str, float]] = None
        for hv in candidates:
            hv_apr = sigs[hv]
            if recv_dir == +1:
                long_apr, short_apr = recv_apr, hv_apr
            else:
                long_apr, short_apr = hv_apr, recv_apr
            carry = short_apr - long_apr
            # Convert per-leg round-trip fee to APR proxy over a 7d hold.
            fee_apr = (
                2 * (self.caps.get(recv_venue).maker_fee_bps + self.caps.get(hv).maker_fee_bps)
                * 1e-4 * (365.0 / 7.0)
            )
            score = carry - fee_apr
            if best is None or score > best[0]:
                best = (score, hv, carry)

        if best is None or best[0] <= 0:
            return None  # no hedge gives positive net expected carry
        hedge_venue = best[1]
        hedge_apr = sigs[hedge_venue]
        if recv_dir == +1:
            long_venue, short_venue = recv_venue, hedge_venue
            long_apr, short_apr = recv_apr, hedge_apr
        else:
            long_venue, short_venue = hedge_venue, recv_venue
            long_apr, short_apr = hedge_apr, recv_apr
        return RoutedPair(long_venue=long_venue, short_venue=short_venue,
                          hedge_mode="perp_perp",
                          long_funding_apr=long_apr, short_funding_apr=short_apr,
                          expected_carry_apr=best[2])

    def _route_dispersion(self, event: FundingEvent) -> Optional[RoutedPair]:
        sigs = event.venue_signal
        if len(sigs) < 2:
            return None
        # Pick (max apr, min apr): short the high, long the low.
        short_venue = max(sigs.keys(), key=lambda v: sigs[v])
        long_venue = min(sigs.keys(), key=lambda v: sigs[v])
        if short_venue == long_venue:
            return None
        short_apr, long_apr = sigs[short_venue], sigs[long_venue]
        carry = short_apr - long_apr
        is_dex = self.caps.get(short_venue).is_dex or self.caps.get(long_venue).is_dex
        return RoutedPair(long_venue=long_venue, short_venue=short_venue,
                          hedge_mode="cross_dex_perp_perp" if is_dex else "perp_perp",
                          long_funding_apr=long_apr, short_funding_apr=short_apr,
                          expected_carry_apr=carry)
