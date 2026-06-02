"""Local order book builder with desync detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple

from v1.book.desync_guard import DesyncGuard
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import DepthDeltaEvent


@dataclass
class BookState:
    symbol: str
    bids: Dict[Decimal, Decimal] = field(default_factory=dict)
    asks: Dict[Decimal, Decimal] = field(default_factory=dict)
    last_update_id: Optional[int] = None
    last_exchange_ts_ms: int = 0


class LocalOrderBookEngine:
    def __init__(
        self,
        symbols: Iterable[str],
        stale_after_ms: int = 2000,
        max_depth_levels: int = 0,
    ) -> None:
        self.states: Dict[str, BookState] = {s: BookState(symbol=s) for s in symbols}
        self.guards: Dict[str, DesyncGuard] = {s: DesyncGuard(s, stale_after_ms=stale_after_ms) for s in symbols}
        # 0 = unlimited (existing behaviour); positive value caps each side.
        self._max_depth_levels = int(max_depth_levels)

    def _trim_book(self, state: BookState) -> None:
        """Prune the book to max_depth_levels per side (best prices only)."""
        n = self._max_depth_levels
        if n <= 0:
            return
        if len(state.bids) > n:
            # Keep the n highest bids.
            top_bids = sorted(state.bids.keys(), reverse=True)[:n]
            state.bids = {px: state.bids[px] for px in top_bids}
        if len(state.asks) > n:
            # Keep the n lowest asks.
            top_asks = sorted(state.asks.keys())[:n]
            state.asks = {px: state.asks[px] for px in top_asks}

    def apply_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        state = self.states[snapshot.symbol]
        state.bids = {px: qty for px, qty in snapshot.bids if qty > 0}
        state.asks = {px: qty for px, qty in snapshot.asks if qty > 0}
        state.last_update_id = snapshot.last_update_id
        state.last_exchange_ts_ms = snapshot.exchange_ts_ms
        self._trim_book(state)
        self.guards[snapshot.symbol].mark_snapshot(snapshot.exchange_ts_ms)

    def _apply_levels(self, levels: List[List[Decimal]], side_book: Dict[Decimal, Decimal]) -> None:
        for px, qty in levels:
            if qty <= 0:
                side_book.pop(px, None)
            else:
                side_book[px] = qty

    def apply_depth_delta(self, event: DepthDeltaEvent) -> bool:
        symbol = event.symbol
        state = self.states[symbol]
        guard = self.guards[symbol]

        if state.last_update_id is None:
            guard.mark_desync("no_snapshot")
            return False

        if event.final_update_id <= state.last_update_id:
            return True

        has_prev = event.prev_final_update_id is not None
        bridge_event = event.first_update_id <= (state.last_update_id + 1) <= event.final_update_id

        if has_prev and event.prev_final_update_id != state.last_update_id and not bridge_event:
            guard.mark_desync(
                f"prev_final_mismatch:{event.prev_final_update_id}!={state.last_update_id}"
            )
            return False

        if (not has_prev) and event.first_update_id > state.last_update_id + 1:
            guard.mark_desync(f"gap:{state.last_update_id}->{event.first_update_id}")
            return False

        # valid contiguous (or overlapping) sequence
        self._apply_levels(event.bid_deltas, state.bids)
        self._apply_levels(event.ask_deltas, state.asks)
        self._trim_book(state)
        state.last_update_id = event.final_update_id
        state.last_exchange_ts_ms = event.exchange_ts_ms
        guard.mark_delta_ok(event.exchange_ts_ms)
        return True

    def top_of_book(self, symbol: str) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
        state = self.states[symbol]
        if not state.bids or not state.asks:
            return None
        best_bid_px = max(state.bids.keys())
        best_ask_px = min(state.asks.keys())
        return best_bid_px, state.bids[best_bid_px], best_ask_px, state.asks[best_ask_px]

    def spread_bps(self, symbol: str) -> Optional[float]:
        tob = self.top_of_book(symbol)
        if tob is None:
            return None
        best_bid, _, best_ask, _ = tob
        mid = (best_bid + best_ask) / Decimal("2")
        if mid <= 0:
            return None
        spread = best_ask - best_bid
        return float((spread / mid) * Decimal("10000"))

    def mid_price(self, symbol: str) -> Optional[Decimal]:
        tob = self.top_of_book(symbol)
        if tob is None:
            return None
        best_bid, _, best_ask, _ = tob
        return (best_bid + best_ask) / Decimal("2")

    def refresh_staleness(self, symbol: str, now_exchange_ts_ms: int) -> None:
        self.guards[symbol].refresh_staleness(now_exchange_ts_ms)

    def is_healthy(self, symbol: str) -> bool:
        return self.guards[symbol].healthy

    def desync_reason(self, symbol: str) -> str:
        return self.guards[symbol].reason
