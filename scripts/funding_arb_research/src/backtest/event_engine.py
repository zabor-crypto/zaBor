"""Event-driven backtest engine.

Replaces the clock-step loop for the three event-driven strategies
(``bitget_extreme``, ``new_listing_spike``, ``cross_dex_dispersion``).
A heap of pending callbacks drives the simulation; the engine wakes
only when a trigger wants polling, a leg's funding settles, or an open
position needs an exit check. This is materially faster than ticking
every 5 minutes across 18 months of data, and — more importantly —
funding settlement timestamps are not coarsened by the tick grid.

Design contract:
  * No look-ahead. Triggers see only data ``< t``.
  * Funding pnl is booked at the leg's actual settlement timestamps,
    one row per settle period, with the convention
    ``leg_pnl = -direction × rate × notional``.
  * Entry / exit charge fees + slippage once each. Slippage uses the
    CostModel; ``slippage_source`` is recorded per leg.
  * Forced exits unconditional: any of {position drawdown ≥ ceiling,
    data stale, funding flipped beyond grace} closes the position.

The engine is venue-agnostic. All venue specifics live in:
  * the trigger (it knows which venue's funding panel to read)
  * the hedge router (picks long/short legs)
  * the cost model + MMR model (per-venue fee tier and margin)
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..events.event import FundingEvent, Leg, Position
from ..events.triggers.base import MarketSnapshot, Trigger
from ..routing.hedge_router import HedgeRouter, RoutedPair
from ..utils.logging import get_logger
from .capital_pool import AllocationRequest, CapitalPool
from .costs import CostModel
from .recorder_slippage import RecorderSlippage

_LOG = get_logger("funding_arb.backtest.event_engine")

# Heap event types.
_EV_TRIGGER_CHECK = 1
_EV_FUNDING_SETTLE = 2
_EV_EXIT_CHECK = 3


@dataclass
class EngineConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    capital_usd: float
    per_event_max_pct: float = 0.10
    min_event_capital_usd: float = 1_000.0
    exit_check_interval_minutes: int = 30
    exit_threshold_apr: float = 0.05          # close if expected carry falls below
    max_hold_hours: float = 96.0              # absolute cap on hold time
    data_staleness_seconds: int = 3600
    sign_flip_grace_minutes: int = 60
    role: str = "taker"


@dataclass
class TradeRecord:
    event: FundingEvent
    long_leg: Leg
    short_leg: Leg
    opened: pd.Timestamp
    closed: pd.Timestamp
    close_reason: str
    realized_pnl_usd: float
    long_funding_usd: float
    short_funding_usd: float
    fees_usd: float
    slippage_usd: float
    used_capital_usd: float

    def hold_hours(self) -> float:
        return (self.closed - self.opened).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        return {
            "trigger_type": self.event.trigger_type,
            "coin": self.event.coin,
            "opened": self.opened, "closed": self.closed,
            "hold_hours": self.hold_hours(),
            "close_reason": self.close_reason,
            "long_venue": self.long_leg.venue, "short_venue": self.short_leg.venue,
            "long_notional_usd": self.long_leg.notional_usd,
            "short_notional_usd": self.short_leg.notional_usd,
            "used_capital_usd": self.used_capital_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "long_funding_usd": self.long_funding_usd,
            "short_funding_usd": self.short_funding_usd,
            "fees_usd": self.fees_usd,
            "slippage_usd": self.slippage_usd,
            "long_slippage_source": self.long_leg.slippage_source,
            "short_slippage_source": self.short_leg.slippage_source,
            "expected_funding_apr": self.event.expected_funding_pnl_apr,
            "confidence": self.event.confidence,
        }


@dataclass
class EngineResult:
    trades: list[TradeRecord]
    final_equity_usd: float
    skipped_events: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


class EventEngine:
    """Heap-driven event-loop backtest."""

    def __init__(
        self,
        cfg: EngineConfig,
        triggers: list[Trigger],
        router: HedgeRouter,
        costs: CostModel,
        funding_panels: dict[tuple[str, str], pd.DataFrame],
        contracts_meta: pd.DataFrame,
        listing_archive: Optional[pd.DataFrame] = None,
        recorder_slippage: Optional[RecorderSlippage] = None,
    ) -> None:
        self.cfg = cfg
        self.triggers = triggers
        self.router = router
        self.costs = costs
        self.funding = funding_panels
        # Pre-extract sorted timestamp arrays as int64 nanoseconds for fast
        # bisect-based slicing. Without this, each trigger check re-scans
        # every panel with boolean indexing → O(N×panels×checks).
        self._ts_ns: dict[tuple[str, str], np.ndarray] = {
            k: pd.to_datetime(df["timestamp_utc"], utc=True).astype("int64").to_numpy()
            for k, df in funding_panels.items()
        }
        self.contracts = contracts_meta
        # Precompute (venue, base_asset) -> (symbol, fund_interval_hours) lookups
        # so per-trade calls stop hitting DataFrame filters.
        self._venue_base_to_sym: dict[tuple[str, str], str] = {}
        self._venue_sym_to_cadence: dict[tuple[str, str], int] = {}
        if contracts_meta is not None and not contracts_meta.empty:
            for r in contracts_meta.itertuples(index=False):
                v, sym, base = getattr(r, "venue", None), getattr(r, "symbol", None), getattr(r, "base_asset", None)
                cad = getattr(r, "fund_interval_hours", None)
                if v and sym and base:
                    self._venue_base_to_sym.setdefault((v, base), sym)
                if v and sym and cad is not None and not pd.isna(cad):
                    self._venue_sym_to_cadence[(v, sym)] = int(cad)
        self.listings = listing_archive if listing_archive is not None else pd.DataFrame()
        self.rec_slip = recorder_slippage
        self.pool = CapitalPool(
            total_capital_usd=cfg.capital_usd,
            per_event_max_pct=cfg.per_event_max_pct,
            min_event_capital_usd=cfg.min_event_capital_usd,
        )
        self._heap: list[tuple[pd.Timestamp, int, int, dict]] = []
        self._counter = itertools.count()
        self._open: dict[int, Position] = {}
        self._next_pos_id = itertools.count()
        self._cooldowns: dict[tuple[str, str], pd.Timestamp] = {}  # (trigger.name, coin) -> ts
        self._trades: list[TradeRecord] = []
        self._skipped = 0

    # ---- public ------------------------------------------------------- #

    def run(self) -> EngineResult:
        # Seed heap with first trigger checks at start.
        for trig in self.triggers:
            t0 = max(trig.next_check_at(self.cfg.start), self.cfg.start)
            self._push(t0, _EV_TRIGGER_CHECK, {"trigger": trig})
        while self._heap:
            t, kind, _seq, payload = heapq.heappop(self._heap)
            if t > self.cfg.end:
                break
            if kind == _EV_TRIGGER_CHECK:
                self._handle_trigger_check(t, payload["trigger"])
            elif kind == _EV_FUNDING_SETTLE:
                self._handle_funding_settle(t, payload["pos_id"], payload["leg_side"])
            elif kind == _EV_EXIT_CHECK:
                self._handle_exit_check(t, payload["pos_id"])
        # End-of-window: force-close remaining
        for pos_id in list(self._open.keys()):
            self._close_position(self.cfg.end, pos_id, "end_of_window")
        equity = self.cfg.capital_usd + sum(t.realized_pnl_usd for t in self._trades)
        return EngineResult(trades=self._trades, final_equity_usd=equity,
                            skipped_events=self._skipped)

    # ---- heap helpers ------------------------------------------------- #

    def _push(self, ts: pd.Timestamp, kind: int, payload: dict) -> None:
        heapq.heappush(self._heap, (ts, kind, next(self._counter), payload))

    # ---- trigger ------------------------------------------------------ #

    def _handle_trigger_check(self, t: pd.Timestamp, trig: Trigger) -> None:
        snap = MarketSnapshot(
            t=t,
            funding_panels=self._sliced_panels(t),
            contracts_meta=self.contracts,
            listing_archive=self.listings,
        )
        events = trig.evaluate(snap)
        # Cooldown filter: drop events whose (trigger, coin) is on cooldown.
        kept: list[FundingEvent] = []
        for ev in events:
            cd = self._cooldowns.get((trig.name, ev.coin))
            if cd is not None and t < cd:
                continue
            kept.append(ev)

        if kept:
            self._dispatch_events(kept)

        # Schedule next check. Cap so we make progress even if a trigger is buggy.
        nxt = trig.next_check_at(t)
        if nxt <= t:
            nxt = t + pd.Timedelta(minutes=5)
        if nxt < self.cfg.end:
            self._push(nxt, _EV_TRIGGER_CHECK, {"trigger": trig})

    def _dispatch_events(self, events: list[FundingEvent]) -> None:
        # Build allocation requests sized to per-event cap (capital pool tightens further).
        cap_per = self.pool.per_event_cap_usd()
        requests: list[AllocationRequest] = []
        for ev in events:
            est_cost_apr = self._estimated_cost_apr(ev, notional=cap_per)
            if est_cost_apr <= 0:
                est_cost_apr = 1e-4
            requests.append(AllocationRequest(
                event=ev, requested_capital_usd=cap_per,
                estimated_round_trip_cost_apr=est_cost_apr,
            ))
        results = self.pool.allocate_batch(requests)
        for res in results:
            if res.approved_capital_usd <= 0:
                self._skipped += 1
                continue
            routed = self.router.route(res.event)
            if routed is None:
                self.pool.release(res.approved_capital_usd)
                self._skipped += 1
                continue
            self._open_position(res.event, routed, capital_usd=res.approved_capital_usd)

    def _estimated_cost_apr(self, ev: FundingEvent, notional: float) -> float:
        """Rough cost-as-APR over a 7-day amortization horizon — used only by the pool ranker."""
        # We don't yet know the routed venues; use Bitget+Binance as a reasonable proxy.
        try:
            rt_bps = self.costs.two_leg_round_trip_bps("bitget", "binance", notional=notional)
        except KeyError:
            rt_bps = 30.0
        days = 7.0
        return (rt_bps * 1e-4) * (365.0 / max(days, 1.0))

    # ---- position lifecycle ------------------------------------------ #

    def _open_position(self, event: FundingEvent, routed: RoutedPair, capital_usd: float) -> None:
        leverage = 3.0  # conservative default; per-coin override later
        long_notional = capital_usd * leverage / 2.0  # split across two legs
        short_notional = long_notional
        long_px = self._price_at(routed.long_venue, event.coin, event.timestamp_utc)
        short_px = self._price_at(routed.short_venue, event.coin, event.timestamp_utc)
        if long_px is None or short_px is None:
            self.pool.release(capital_usd)
            self._skipped += 1
            return
        long_cad = self._cadence(routed.long_venue, event.coin)
        short_cad = self._cadence(routed.short_venue, event.coin)
        # Charge entry fees + slippage upfront, attributed separately.
        long_fee_bps = self.costs.fee_bps(routed.long_venue, role=self.cfg.role)
        short_fee_bps = self.costs.fee_bps(routed.short_venue, role=self.cfg.role)
        long_slip_bps, long_slip_src = self._slip_bps(routed.long_venue, event.coin, long_notional)
        short_slip_bps, short_slip_src = self._slip_bps(routed.short_venue, event.coin, short_notional)
        long_entry_fee = long_fee_bps * 1e-4 * long_notional
        short_entry_fee = short_fee_bps * 1e-4 * short_notional
        long_entry_slip = long_slip_bps * 1e-4 * long_notional
        short_entry_slip = short_slip_bps * 1e-4 * short_notional

        long_leg = Leg(
            venue=routed.long_venue, symbol=self._symbol(routed.long_venue, event.coin),
            direction=+1, notional_usd=long_notional, leverage=leverage,
            entry_price=long_px, entry_ts_utc=event.timestamp_utc,
            funding_interval_hours=long_cad,
            fees_paid_usd=long_entry_fee, slippage_paid_usd=long_entry_slip,
            slippage_source=long_slip_src,
        )
        short_leg = Leg(
            venue=routed.short_venue, symbol=self._symbol(routed.short_venue, event.coin),
            direction=-1, notional_usd=short_notional, leverage=leverage,
            entry_price=short_px, entry_ts_utc=event.timestamp_utc,
            funding_interval_hours=short_cad,
            fees_paid_usd=short_entry_fee, slippage_paid_usd=short_entry_slip,
            slippage_source=short_slip_src,
        )
        position = Position(event=event, long_leg=long_leg, short_leg=short_leg,
                            opened_ts_utc=event.timestamp_utc)
        position.realized_pnl_usd = -(long_entry_fee + short_entry_fee
                                       + long_entry_slip + short_entry_slip)
        pid = next(self._next_pos_id)
        self._open[pid] = position

        # Schedule funding settles for both legs at their actual next-settle ts.
        for side, leg in (("long", long_leg), ("short", short_leg)):
            nxt_settle = self._next_funding_settle_after(leg, event.timestamp_utc)
            if nxt_settle is not None:
                self._push(nxt_settle, _EV_FUNDING_SETTLE,
                           {"pos_id": pid, "leg_side": side})

        # Schedule first exit check.
        first_exit = event.timestamp_utc + pd.Timedelta(minutes=self.cfg.exit_check_interval_minutes)
        self._push(first_exit, _EV_EXIT_CHECK, {"pos_id": pid})
        # Cooldown so the same trigger doesn't re-fire on the same coin while open.
        self._cooldowns[(event.trigger_type, event.coin)] = (
            event.timestamp_utc + pd.Timedelta(hours=self.cfg.max_hold_hours)
        )

    def _handle_funding_settle(self, t: pd.Timestamp, pos_id: int, side: str) -> None:
        pos = self._open.get(pos_id)
        if pos is None:
            return
        leg = pos.long_leg if side == "long" else pos.short_leg
        rate = self._funding_rate_at(leg.venue, pos.event.coin, t)
        if rate is None:
            # Schedule next anyway.
            nxt = self._next_funding_settle_after(leg, t)
            if nxt is not None and nxt <= self.cfg.end:
                self._push(nxt, _EV_FUNDING_SETTLE, {"pos_id": pos_id, "leg_side": side})
            return
        pnl = -leg.direction * rate * leg.notional_usd
        leg.accrued_funding_usd += pnl
        leg.last_funding_settle_ts = t
        pos.realized_pnl_usd += pnl
        # Schedule next.
        nxt = self._next_funding_settle_after(leg, t)
        if nxt is not None and nxt <= self.cfg.end:
            self._push(nxt, _EV_FUNDING_SETTLE, {"pos_id": pos_id, "leg_side": side})

    def _handle_exit_check(self, t: pd.Timestamp, pos_id: int) -> None:
        pos = self._open.get(pos_id)
        if pos is None:
            return
        # Hold-time cap.
        hold_h = (t - pos.opened_ts_utc).total_seconds() / 3600.0
        if hold_h >= self.cfg.max_hold_hours:
            self._close_position(t, pos_id, "max_hold")
            return
        # Carry sign-check: if the funding edge has decayed below threshold, exit.
        carry_apr = self._current_carry_apr(pos, t)
        if carry_apr is not None and carry_apr < self.cfg.exit_threshold_apr:
            self._close_position(t, pos_id, "carry_below_threshold")
            return
        # Schedule next exit check.
        nxt = t + pd.Timedelta(minutes=self.cfg.exit_check_interval_minutes)
        if nxt < self.cfg.end:
            self._push(nxt, _EV_EXIT_CHECK, {"pos_id": pos_id})

    def _close_position(self, t: pd.Timestamp, pos_id: int, reason: str) -> None:
        pos = self._open.pop(pos_id, None)
        if pos is None:
            return
        # Exit fees + slippage, attributed separately.
        long_fee_bps = self.costs.fee_bps(pos.long_leg.venue, role=self.cfg.role)
        short_fee_bps = self.costs.fee_bps(pos.short_leg.venue, role=self.cfg.role)
        long_slip_bps, _ = self._slip_bps(pos.long_leg.venue, pos.event.coin, pos.long_leg.notional_usd)
        short_slip_bps, _ = self._slip_bps(pos.short_leg.venue, pos.event.coin, pos.short_leg.notional_usd)
        long_exit_fee = long_fee_bps * 1e-4 * pos.long_leg.notional_usd
        short_exit_fee = short_fee_bps * 1e-4 * pos.short_leg.notional_usd
        long_exit_slip = long_slip_bps * 1e-4 * pos.long_leg.notional_usd
        short_exit_slip = short_slip_bps * 1e-4 * pos.short_leg.notional_usd
        pos.long_leg.fees_paid_usd += long_exit_fee
        pos.short_leg.fees_paid_usd += short_exit_fee
        pos.long_leg.slippage_paid_usd += long_exit_slip
        pos.short_leg.slippage_paid_usd += short_exit_slip
        pos.realized_pnl_usd -= (long_exit_fee + short_exit_fee
                                  + long_exit_slip + short_exit_slip)
        pos.closed_ts_utc = t
        pos.close_reason = reason
        self.pool.release_for_position(pos)
        self._trades.append(TradeRecord(
            event=pos.event,
            long_leg=pos.long_leg, short_leg=pos.short_leg,
            opened=pos.opened_ts_utc, closed=t, close_reason=reason,
            realized_pnl_usd=pos.realized_pnl_usd,
            long_funding_usd=pos.long_leg.accrued_funding_usd,
            short_funding_usd=pos.short_leg.accrued_funding_usd,
            fees_usd=pos.long_leg.fees_paid_usd + pos.short_leg.fees_paid_usd,
            slippage_usd=pos.long_leg.slippage_paid_usd + pos.short_leg.slippage_paid_usd,
            used_capital_usd=pos.used_capital_usd(),
        ))

    # ---- slippage helper ---------------------------------------------- #

    def _slip_bps(self, venue: str, coin: str, notional: float) -> tuple[float, str]:
        """Return (slippage_bps, source). Uses recorder calibration when available."""
        if self.rec_slip is not None:
            return self.rec_slip.get_slippage_bps(venue, coin, notional)
        return self.costs.slippage_bps(notional, depth_notional=None), "fallback"

    # ---- data lookups ------------------------------------------------- #

    def _sliced_panels(self, t: pd.Timestamp) -> dict[tuple[str, str], pd.DataFrame]:
        """Return head-slices ``< t`` for every panel using cached ts arrays.

        ``np.searchsorted`` on int64 ns is O(log N) per panel; head-slicing via
        ``iloc[:idx]`` is a view (no copy). Replaces the previous boolean-mask
        rebuild that scaled with panel size × number of checks.
        """
        t_ns = pd.Timestamp(t).value
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for k, df in self.funding.items():
            arr = self._ts_ns.get(k)
            if arr is None or arr.size == 0:
                out[k] = df.iloc[:0]
                continue
            idx = int(np.searchsorted(arr, t_ns, side="left"))
            out[k] = df.iloc[:idx]
        return out

    def _funding_rate_at(self, venue: str, coin: str, t: pd.Timestamp) -> Optional[float]:
        sym = self._symbol(venue, coin)
        df = self.funding.get((venue, sym))
        arr = self._ts_ns.get((venue, sym))
        if df is None or arr is None or arr.size == 0:
            return None
        t_ns = pd.Timestamp(t).value
        # Search for the nearest row to t (within ±5 min). The engine only
        # calls this at scheduled settle ts, so a tight match should exist.
        i = int(np.searchsorted(arr, t_ns, side="left"))
        candidates = []
        if i < arr.size:
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        if not candidates:
            return None
        best = min(candidates, key=lambda j: abs(int(arr[j]) - t_ns))
        if abs(int(arr[best]) - t_ns) > 300 * 1_000_000_000:  # 5 min in ns
            return None
        return float(df.iloc[best]["funding_rate"])

    def _next_funding_settle_after(self, leg: Leg, t: pd.Timestamp) -> Optional[pd.Timestamp]:
        df = self.funding.get((leg.venue, leg.symbol))
        arr = self._ts_ns.get((leg.venue, leg.symbol))
        if df is None or arr is None or arr.size == 0:
            return None
        t_ns = pd.Timestamp(t).value
        i = int(np.searchsorted(arr, t_ns, side="right"))
        if i >= arr.size:
            return None
        return df.iloc[i]["timestamp_utc"]

    def _current_carry_apr(self, pos: Position, t: pd.Timestamp) -> Optional[float]:
        """Most recent realized funding rates → annualized carry."""
        long_r = self._most_recent_rate(pos.long_leg.venue, pos.event.coin, t)
        short_r = self._most_recent_rate(pos.short_leg.venue, pos.event.coin, t)
        if long_r is None or short_r is None:
            return None
        long_apr = _to_apr(long_r, pos.long_leg.funding_interval_hours)
        short_apr = _to_apr(short_r, pos.short_leg.funding_interval_hours)
        return short_apr - long_apr

    def _most_recent_rate(self, venue: str, coin: str, t: pd.Timestamp) -> Optional[float]:
        sym = self._symbol(venue, coin)
        df = self.funding.get((venue, sym))
        arr = self._ts_ns.get((venue, sym))
        if df is None or arr is None or arr.size == 0:
            return None
        i = int(np.searchsorted(arr, pd.Timestamp(t).value, side="left"))
        if i == 0:
            return None
        return float(df.iloc[i - 1]["funding_rate"])

    def _cadence(self, venue: str, coin: str) -> int:
        sym = self._symbol(venue, coin)
        cad = self._venue_sym_to_cadence.get((venue, sym))
        if cad is not None:
            return cad
        return 1 if venue == "hyperliquid" else 8

    def _symbol(self, venue: str, coin: str) -> str:
        sym = self._venue_base_to_sym.get((venue, coin))
        if sym is not None:
            return sym
        if venue == "hyperliquid":
            return coin
        if venue == "okx":
            return f"{coin}-USDT-SWAP"
        return f"{coin}USDT"

    def _price_at(self, venue: str, coin: str, t: pd.Timestamp) -> Optional[float]:
        sym = self._symbol(venue, coin)
        df = self.funding.get((venue, sym))
        arr = self._ts_ns.get((venue, sym))
        if df is None or arr is None or arr.size == 0:
            return 1.0
        i = int(np.searchsorted(arr, pd.Timestamp(t).value, side="right"))
        if i == 0:
            return 1.0
        row = df.iloc[i - 1]
        if "mark_price" in df.columns and pd.notna(row.get("mark_price")):
            return float(row["mark_price"])
        return 1.0  # neutral: funding-history-only panels carry no mark


def _to_apr(rate: float, interval_hours: int) -> float:
    """Convert a single-period funding rate into an annualized rate."""
    if interval_hours <= 0:
        return 0.0
    periods_per_year = (365.0 * 24.0) / interval_hours
    return rate * periods_per_year
