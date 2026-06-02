"""Open position tracking and microstructure-driven exit logic for v1."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    symbol: str
    side: str           # "LONG" or "SHORT"
    entry_price: float
    entry_ts_ms: int
    qty: float
    decision_id: str
    episode_id: str
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    flow_reversal_count: int = field(default=0, repr=False)
    # entry context — stored at open time for dynamic exit calibration and logging
    score: float = 0.0                  # continuation score at entry (0–1)
    liq_notional_1s: float = 0.0        # liquidation notional at entry (USD)
    dynamic_tp_bps: float = 25.0        # per-position TP computed from score + cascade size
    # maker-mode tracking (Phase 1 / Phase 3 investigation)
    entry_maker: bool = False           # True if entry was filled as a maker (limit) order


@dataclass(frozen=True)
class ExitSignal:
    symbol: str
    reason: str
    close_side: str     # "SELL" to close LONG, "BUY" to close SHORT
    mfe_bps: float
    mae_bps: float
    hold_ms: int
    exit_maker: bool = False   # True when exit can be treated as maker (trailing_stop)


class PositionManager:
    """
    Tracks one open position per symbol and evaluates exit conditions.

    Exit priority (first match wins):
      1. Hard stop      — price moves HARD_STOP_BPS adverse
      2. Dynamic TP     — score-proportional target: SCORE_TP_BASE + score*SCORE_TP_SCALE
                          + cascade-size bonus up to LIQ_TP_MAX_BONUS bps
      3. Trailing stop  — once MFE >= TRAILING_ACTIVATE_BPS, protect MFE − TRAIL_BPS
      4. Time limit     — held >= TIME_LIMIT_MS
      5. Flow reversal  — adverse flow_imbalance_1s for effective_flow_count consecutive checks
                          (base FLOW_REVERSAL_COUNT, +2 for large cascades ≥ LIQ_WIDE_FLOW_THRESHOLD)
      6. Depth recovery — depth collapse resolved + no failed_recovery + held >= DEPTH_RECOVERY_MIN_HOLD_MS

    Calibration notes (2026-03-22, 13 live paper trades + SOL/HYPE replay):
      - All 13 live trades exited in ≤17s (mean 3.8s) via flow reversal; TIME_LIMIT never used.
      - Original FLOW_REVERSAL_COUNT=2 fired within ~500ms of cascade peak — too fast.
        Raised to 4 (~1s confirmation) to give momentum time to develop.
      - Hard stop (HARD_STOP_BPS) never triggered; flow reversal always fires first.
        Left at 15 bps as pure safety backstop.
      - Best live winner: +21.93 bps (BTC SHORT, 11s hold). Raised TP to 25 bps.
      - TIME_LIMIT reduced from 180s to 60s; all observed exits well under 20s.

    Calibration notes (2026-03-23, 68 paper trades including Mar 22 liquidation event):
      - flow_reversal: 62 trades WR=0%, net=-$2.43. All losses; dominant exit mechanism.
      - hard_tp: 4 trades WR=100%, net=+$0.074. Every TP hit is profitable.
      - Critical case: ETH MFE=+14.5 bps → flow continued → hard_stop at -17.71 bps.
        Trailing stop (activate@8, trail@5) would have exited at +9.5 bps — saving 27 bps.
      - ETH MFE=+8.8 bps → flow_reversal exit at -7.47 bps. Trailing saves +11 bps.
      - Added trailing stop as priority 3 between hard_tp and time_limit.
      - FLOW_REVERSAL_COUNT raised 4→6 (1.5s confirmation) to give trailing time to engage.
      - Added score-proportional + cascade-size TP, and cascade-size flow reversal widening.

    Calibration notes (2026-03-24, 96 paper trades, 48h liquidation event):
      - Root cause of trailing stop failure: exit check was throttled by 250ms decision
        interval — price could gap 20+ bps between checks. Fixed: fast-path exit check
        (price_only=True) now runs on EVERY market event before the throttle gate.
      - TRAILING_ACTIVATE_BPS lowered 8→5: catches ETH-class moves (MFE=5 bps → exit
        before turning negative) without triggering on noise.
      - TRAIL_BPS lowered 5→3: tighter protection once armed; floor is MFE−3 bps.
      - These two changes together: ETH MFE=+5.20 exit would improve from -7.47 to +2.20.

    Calibration notes (2026-03-26, 129 paper trades, updated analysis):
      - Trailing stop working: 9/9 wins at mean +4.28 bps, capture rate 28–70% (avg 45%).
      - TRAIL_BPS lowered 3→2: each trailing_stop exit improves by 1 bps on average.
        At MFE=8: floor rises 5→6 bps. At MFE=20: floor rises 17→18 bps.
      - TRAILING_ACTIVATE_BPS lowered 5→3: arm earlier to protect trades reaching 3–5 bps
        MFE before reversing. With TRAIL=2: MFE=3→floor=1, MFE=5→floor=3.
        Trailing fires before flow_reversal (priority 3 vs 5) — converts -1 bps exits to +1.
      - HARD_STOP_BPS lowered 15→12: HYPE hard_stop at -16.87 bps (MFE=+0.13, wasted).
        All hard_stops in data exceeded 15 bps loss; 12 bps saves 3–5 bps per event.
      - TP formula unchanged: 5 hard_tp hits averaged +27.77 bps. Capping at <20 bps
        would cost ~70 bps on those 5 trades; trailing stop handles medium moves (10–20 bps).
    """

    # ── hard limits ────────────────────────────────────────────────────────────
    # Lowered 12→8 bps (grid-search best on 451 trades: saves ~4 bps on 15% of
    # hard-stop trades; tighter activation also improves gross expectancy +2.4 bps)
    HARD_STOP_BPS: float = 8.0

    # ── dynamic TP calibration ─────────────────────────────────────────────────
    # tp = min(SCORE_TP_BASE + score*SCORE_TP_SCALE + liq_bonus, HARD_TP_MAX_BPS)
    # At score=0.42: 12+10.1=22.1 bps; score=0.55: 12+13.2=25.2 bps
    SCORE_TP_BASE: float = 12.0
    SCORE_TP_SCALE: float = 24.0
    # cascade bonus: +up to LIQ_TP_MAX_BONUS at liq_notional=LIQ_TP_NOTIONAL_SCALE
    LIQ_TP_MAX_BONUS: float = 5.0
    LIQ_TP_NOTIONAL_SCALE: float = 100_000.0    # $100K liquidation → full bonus
    HARD_TP_MAX_BPS: float = 40.0               # absolute ceiling on TP

    # ── trailing stop ──────────────────────────────────────────────────────────
    # Lowered activate 5→2 bps: 75% of V3 trades reach MFE≥2; at activate=5
    # only 58% arm trailing.  Extra 17% now lock in +1 bps instead of giving
    # the move back to 0.  Grid-search: +2.4 bps gross improvement overall.
    # Lowered trail 3→1 bps: floor rises from MFE−3 to MFE−1; every trailing_stop
    # exit gains +2 bps.  MFE=2 → exits at +1 bps (was 0 or negative at activate=5).
    TRAILING_ACTIVATE_BPS: float = 2.0     # arm at 2 bps MFE — capture shallow bounces
    TRAIL_BPS: float = 1.0                 # protect MFE − 1 bps once armed

    # ── time limit ─────────────────────────────────────────────────────────────
    TIME_LIMIT_MS: int = 60_000            # unchanged — exits well under 60s in practice

    # ── flow reversal ──────────────────────────────────────────────────────────
    FLOW_REVERSAL_THRESHOLD: float = 0.15
    FLOW_REVERSAL_COUNT: int = 20          # 5s confirmation (was 10/2.5s — WR=1% on 142 exits, Apr 2026)
    LIQ_WIDE_FLOW_THRESHOLD: float = 50_000.0   # cascade ≥ $50K → +2 extra confirmations
    FLOW_REVERSAL_COUNT_BONUS: int = 2          # extra checks for large cascades

    # ── depth recovery ─────────────────────────────────────────────────────────
    DEPTH_RECOVERY_MIN_HOLD_MS: int = 15_000
    DEPTH_COLLAPSE_RECOVERED: float = 0.03

    def __init__(self) -> None:
        self._positions: Dict[str, OpenPosition] = {}

    def _compute_dynamic_tp(self, score: float, liq_notional_1s: float) -> float:
        """Score-proportional TP with cascade-size bonus.

        Higher-score signals have deeper depth collapse and stronger flow, so price
        can travel further before exhaustion.  Large liquidation cascades inject more
        forced-seller momentum.
        """
        score_tp = self.SCORE_TP_BASE + score * self.SCORE_TP_SCALE
        liq_bonus = min(
            self.LIQ_TP_MAX_BONUS,
            liq_notional_1s / self.LIQ_TP_NOTIONAL_SCALE * self.LIQ_TP_MAX_BONUS,
        )
        return min(score_tp + liq_bonus, self.HARD_TP_MAX_BPS)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        return self._positions.get(symbol)

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        entry_ts_ms: int,
        qty: float,
        decision_id: str,
        episode_id: str,
        score: float = 0.0,
        liq_notional_1s: float = 0.0,
        entry_maker: bool = False,
    ) -> None:
        if symbol in self._positions:
            logger.warning(
                "open_position called with existing position for %s — ignoring (decision_id=%s)",
                symbol,
                decision_id,
            )
            return
        dynamic_tp = self._compute_dynamic_tp(score, liq_notional_1s)
        self._positions[symbol] = OpenPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_ts_ms=entry_ts_ms,
            qty=qty,
            decision_id=decision_id,
            episode_id=episode_id,
            score=score,
            liq_notional_1s=liq_notional_1s,
            dynamic_tp_bps=dynamic_tp,
            entry_maker=entry_maker,
        )
        logger.info(
            "position_opened symbol=%s side=%s entry=%.4f qty=%.4f score=%.3f "
            "liq=%.0f dynamic_tp=%.1fbps decision_id=%s",
            symbol, side, entry_price, qty, score, liq_notional_1s, dynamic_tp, decision_id,
        )

    def close_position(self, symbol: str) -> Optional[OpenPosition]:
        pos = self._positions.pop(symbol, None)
        if pos is not None:
            logger.info(
                "position_closed symbol=%s side=%s entry=%.4f mfe=%.2f mae=%.2f "
                "score=%.3f dynamic_tp=%.1fbps decision_id=%s",
                symbol, pos.side, pos.entry_price, pos.mfe_bps, pos.mae_bps,
                pos.score, pos.dynamic_tp_bps, pos.decision_id,
            )
        return pos

    def check_exits(
        self,
        *,
        symbol: str,
        ts_ms: int,
        features: Dict[str, object],
        mid_price: float,
        price_only: bool = False,
    ) -> Optional[ExitSignal]:
        """Evaluate exit conditions for an open position.

        Args:
            price_only: If True, only evaluate price-based exits (hard_stop, hard_tp,
                trailing_stop, time_limit) without touching the flow_reversal counter.
                Used for the high-frequency fast-path check on every market event.
                Flow/depth exits are evaluated in the normal 250ms decision cycle.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        hold_ms = ts_ms - pos.entry_ts_ms
        sign = 1.0 if pos.side == "LONG" else -1.0
        pnl_bps = sign * (mid_price - pos.entry_price) / pos.entry_price * 10_000.0

        # update MFE / MAE (mutate in place — dataclass not frozen)
        if pnl_bps > pos.mfe_bps:
            pos.mfe_bps = pnl_bps
        if pnl_bps < pos.mae_bps:
            pos.mae_bps = pnl_bps

        close_side = "SELL" if pos.side == "LONG" else "BUY"

        def _signal(reason: str, *, exit_maker: bool = False) -> ExitSignal:
            # trailing_stop exits are inherently passive (price hits our resting floor)
            # and can be served as maker orders in Phase 3 investigation.
            return ExitSignal(
                symbol=symbol,
                reason=reason,
                close_side=close_side,
                mfe_bps=pos.mfe_bps,
                mae_bps=pos.mae_bps,
                hold_ms=hold_ms,
                exit_maker=exit_maker,
            )

        # 1. Hard stop
        if pnl_bps <= -self.HARD_STOP_BPS:
            return _signal("hard_stop")

        # 2. Dynamic TP — score-proportional + cascade-size bonus, computed at open time
        if pnl_bps >= pos.dynamic_tp_bps:
            return _signal("hard_tp")

        # 3. Trailing stop — armed once MFE >= TRAILING_ACTIVATE_BPS
        #    Protects (MFE − TRAIL_BPS) once engaged.  Fires before time_limit and
        #    flow_reversal so a developing move locks in profit rather than giving it back.
        #    Tagged exit_maker=True: price crossing our floor from above is inherently
        #    passive (we could serve this as a resting limit order — Phase 3 maker exit).
        if pos.mfe_bps >= self.TRAILING_ACTIVATE_BPS:
            trail_floor = pos.mfe_bps - self.TRAIL_BPS
            if pnl_bps < trail_floor:
                return _signal("trailing_stop", exit_maker=True)

        # 4. Time limit
        if hold_ms >= self.TIME_LIMIT_MS:
            return _signal("time_limit")

        # Fast-path callers stop here — flow/depth exits handled in the decision cycle.
        if price_only:
            return None

        # 5. Flow reversal — large cascades get extra confirmation time
        #    because more forced sellers may still be pending.
        effective_flow_count = self.FLOW_REVERSAL_COUNT
        if pos.liq_notional_1s >= self.LIQ_WIDE_FLOW_THRESHOLD:
            effective_flow_count = self.FLOW_REVERSAL_COUNT + self.FLOW_REVERSAL_COUNT_BONUS

        flow = float(features.get("flow_imbalance_1s") or 0.0)
        flow_adverse = (
            (pos.side == "LONG" and flow < -self.FLOW_REVERSAL_THRESHOLD)
            or (pos.side == "SHORT" and flow > self.FLOW_REVERSAL_THRESHOLD)
        )
        if flow_adverse:
            # MFE-aware: if trade is solidly in profit, decay counter instead of
            # incrementing — don't kill a winning trade on brief adverse flow.
            if pnl_bps > self.TRAIL_BPS:
                pos.flow_reversal_count = max(0, pos.flow_reversal_count - 1)
            else:
                pos.flow_reversal_count += 1
        else:
            pos.flow_reversal_count = 0
        if pos.flow_reversal_count >= effective_flow_count:
            return _signal("flow_reversal")

        # 6. Depth recovery — structure normalised, momentum exhausted
        if hold_ms >= self.DEPTH_RECOVERY_MIN_HOLD_MS:
            depth_collapse = float(features.get("depth_collapse_ratio") or 1.0)
            failed_recovery = float(features.get("failed_recovery") or 0.0)
            if depth_collapse < self.DEPTH_COLLAPSE_RECOVERED and failed_recovery == 0.0:
                return _signal("depth_recovered")

        return None
