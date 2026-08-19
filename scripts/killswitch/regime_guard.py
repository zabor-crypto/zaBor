#!/usr/bin/env python3
"""Regime guard — REGIME / PORTFOLIO-level protection for the kill-switch.

Design philosophy: every position is opened by a trading bot with its own backtested logic and
its own stops. The kill-switch must NOT impose a blanket per-trade stop on top of that. Its job
is to detect that *market conditions / regime* have turned against the book, and close the
positions that are unfavourable FOR THAT REGIME — a portfolio-level decision, not a per-trade one.

This is an ADDITIVE layer that runs every poll cycle alongside the existing fast-crash stages.

Layers (each independent; all close-only — they do NOT cancel/flip, only reduce risk):
  L2 PEAK_DD       portfolio equity <= -dd_trig below its rolling peak_lb_h peak, sustained
                   persist cycles -> close losing positions on the dominant losing side.
                   Uptrend-gated + flush-gated.
  L3 CLUSTER       >= cluster_k same-side positions ALL losing <= -cl_pct on margin -> close that
                   side. This is regime-identification via the book itself: one position at -6% is
                   left for its bot to manage, but K+ of them bleeding together = the regime has
                   turned against that side. Uptrend-gated.
  L4 DAILY_LOSS    portfolio equity down >= dl_trig from UTC day-open -> close dominant losing side.
  L1 PER_POSITION  (OFF by default) optional blanket per-position hard/time stop. Disabled because
                   it overrides each bot's own optimised exit logic. Kept for optionality only.

Refinements:
  - uptrend gate: skip the momentum layers (L2/L3) when BTC 24h return > up_gate (don't flush into a rally).
  - flush gate: skip L2 during an acute BTC flush (6h return <= -flush_block6) to avoid cutting at V-bounce lows.
  - L1 is NEVER gated — a position past its hard-stop is broken in any regime.
Validated by counterfactual replay of one real account's own 60-second history (2026-05-06..06-10):
reconstructed worst-case drawdown approximately -50.7% -> approximately -26%, benign in up-regime,
near-neutral in sharp-down. This is a reconstructed/counterfactual result, not realised live
performance, and it is not reproducible from this repository: the underlying account history is not
published.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from order_safety import CloseInstruction


@dataclass
class RegimeGuardConfig:
    enabled: bool = False
    # ---- deploy guardrails ----
    log_only: bool = False          # observe only: log + TG "WOULD close X", never execute (regime
                                    # guard only — the existing fast stages keep their own dry_run).
    reentry_cooldown_min: int = 360  # after the guard closes (symbol,side), re-close it if a bot
                                    # re-opens it within this window (stops guard↔bot churn). 0 = off.
    max_closes_per_day: int = 12    # cap guard-initiated closes per UTC day (runaway backstop). 0 = off.
    # L0 single-position CATASTROPHE backstop (ON by default). NOT a per-trade stop — it caps how much
    # ONE position may cost the WHOLE account (concentration/ruin protection = the kill-switch's job, not
    # the bot's). Regime-agnostic, highest priority. Catches an idiosyncratic blow-up (e.g. a coin pumped
    # by MMs against a short) that the portfolio/regime triggers (L2/L3/L4) miss because it's 1-2 names.
    # In the replayed account history it fires on roughly 1 position in 180 (a single outlier short well
    # past the cap) -> cuts that one to the cap, never touches normal trades, and acts every 60s vs a
    # bot's slow daily-close stop.
    l0_enabled: bool = True
    max_loss_pct_equity: float = 0.08   # close any single position whose open loss exceeds 8% of equity
    # L1 per-position blanket MARGIN stop — OFF by default (overrides each bot's own optimised exit).
    l1_enabled: bool = False
    hard_pct: float = 0.25          # (if enabled) close if pnl_on_margin <= -25%
    til_hours: float = 18.0         # (if enabled) continuously underwater this long ...
    til_pct: float = 0.06           # ... AND at least -6% on margin -> close
    # L2 peak drawdown (portfolio-level regime trigger)
    l2_enabled: bool = True
    peak_lookback_h: float = 48.0
    dd_trig: float = 0.05           # -5% from rolling peak
    persist_cycles: int = 3         # sustained N cycles (slow-bleed confirm, not a spike)
    l2_cooldown_min: int = 720      # 12h between L2 fires
    l2_lock: bool = False           # close-only (no trading lock) by default
    # L3 cluster (regime-identification via the book: K+ same-side all bleeding)
    l3_enabled: bool = True
    cl_pct: float = 0.06            # each position <= -6% on margin
    cluster_k: int = 4              # >= 4 same-side losers
    l3_cooldown_min: int = 480
    # L4 daily loss (portfolio-level)
    l4_enabled: bool = True
    dl_trig: float = 0.06           # -6% from UTC day-open
    l4_lock: bool = False
    # Minimum per-position loss (on margin) to be ELIGIBLE for an L2/L4 flush. 0.0 = any losing position
    # (validated default). A small value (e.g. 0.03 ≈ -0.3% price at 10x) skips essentially-flat positions;
    # a LARGE value measurably WEAKENS downtrend protection (backtest: floor 10% margin → +543 vs +631,
    # MDD -31.5% vs -26.1%) because in an adverse macro a near-breakeven long usually keeps falling.
    # The velocity selector already avoids cutting stable flat positions, so keep this small or 0.
    min_loss_floor: float = 0.0
    # position SELECTION when a trigger fires
    close_top_n: int = 3            # close the N fastest-bleeding positions on the losing side
                                    # (0 = close ALL losers on the side)
    velocity_windows_min: tuple = (15, 60, 240)  # blend short+medium+long bleed-rate ($/h)
    # gates
    up_gate: float = 0.025          # skip L2/L3 if BTC 24h return > +2.5% (sharp up-spike)
    trend_gate_7d: float = 0.0      # skip L2/L3 unless BTC 7d return < this (only de-risk in a
                                    # CONFIRMED adverse macro; avoids flushing dips inside a slow bull).
                                    # L4 (daily hard loss) is NEVER trend-gated. Validated on the
                                    # reconstructed Apr->Jun window incl. the +8% mid-Apr->mid-May BTC
                                    # uptrend: cut uptrend false-cuts by roughly 80% at near-zero
                                    # downtrend cost (shadow testing). Set None to disable.
    flush_block6: float = 0.03      # skip L2 if BTC 6h return <= -3%
    source_threshold: float = 0.55  # dominant side must own >= this share of open loss


@dataclass
class GuardDecision:
    fired: bool
    layer: str                      # "L1_hard" | "L1_time" | "L2_peakdd" | "L3_cluster" | "L4_daily" | ""
    plan: List[CloseInstruction] = field(default_factory=list)
    set_lock: bool = False
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def _dominant_losing_side(positions, source_threshold: float) -> Optional[str]:
    long_loss = sum(-p.unrealized_pnl_usdt for p in positions if p.side == "long" and p.unrealized_pnl_usdt < 0)
    short_loss = sum(-p.unrealized_pnl_usdt for p in positions if p.side == "short" and p.unrealized_pnl_usdt < 0)
    tot = long_loss + short_loss
    if tot <= 0:
        return None
    if long_loss / tot >= source_threshold:
        return "long"
    if short_loss / tot >= source_threshold:
        return "short"
    # mixed: pick the larger
    return "long" if long_loss >= short_loss else "short"


def _close(p, reason) -> CloseInstruction:
    return CloseInstruction(symbol=p.symbol, side=p.side, contracts=p.contracts,
                            close_fraction=1.0, reason=reason, risk_score=0.0,
                            pnl_delta_usdt=0.0, unrealized_pnl_usdt=p.unrealized_pnl_usdt)


def _floor_ok(p, cfg):
    """True if the position is losing ENOUGH (on margin) to be eligible for an L2/L4 flush."""
    if cfg.min_loss_floor <= 0:
        return True
    return p.pnl_pct_on_margin is not None and p.pnl_pct_on_margin <= -cfg.min_loss_floor


def _pick(losers, cfg, velocity):
    """From the eligible losing positions choose which to actually close.
    close_top_n==0 -> all; else the N with the most-negative blended bleed-rate (fastest bleeders).
    Falls back to current uPnL ($, most negative) when velocity is missing for a position."""
    n = cfg.close_top_n
    if n <= 0 or len(losers) <= n:
        return losers
    def score(p):
        v = velocity.get((p.symbol, p.side)) if velocity else None
        return v if v is not None else p.unrealized_pnl_usdt   # most negative first
    return sorted(losers, key=score)[:n]


def evaluate_regime_guard(
    cfg: RegimeGuardConfig,
    positions: List,                       # List[PositionSnapshot] (current)
    equity: float,
    peak_equity_lookback: float,           # max equity over peak_lookback_h
    day_open_equity: float,                # first equity of current UTC day
    btc_ret_24h: Optional[float],
    btc_ret_6h: Optional[float],
    dd_persist_count: int,                  # how many consecutive cycles dd condition has held (caller tracks)
    underwater_hours: Dict[Tuple[str, str], float],  # (symbol,side) -> continuous hours underwater
    l2_in_cooldown: bool,
    l3_in_cooldown: bool,
    l4_in_cooldown: bool,
    velocity: Optional[Dict[Tuple[str, str], float]] = None,  # (sym,side)->blended bleed-rate $/h
    btc_ret_7d: Optional[float] = None,     # 7d macro trend; gates L2/L3 (NOT L4)
) -> GuardDecision:
    """Pure decision function (no I/O) so it is unit-testable. Caller wires state/cooldowns/locks.
    `velocity` (most-negative = fastest bleed) drives top-N selection when close_top_n > 0."""
    if not cfg.enabled or not positions:
        return GuardDecision(False, "")

    # SIDE-AWARE gates (mirrored). A side may be flushed only when the macro moves AGAINST it (it won't
    # recover): longs need BTC falling, shorts need BTC rising. We sign every BTC return by side
    # (s=+1 long, s=-1 short) so the same thresholds apply symmetrically:
    #   - trend gate: t7 = s*BTC_7d < trend_gate_7d  (longs: BTC_7d<0 ; shorts: BTC_7d>0)
    #   - sharp adverse-spike gate: t24 = s*BTC_24h <= up_gate (don't flush into a spike favouring recovery)
    #   - fresh-flush (L2): t6 = s*BTC_6h <= -flush_block6 (don't cut at the side's own V-extreme)
    # Missing 24h/6h data fails OPEN; the 7d trend gate, when configured, requires a real reading.
    def side_gate(side):
        s = 1.0 if side == "long" else -1.0
        ok = (btc_ret_24h is None) or (s * btc_ret_24h <= cfg.up_gate)
        if cfg.trend_gate_7d is not None and btc_ret_7d is not None:
            ok = ok and (s * btc_ret_7d < cfg.trend_gate_7d)
        return ok

    def fresh_flush_side(side):
        s = 1.0 if side == "long" else -1.0
        return (cfg.flush_block6 is not None) and (btc_ret_6h is not None) and (s * btc_ret_6h <= -cfg.flush_block6)

    # ---- L0 single-position catastrophe backstop (ungated, HIGHEST priority) ----
    # One position must never threaten the whole account, regardless of regime or bot logic.
    if cfg.l0_enabled and equity > 0 and cfg.max_loss_pct_equity:
        cap = -cfg.max_loss_pct_equity * equity
        blown = [p for p in positions if p.unrealized_pnl_usdt <= cap]
        if blown:
            plan = [_close(p, "regime_L0_equitycap") for p in blown]
            worst = min(p.unrealized_pnl_usdt / equity for p in blown)
            return GuardDecision(True, "L0_equitycap", plan, set_lock=False,
                                 reason=f"L0 catastrophe: {len(plan)} position(s) losing > "
                                        f"{cfg.max_loss_pct_equity:.0%} of equity (worst {worst:.0%})",
                                 detail={"n": len(plan)})

    # ---- L1 per-position MARGIN stop (ungated; OFF by default) ----
    if cfg.l1_enabled:
        plan = []
        for p in positions:
            pct = p.pnl_pct_on_margin
            if pct is None:
                continue
            if pct <= -cfg.hard_pct:
                plan.append(_close(p, "regime_L1_hardstop"))
                continue
            uh = underwater_hours.get((p.symbol, p.side), 0.0)
            if uh >= cfg.til_hours and pct <= -cfg.til_pct:
                plan.append(_close(p, "regime_L1_timestop"))
        if plan:
            layer = "L1_hard" if any("hardstop" in c.reason for c in plan) else "L1_time"
            return GuardDecision(True, layer, plan, set_lock=False,
                                 reason=f"L1 per-position stop: {len(plan)} positions",
                                 detail={"n": len(plan)})

    # ---- L2 peak drawdown ----
    if cfg.l2_enabled and not l2_in_cooldown and peak_equity_lookback > 0:
        dd = equity / peak_equity_lookback - 1.0
        if dd <= -cfg.dd_trig and dd_persist_count >= cfg.persist_cycles:
            side = _dominant_losing_side(positions, cfg.source_threshold)
            if side and side_gate(side) and not fresh_flush_side(side):
                losers = [p for p in positions if p.side == side and p.unrealized_pnl_usdt < 0
                          and _floor_ok(p, cfg)]
                chosen = _pick(losers, cfg, velocity)
                plan = [_close(p, "regime_L2_peakdd") for p in chosen]
                if plan:
                    return GuardDecision(True, "L2_peakdd", plan, set_lock=cfg.l2_lock,
                                         reason=f"L2 peak-DD {dd:.1%} from {cfg.peak_lookback_h:.0f}h peak, "
                                                f"flush {len(plan)} fastest-bleeding {side}s",
                                         detail={"dd": dd, "side": side})

    # ---- L3 cluster ----
    if cfg.l3_enabled and not l3_in_cooldown:
        for side in ("long", "short"):
            if not side_gate(side):
                continue
            losers = [p for p in positions if p.side == side and p.pnl_pct_on_margin is not None
                      and p.pnl_pct_on_margin <= -cfg.cl_pct]
            if len(losers) >= cfg.cluster_k:
                chosen = _pick(losers, cfg, velocity)
                plan = [_close(p, "regime_L3_cluster") for p in chosen]
                return GuardDecision(True, "L3_cluster", plan, set_lock=False,
                                     reason=f"L3 cluster: {len(losers)} {side}s ≤ -{cfg.cl_pct:.0%}, "
                                            f"closing {len(plan)} fastest",
                                     detail={"side": side, "n": len(losers)})

    # ---- L4 daily loss ----
    if cfg.l4_enabled and not l4_in_cooldown and day_open_equity > 0:
        dl = equity / day_open_equity - 1.0
        if dl <= -cfg.dl_trig:
            side = _dominant_losing_side(positions, cfg.source_threshold)
            if side:
                losers = [p for p in positions if p.side == side and p.unrealized_pnl_usdt < 0
                          and _floor_ok(p, cfg)]
                chosen = _pick(losers, cfg, velocity)
                plan = [_close(p, "regime_L4_daily") for p in chosen]
                if plan:
                    return GuardDecision(True, "L4_daily", plan, set_lock=cfg.l4_lock,
                                         reason=f"L4 daily loss {dl:.1%} from day-open, "
                                                f"flush {len(plan)} fastest-bleeding {side}s",
                                         detail={"dl": dl, "side": side})

    return GuardDecision(False, "")
