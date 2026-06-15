#!/usr/bin/env python3
"""Unit tests for the regime guard (slow-bleed / cumulative-drawdown layer)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_attribution import PositionSnapshot
from killswitch import Scope, ExchangeId, AccountType
from regime_guard import RegimeGuardConfig, evaluate_regime_guard


def _scope():
    return Scope(ExchangeId.MOCK, AccountType.FUTURES)


def _snap(symbol, side, upnl, margin=100.0, notional=500.0):
    return PositionSnapshot(
        ts=int(time.time()), scope=_scope(), symbol=symbol, side=side,
        contracts=5.0, notional_usdt=notional, entry_price=90.0, mark_price=100.0,
        liquidation_price=None, margin_usdt=margin, leverage=5.0,
        unrealized_pnl_usdt=upnl, pnl_pct_on_margin=(upnl / margin if margin else None), raw={},
    )


def _cfg(**over):
    # tests exercise all layers incl. the optional L1 path, so enable it here even though
    # production defaults L1 off (it overrides each bot's own exit logic).
    base = dict(enabled=True, l1_enabled=True, hard_pct=0.25, til_hours=18.0, til_pct=0.06,
                peak_lookback_h=48.0, dd_trig=0.06, persist_cycles=3, cl_pct=0.06,
                cluster_k=4, dl_trig=0.06, up_gate=0.025, flush_block6=0.03)
    base.update(over)
    return RegimeGuardConfig(**base)


def _eval(cfg, positions, equity=1000.0, peak=1000.0, day_open=1000.0,
          btc24=0.0, btc6=0.0, ddc=5, uw=None, l2cd=False, l3cd=False, l4cd=False):
    return evaluate_regime_guard(cfg, positions, equity, peak, day_open, btc24, btc6,
                                 ddc, uw or {}, l2cd, l3cd, l4cd)


def test_disabled_noop():
    d = _eval(_cfg(enabled=False), [_snap("A", "long", -50)])
    assert not d.fired


def test_l1_hard_stop_fires_any_regime():
    # -30% on margin (<= -25%) -> hard stop, even in an uptrend (gate does not apply to L1)
    d = _eval(_cfg(), [_snap("A", "long", -30, margin=100)], btc24=0.10)
    assert d.fired and d.layer == "L1_hard"
    assert d.plan[0].symbol == "A"


def test_l1_no_stop_when_shallow():
    d = _eval(_cfg(), [_snap("A", "long", -5, margin=100)])  # -5% only
    assert not d.fired


def test_l1_time_stop():
    # underwater 20h and -8% on margin -> time stop
    cfg = _cfg()
    uw = {("A", "long"): 20.0}
    d = _eval(cfg, [_snap("A", "long", -8, margin=100)], uw=uw)
    assert d.fired and d.layer == "L1_time"


def test_l1_time_stop_needs_both_duration_and_loss():
    cfg = _cfg()
    # 20h underwater but only -3% -> below til_pct, no time stop (and not hard)
    d = _eval(cfg, [_snap("A", "long", -3, margin=100)], uw={("A", "long"): 20.0})
    assert not d.fired


def test_l2_peakdd_flushes_dominant_losing_side():
    cfg = _cfg()
    pos = [_snap("A", "long", -20), _snap("B", "long", -15), _snap("C", "short", +5)]
    d = _eval(cfg, pos, equity=900, peak=1000, ddc=5)  # -10% dd, persisted
    assert d.fired and d.layer == "L2_peakdd"
    assert {c.symbol for c in d.plan} == {"A", "B"}     # only the losing longs


def test_l2_blocked_by_uptrend_gate():
    cfg = _cfg()
    pos = [_snap("A", "long", -20)]
    d = _eval(cfg, pos, equity=900, peak=1000, day_open=900, ddc=5, btc24=0.05)  # BTC +5% > up_gate
    # L1 shallow (-20% margin? -20/100=-20% > -25 so no hard). So no fire.
    assert not d.fired


def test_l2_blocked_by_fresh_flush():
    cfg = _cfg()
    pos = [_snap("A", "long", -20)]
    d = _eval(cfg, pos, equity=900, peak=1000, day_open=900, ddc=5, btc6=-0.05)  # BTC6h -5% acute flush
    assert not d.fired


def test_l2_needs_persistence():
    cfg = _cfg(persist_cycles=3)
    pos = [_snap("A", "long", -20)]
    d = _eval(cfg, pos, equity=900, peak=1000, day_open=900, ddc=1)  # only 1 cycle persisted
    assert not d.fired


def test_l3_cluster():
    cfg = _cfg(cluster_k=4, cl_pct=0.06)  # close_top_n defaults to 3
    # 4 longs each -10% on margin -> cluster fires; equity flat so L2/L4 quiet
    pos = [_snap(s, "long", -10, margin=100) for s in ["A", "B", "C", "D"]]
    d = _eval(cfg, pos, equity=1000, peak=1000, ddc=0)
    assert d.fired and d.layer == "L3_cluster"
    assert len(d.plan) == 3   # cluster detected on 4, closes the top-3 fastest bleeders


def test_l3_cluster_close_all_when_top_n_zero():
    cfg = _cfg(cluster_k=4, cl_pct=0.06, close_top_n=0)
    pos = [_snap(s, "long", -10, margin=100) for s in ["A", "B", "C", "D"]]
    d = _eval(cfg, pos, equity=1000, peak=1000, ddc=0)
    assert d.fired and len(d.plan) == 4


def test_l3_below_threshold():
    cfg = _cfg(cluster_k=4)
    pos = [_snap(s, "long", -10) for s in ["A", "B", "C"]]  # only 3
    d = _eval(cfg, pos, equity=1000, peak=1000, ddc=0)
    assert not d.fired


def test_l4_daily_loss():
    cfg = _cfg(dl_trig=0.06)
    pos = [_snap("A", "long", -20), _snap("B", "short", -5)]
    # equity 930 vs day-open 1000 = -7%, peak=equity so no L2; cluster k not met
    d = _eval(cfg, pos, equity=930, peak=930, day_open=1000, ddc=0)
    assert d.fired and d.layer == "L4_daily"
    assert {c.symbol for c in d.plan} == {"A"}   # dominant losing side = long


def test_cooldowns_block_layers():
    cfg = _cfg()
    pos = [_snap("A", "long", -20), _snap("B", "long", -15)]
    d = _eval(cfg, pos, equity=900, peak=1000, day_open=900, ddc=5, l2cd=True)
    # L2 in cooldown; no cluster (only 2); daily flat -> nothing
    assert not d.fired


def test_l1_off_by_default_single_position_left_alone():
    # Production default: L1 disabled. A single position deep underwater (-90% margin) but with
    # no portfolio/regime trigger (equity at peak, no cluster, day flat) must NOT be touched —
    # that is the owning bot's job, not the kill-switch's.
    cfg = RegimeGuardConfig(enabled=True)  # all defaults -> l1 off, l0 on (8% cap)
    d = _eval(cfg, [_snap("LONE", "long", -50, margin=100)],  # -5% of equity: under L0 cap, L1 off
              equity=1000, peak=1000, day_open=1000, ddc=0)
    assert not d.fired


def test_top_n_velocity_selection_closes_fastest_bleeders():
    # 5 losing longs; with close_top_n=3 and a velocity map, close the 3 fastest bleeders.
    cfg = _cfg(close_top_n=3)
    pos = [_snap(s, "long", -20) for s in ["A", "B", "C", "D", "E"]]
    pos.append(_snap("SH", "short", -1))  # tiny short, not dominant
    # velocity: most negative = fastest. C,E,A fastest; B,D slow.
    vel = {("A", "long"): -50, ("B", "long"): -5, ("C", "long"): -90,
           ("D", "long"): -2, ("E", "long"): -70, ("SH", "short"): -1}
    d = evaluate_regime_guard(cfg, pos, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity=vel)
    assert d.fired and d.layer == "L2_peakdd"
    assert {c.symbol for c in d.plan} == {"C", "E", "A"}   # the 3 fastest bleeders
    assert len(d.plan) == 3


def test_close_top_n_zero_closes_all():
    cfg = _cfg(close_top_n=0)
    pos = [_snap(s, "long", -20) for s in ["A", "B", "C", "D", "E"]]
    d = evaluate_regime_guard(cfg, pos, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={})
    assert d.fired and len(d.plan) == 5


def test_trend_gate_blocks_l2_l3_in_uptrend_but_not_l4():
    # BTC 7d strongly up -> L2 (peak-DD) and L3 (cluster) suppressed; L4 (daily loss) still fires.
    cfg = _cfg(trend_gate_7d=0.0, cluster_k=4)
    longs = [_snap(s, "long", -20) for s in ["A", "B", "C", "D", "E"]]
    # L2 condition present (dd -10%, persisted) but BTC 7d = +6% -> blocked
    d = evaluate_regime_guard(cfg, longs, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=0.06)
    assert not d.fired   # L2 and L3 gated; day flat so no L4
    # now a daily loss in the SAME uptrend -> L4 must still fire (not trend-gated)
    d2 = evaluate_regime_guard(cfg, longs, 900, 1000, 1000, 0.0, 0.0, 0, {},
                               False, False, False, velocity={}, btc_ret_7d=0.06)
    assert d2.fired and d2.layer == "L4_daily"


def test_trend_gate_allows_l2_when_macro_adverse():
    cfg = _cfg(trend_gate_7d=0.0)
    longs = [_snap(s, "long", -20) for s in ["A", "B", "C"]]
    d = evaluate_regime_guard(cfg, longs, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=-0.04)  # BTC 7d down
    assert d.fired and d.layer == "L2_peakdd"


def test_l0_catastrophe_cap_cuts_single_blowup_any_regime():
    # A single short bleeding > 8% of equity must be cut REGARDLESS of regime / cluster / portfolio DD,
    # even in a rising market (BTC 7d up) where it's the only loser.
    cfg = _cfg()  # l0 on by default (max_loss_pct_equity 0.08)
    pos = [_snap("PUMPED", "short", -90, margin=24),   # -90 on a 1000 equity = -9% -> over cap
           _snap("FINE", "long", +20, margin=50)]
    d = evaluate_regime_guard(cfg, pos, 1000, 1000, 1000, 0.0, 0.0, 0, {},
                              False, False, False, velocity={}, btc_ret_7d=0.06)  # BTC up, regime quiet
    assert d.fired and d.layer == "L0_equitycap"
    assert {c.symbol for c in d.plan} == {"PUMPED"}   # only the blow-up, not the winner


def test_l0_does_not_fire_below_cap():
    cfg = _cfg(l1_enabled=False)               # isolate L0 (L1 margin-stop off)
    pos = [_snap("A", "short", -50, margin=100)]   # -50/1000 = -5% of equity < 8% cap
    d = evaluate_regime_guard(cfg, pos, 1000, 1000, 1000, 0.0, 0.0, 0, {},
                              False, False, False, velocity={}, btc_ret_7d=0.06)
    assert not d.fired


def test_short_bleed_in_uptrend_fires_l2_mirrored_gate():
    # Losing SHORTS bleed when BTC rises. The mirrored gate must ALLOW cutting shorts when BTC 7d > 0.
    cfg = _cfg(trend_gate_7d=0.0)
    shorts = [_snap(s, "short", -20) for s in ["A", "B", "C"]]
    d = evaluate_regime_guard(cfg, shorts, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=0.05)  # BTC 7d UP
    assert d.fired and d.layer == "L2_peakdd"
    assert {c.symbol for c in d.plan} <= {"A", "B", "C"}


def test_short_bleed_blocked_when_btc_down():
    # In a downtrend (BTC 7d < 0) losing shorts may RECOVER -> do not cut them (mirror of the long case).
    cfg = _cfg(trend_gate_7d=0.0)
    shorts = [_snap(s, "short", -20) for s in ["A", "B", "C"]]
    d = evaluate_regime_guard(cfg, shorts, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=-0.05)  # BTC 7d DOWN
    assert not d.fired   # L2/L3 gated for shorts when macro favours their recovery; day flat -> no L4


def test_long_case_unchanged_by_side_aware_gate():
    # Regression: the long path must behave exactly as before (BTC 7d < 0 allows, > 0 blocks).
    cfg = _cfg(trend_gate_7d=0.0)
    longs = [_snap(s, "long", -20) for s in ["A", "B", "C"]]
    assert evaluate_regime_guard(cfg, longs, 900, 1000, 900, 0.0, 0.0, 5, {},
                                 False, False, False, velocity={}, btc_ret_7d=-0.05).fired
    assert not evaluate_regime_guard(cfg, longs, 900, 1000, 900, 0.0, 0.0, 5, {},
                                     False, False, False, velocity={}, btc_ret_7d=0.05).fired


def test_min_loss_floor_excludes_near_breakeven_from_l2():
    # With a floor, L2 must not flush positions that are losing LESS than the floor (on margin).
    cfg = _cfg(min_loss_floor=0.10, l1_enabled=False)   # L1 off so we isolate L2's floor
    pos = [_snap("FLAT", "long", -2, margin=100),    # -2% on margin -> excluded by floor
           _snap("DEEP", "long", -20, margin=100)]   # -20% on margin -> eligible (and under L0 8%-equity cap)
    d = evaluate_regime_guard(cfg, pos, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=-0.04)
    assert d.fired and d.layer == "L2_peakdd"
    assert {c.symbol for c in d.plan} == {"DEEP"}   # FLAT (-2%) left alone


def test_floor_zero_is_current_behaviour():
    cfg = _cfg(min_loss_floor=0.0)
    pos = [_snap("A", "long", -2, margin=100), _snap("B", "long", -3, margin=100)]
    d = evaluate_regime_guard(cfg, pos, 900, 1000, 900, 0.0, 0.0, 5, {},
                              False, False, False, velocity={}, btc_ret_7d=-0.04)
    assert d.fired and {c.symbol for c in d.plan} == {"A", "B"}   # both eligible at floor 0


def test_only_losing_positions_closed_never_winners():
    cfg = _cfg()
    pos = [_snap("A", "long", -20), _snap("WIN", "long", +50)]
    d = _eval(cfg, pos, equity=900, peak=1000, ddc=5)
    assert d.fired
    assert "WIN" not in {c.symbol for c in d.plan}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
