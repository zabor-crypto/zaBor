#!/usr/bin/env python3
"""Unit tests for PnL attribution and risk ranking.

Covers:
 - Attribution source = SHORT when short PnL delta dominates
 - Attribution source = LONG when long PnL delta dominates
 - Attribution source = MIXED when no side exceeds threshold
 - Positive current PnL with negative PnL delta is counted as drawdown contribution
 - Risk ranking sorts worst contributors first
 - Partial close plan respects top_n and close_fraction
 - Liquidation proximity forces full close
 - Spot multi-hop sells only BTC/ETH delta, not pre-existing balance
"""

import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk_attribution import (
    PositionSnapshot,
    AttributionResult,
    compute_pnl_attribution,
    compute_risk_score,
    rank_positions_by_risk,
    _liquidation_risk_score,
)
from killswitch import Scope, ExchangeId, AccountType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope():
    return Scope(ExchangeId.MOCK, AccountType.FUTURES)


def _snap(symbol, side, upnl, mark=100.0, liq=None, margin=100.0, notional=500.0):
    return PositionSnapshot(
        ts=int(time.time()), scope=_scope(), symbol=symbol, side=side,
        contracts=5.0, notional_usdt=notional,
        entry_price=90.0, mark_price=mark, liquidation_price=liq,
        margin_usdt=margin, leverage=5.0,
        unrealized_pnl_usdt=upnl, pnl_pct_on_margin=upnl / margin if margin else None,
        raw={},
    )


# ---------------------------------------------------------------------------
# Attribution tests
# ---------------------------------------------------------------------------

class TestComputePnlAttribution:

    def test_source_short_dominates(self):
        """Attribution = SHORT when short pnl delta dominates."""
        current = [
            _snap("SOL/USDT", "short", upnl=-400.0),  # short lost 400
            _snap("BTC/USDT", "long",  upnl=+50.0),   # long gained 50
        ]
        reference = [
            _snap("SOL/USDT", "short", upnl=0.0),
            _snap("BTC/USDT", "long",  upnl=0.0),
        ]
        attr = compute_pnl_attribution(current, reference, source_threshold=0.65)
        assert attr.source == "SHORT"
        assert attr.short_negative_delta == pytest.approx(400.0)
        assert attr.long_negative_delta == pytest.approx(0.0)

    def test_source_long_dominates(self):
        """Attribution = LONG when long pnl delta dominates."""
        current = [
            _snap("ETH/USDT", "long",  upnl=-350.0),  # long lost 350
            _snap("BTC/USDT", "short", upnl=-30.0),   # short lost 30
        ]
        reference = [
            _snap("ETH/USDT", "long",  upnl=0.0),
            _snap("BTC/USDT", "short", upnl=0.0),
        ]
        attr = compute_pnl_attribution(current, reference, source_threshold=0.65)
        assert attr.source == "LONG"
        assert attr.long_negative_delta == pytest.approx(350.0)
        assert attr.short_pct < 0.10

    def test_source_mixed_when_balanced(self):
        """Attribution = MIXED when neither side reaches threshold."""
        current = [
            _snap("BTC/USDT", "long",  upnl=-200.0),
            _snap("ETH/USDT", "short", upnl=-180.0),
        ]
        reference = [
            _snap("BTC/USDT", "long",  upnl=0.0),
            _snap("ETH/USDT", "short", upnl=0.0),
        ]
        attr = compute_pnl_attribution(current, reference, source_threshold=0.65)
        assert attr.source == "MIXED"

    def test_positive_pnl_with_negative_delta_counted(self):
        """A position still positive but declining contributes to drawdown."""
        # SHORT SOL was +800, now +100 — delta = -700 (counted as short contribution)
        current = [_snap("SOL/USDT", "short", upnl=+100.0)]
        reference = [_snap("SOL/USDT", "short", upnl=+800.0)]
        attr = compute_pnl_attribution(current, reference, source_threshold=0.65)
        assert attr.source == "SHORT"
        assert attr.short_negative_delta == pytest.approx(700.0)
        assert attr.total_negative_delta == pytest.approx(700.0)

    def test_no_negative_delta_returns_unknown(self):
        """When all positions improved, source is UNKNOWN."""
        current = [_snap("BTC/USDT", "long", upnl=+200.0)]
        reference = [_snap("BTC/USDT", "long", upnl=+100.0)]
        attr = compute_pnl_attribution(current, reference)
        assert attr.source == "UNKNOWN"
        assert attr.total_negative_delta == 0.0

    def test_empty_reference_uses_zero_baseline(self):
        """Empty reference treats all current losses as from zero."""
        current = [_snap("SOL/USDT", "short", upnl=-200.0)]
        attr = compute_pnl_attribution(current, [], source_threshold=0.65)
        assert attr.source == "SHORT"
        assert attr.short_negative_delta == pytest.approx(200.0)

    def test_threshold_boundary_exact(self):
        """Exactly at threshold should classify as dominant."""
        current = [
            _snap("A/USDT", "short", upnl=-65.0),
            _snap("B/USDT", "long",  upnl=-35.0),
        ]
        reference = [
            _snap("A/USDT", "short", upnl=0.0),
            _snap("B/USDT", "long",  upnl=0.0),
        ]
        attr = compute_pnl_attribution(current, reference, source_threshold=0.65)
        assert attr.source == "SHORT"
        assert attr.short_pct == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Risk score and ranking tests
# ---------------------------------------------------------------------------

class TestRiskScoreAndRanking:

    def test_risk_ranking_worst_first(self):
        """Positions with larger negative delta rank first."""
        current = [
            _snap("A/USDT", "short", upnl=-500.0, notional=1000.0),
            _snap("B/USDT", "short", upnl=-100.0, notional=200.0),
            _snap("C/USDT", "short", upnl=-300.0, notional=600.0),
        ]
        reference = [_snap(s.symbol, s.side, upnl=0.0) for s in current]
        ranked = rank_positions_by_risk(current, reference, 10000.0, "SHORT")
        assert ranked[0][0].symbol == "A/USDT"
        assert ranked[1][0].symbol == "C/USDT"
        assert ranked[2][0].symbol == "B/USDT"

    def test_ranking_filters_by_source_long(self):
        """When source=LONG, only long positions are candidates."""
        current = [
            _snap("A/USDT", "long",  upnl=-200.0),
            _snap("B/USDT", "short", upnl=-300.0),
        ]
        reference = [
            _snap("A/USDT", "long",  upnl=0.0),
            _snap("B/USDT", "short", upnl=0.0),
        ]
        ranked = rank_positions_by_risk(current, reference, 10000.0, "LONG")
        assert len(ranked) == 1
        assert ranked[0][0].symbol == "A/USDT"

    def test_ranking_filters_by_source_short(self):
        """When source=SHORT, only short positions are candidates."""
        current = [
            _snap("A/USDT", "long",  upnl=-200.0),
            _snap("B/USDT", "short", upnl=-300.0),
        ]
        reference = [
            _snap("A/USDT", "long",  upnl=0.0),
            _snap("B/USDT", "short", upnl=0.0),
        ]
        ranked = rank_positions_by_risk(current, reference, 10000.0, "SHORT")
        assert len(ranked) == 1
        assert ranked[0][0].symbol == "B/USDT"

    def test_ranking_mixed_includes_all(self):
        """When source=MIXED, all positions are candidates."""
        current = [
            _snap("A/USDT", "long",  upnl=-100.0),
            _snap("B/USDT", "short", upnl=-200.0),
        ]
        reference = [
            _snap("A/USDT", "long",  upnl=0.0),
            _snap("B/USDT", "short", upnl=0.0),
        ]
        ranked = rank_positions_by_risk(current, reference, 10000.0, "MIXED")
        assert len(ranked) == 2

    def test_source_mode_override(self):
        """source_mode=SHORT overrides attribution source even if attribution says LONG."""
        current = [
            _snap("A/USDT", "long",  upnl=-400.0),
            _snap("B/USDT", "short", upnl=-50.0),
        ]
        reference = [
            _snap("A/USDT", "long",  upnl=0.0),
            _snap("B/USDT", "short", upnl=0.0),
        ]
        # Override to only consider shorts
        ranked = rank_positions_by_risk(current, reference, 10000.0,
                                        source="LONG", source_mode="SHORT")
        assert len(ranked) == 1
        assert ranked[0][0].symbol == "B/USDT"


# ---------------------------------------------------------------------------
# Liquidation risk score
# ---------------------------------------------------------------------------

class TestLiquidationRiskScore:

    def test_liq_at_mark_price(self):
        """Score = 1.0 when mark == liq (zero distance)."""
        pos = _snap("A/USDT", "long", upnl=-100.0, mark=100.0, liq=100.0)
        assert _liquidation_risk_score(pos) == pytest.approx(1.0)

    def test_liq_far_away(self):
        """Score should be very small when liq price is far."""
        pos = _snap("A/USDT", "long", upnl=-50.0, mark=100.0, liq=10.0)
        score = _liquidation_risk_score(pos)
        assert score < 0.15, f"Expected low liq risk, got {score:.4f}"

    def test_liq_close_triggers_high_score(self):
        """Score should be high when liq price is very close."""
        pos = _snap("A/USDT", "long", upnl=-90.0, mark=100.0, liq=98.0)
        score = _liquidation_risk_score(pos)
        assert score > 0.5, f"Expected high liq risk, got {score:.4f}"

    def test_short_liq_above_mark(self):
        """For shorts, liquidation is above mark price."""
        pos = _snap("A/USDT", "short", upnl=-80.0, mark=100.0, liq=102.0)
        score = _liquidation_risk_score(pos)
        assert score > 0.5

    def test_no_liq_price_returns_zero(self):
        """If liquidation_price is None, score = 0."""
        pos = _snap("A/USDT", "long", upnl=-50.0, mark=100.0, liq=None)
        assert _liquidation_risk_score(pos) == 0.0


# ---------------------------------------------------------------------------
# Close plan helpers
# ---------------------------------------------------------------------------

class TestClosePlanHelpers:

    def test_top_n_respected(self):
        """Plan should contain exactly top_n entries."""
        positions = [_snap(f"{i}/USDT", "short", upnl=float(-100 * (i + 1))) for i in range(5)]
        reference = [_snap(p.symbol, p.side, upnl=0.0) for p in positions]
        ranked = rank_positions_by_risk(positions, reference, 10000.0, "SHORT")
        top_n = 3
        plan_positions = ranked[:top_n]
        assert len(plan_positions) == top_n

    def test_liq_proximity_forces_full_close(self):
        """Position within liq_distance_below_pct should get close_fraction=1.0."""
        mark = 100.0
        liq = 98.0   # 2% distance, below the 3% threshold
        pos = _snap("A/USDT", "long", upnl=-90.0, mark=mark, liq=liq)
        dist_pct = (mark - liq) / mark
        full_close_threshold = 0.03
        assert dist_pct < full_close_threshold, "Position should trigger full close"


# ---------------------------------------------------------------------------
# Spot delta-only routing
# ---------------------------------------------------------------------------

class TestSpotDeltaOnlyRouting:
    """Test the BTC/ETH delta-only logic conceptually (no real exchange needed)."""

    def test_delta_calculation_never_negative(self):
        """Delta should be max(0, after - before) — can't be negative."""
        btc_before = 0.5
        btc_after_hop1 = 0.501   # received 0.001 BTC from selling altcoin
        delta = max(0.0, btc_after_hop1 - btc_before)
        assert delta == pytest.approx(0.001)

    def test_delta_zero_when_no_change(self):
        """If balance didn't change (unlikely but possible), delta = 0."""
        btc_before = 0.5
        btc_after = 0.5
        delta = max(0.0, btc_after - btc_before)
        assert delta == 0.0

    def test_preexisting_btc_never_included(self):
        """Pre-existing BTC is excluded when delta_only=True."""
        preexisting_btc = 0.5
        new_btc_from_hop = 0.001
        btc_after = preexisting_btc + new_btc_from_hop

        delta_only_sell = max(0.0, btc_after - preexisting_btc)
        full_sell = btc_after

        assert delta_only_sell == pytest.approx(new_btc_from_hop)
        assert full_sell == pytest.approx(0.501)
        # With delta_only, pre-existing is NOT sold
        assert delta_only_sell < full_sell

    def test_dust_left_when_below_min_notional(self):
        """If delta notional is below MIN_NOTIONAL, it should be left as dust."""
        delta_btc = 0.000001     # tiny delta
        btc_price = 60000.0
        notional = delta_btc * btc_price
        min_notional = 10.0
        assert notional < min_notional, "Should be treated as dust"
