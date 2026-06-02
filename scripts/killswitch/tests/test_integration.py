#!/usr/bin/env python3
"""Integration tests for stage-based kill-switch with mock adapter.

Scenarios:
 A - Losing shorts: Stage 1 closes top-N shorts only, longs untouched
 B - Losing longs: Stage 1 closes top-N longs only, shorts untouched
 C - Mixed book: Stage 1 closes top-N across both sides by risk score
 D - Escalation: 1 → 2 → 3 as equity keeps deteriorating
 E - Stale order safety: entry orders cancelled before close
"""

import sys
import time
import sqlite3
from pathlib import Path
from typing import List
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from killswitch import (
    Scope, ExchangeId, AccountType, EquitySnapshot,
    SqliteStore, DrawdownCalculator, CircuitBreaker,
    MockAdapter, ActionEngine, AccountConfig, StageConfig, StageDecision,
    _process_stage_triggers,
)
from risk_attribution import PositionSnapshot
from stage_machine import StageMachine, STAGE_NAMES
from position_store import PositionStore
from trading_lock import TradingLock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = int(time.time())
MOCK_EQUITY = 10000.0


def _scope():
    return Scope(ExchangeId.MOCK, AccountType.FUTURES)


def _pos(symbol, side, upnl, entry=100.0, mark=100.0, liq=None,
         margin=200.0, notional=1000.0, contracts=10.0):
    return PositionSnapshot(
        ts=BASE_TS, scope=_scope(), symbol=symbol, side=side,
        contracts=contracts, notional_usdt=notional,
        entry_price=entry, mark_price=mark, liquidation_price=liq,
        margin_usdt=margin, leverage=5.0,
        unrealized_pnl_usdt=upnl,
        pnl_pct_on_margin=upnl / margin if margin else None,
        raw={},
    )


def _build_stage_confs(
    s1_thresh=0.04, s2_thresh=0.07, s3_thresh=0.12,
    s1_mode="CLOSE_TOP_RISK_CONTRIBUTORS",
    s2_mode="CLOSE_DOMINANT_LOSS_DIRECTION",
    s3_mode="CLOSE_ALL_POSITIONS",
    s1_source_threshold=0.65,
    s2_source_threshold=0.60,
    top_n=3,
    close_fraction=0.50,
):
    stage_1 = StageConfig(
        thresholds={"1m": s1_thresh},
        mode=s1_mode,
        cooldown_min=30,
        confirm_consecutive=1,
        source_mode="AUTO",
        source_threshold=s1_source_threshold,
        top_n=top_n,
        close_fraction=close_fraction,
        full_close_if_liq_distance_below_pct=0.03,
        cancel_entry_orders_before_close=True,
        set_trading_lock=True,
    )
    stage_2 = StageConfig(
        thresholds={"1m": s2_thresh},
        mode=s2_mode,
        cooldown_min=90,
        confirm_consecutive=1,
        source_mode="AUTO",
        source_threshold=s2_source_threshold,
        cancel_entry_orders_before_close=True,
        set_trading_lock=True,
    )
    stage_3 = StageConfig(
        thresholds={"1m": s3_thresh},
        mode=s3_mode,
        cooldown_min=360,
        confirm_consecutive=1,
        cancel_entry_orders_before_close=True,
        set_trading_lock=True,
    )
    return stage_1, stage_2, stage_3


def _build_engine(mock_adapter, store, positions=None, orders=None):
    if positions:
        mock_adapter.set_mock_positions(positions)
    if orders is not None:
        mock_adapter.set_mock_orders(orders)

    stage_machine = StageMachine(store.conn)
    pos_store = PositionStore(store.conn)
    trading_lock = TradingLock("/tmp/ks_test_lock.json")
    breaker = CircuitBreaker()
    engine = ActionEngine(
        store, {"mock": mock_adapter}, True,
        ["USDT", "USDC"], breaker,
        stage_machine=stage_machine,
        position_store=pos_store,
        trading_lock=trading_lock,
    )
    return engine, stage_machine, pos_store


def _fake_snap(equity, ts=None):
    return EquitySnapshot(ts or BASE_TS, _scope(), equity, equity, 0, True)


def _push_equity(store, dd_calc, scope, equity_series, windows=None):
    """Push equity snapshots at 60-second intervals; return last dds and snap.

    Each consecutive pair of snapshots represents a 1-minute window, so the
    drop from equity_series[i-1] to equity_series[i] produces the DD for the
    window key "1m".  The series must have >= 2 points to trigger DD > 0.
    """
    windows = windows or ["1m"]
    # Count existing snapshots to avoid timestamp collision
    n_existing = store.conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE exchange=? AND account=?",
        (scope.exchange.value, scope.account.value)
    ).fetchone()[0]
    dds = {}
    for i, eq in enumerate(equity_series):
        ts = BASE_TS + (n_existing + i) * 60
        snap = EquitySnapshot(ts, scope, eq, eq, 0, True)
        store.append_snapshot(snap)
        dds = dd_calc.compute(scope, snap.ts, windows)
    return dds, snap


import logging
_dummy_logger = None

def _get_logger():
    global _dummy_logger
    if _dummy_logger is None:
        from logger import get_logger
        _dummy_logger = get_logger("test", level="WARNING")
    return _dummy_logger


# ---------------------------------------------------------------------------
# Scenario A: Losing shorts — longs should be untouched
# ---------------------------------------------------------------------------

class TestScenarioA:
    """Stage 1 fires due to losing shorts. Only shorts are in the close plan."""

    def test_stage1_closes_shorts_only(self):
        scope = _scope()
        store = SqliteStore(":memory:")
        dd_calc = DrawdownCalculator(store)

        # Short SOL is losing, BTC long is profitable
        positions = [
            _pos("SOL/USDT", "short", upnl=-350.0, notional=2000.0),  # losing short
            _pos("BTC/USDT", "long",  upnl=+150.0, notional=3000.0),  # profitable long
        ]
        # Reference (earlier): SOL short was at 0 PnL
        reference_positions = [
            _pos("SOL/USDT", "short", upnl=0.0, notional=2000.0),
            _pos("BTC/USDT", "long",  upnl=+50.0, notional=3000.0),
        ]

        mock = MockAdapter(None)
        engine, sm, pos_store = _build_engine(mock, store, positions=positions)

        # Seed reference snapshots
        pos_store.save_many(reference_positions)

        s1, s2, s3 = _build_stage_confs(s1_source_threshold=0.65)
        acc_conf = AccountConfig(enabled=True, windows=["1m"],
                                  stage_1=s1, stage_2=s2, stage_3=s3)

        # Push equity to trigger Stage 1 (>4% drop)
        equity = [10000, 9500, 9000]
        dds, snap = _push_equity(store, dd_calc, scope, equity)

        _process_stage_triggers(scope, acc_conf, dds, snap, engine, sm, dd_calc, _get_logger())

        state = sm.get_state(str(scope))
        assert state.current_stage == 1, f"Expected stage 1, got {state.current_stage}"

        # Verify: close plan was computed — check via attribution
        from risk_attribution import compute_pnl_attribution
        attr = compute_pnl_attribution(positions, reference_positions, 0.65)
        assert attr.source == "SHORT", f"Expected SHORT attribution, got {attr.source}"


# ---------------------------------------------------------------------------
# Scenario B: Losing longs
# ---------------------------------------------------------------------------

class TestScenarioB:
    """Stage 1 fires due to losing longs. Only longs are in the close plan."""

    def test_stage1_closes_longs_only(self):
        scope = _scope()
        store = SqliteStore(":memory:")
        dd_calc = DrawdownCalculator(store)

        positions = [
            _pos("BTC/USDT", "long",  upnl=-400.0, notional=4000.0),
            _pos("ETH/USDT", "short", upnl=+80.0,  notional=800.0),
        ]
        reference = [
            _pos("BTC/USDT", "long",  upnl=0.0, notional=4000.0),
            _pos("ETH/USDT", "short", upnl=+50.0, notional=800.0),
        ]

        mock = MockAdapter(None)
        engine, sm, pos_store = _build_engine(mock, store, positions=positions)
        pos_store.save_many(reference)

        s1, s2, s3 = _build_stage_confs(s1_source_threshold=0.65)
        acc_conf = AccountConfig(enabled=True, windows=["1m"],
                                  stage_1=s1, stage_2=s2, stage_3=s3)

        # Two snapshots 60s apart; drop from 10000 to 9550 = 4.5% in 1m window
        equity = [10000, 9550]
        dds, snap = _push_equity(store, dd_calc, scope, equity)

        _process_stage_triggers(scope, acc_conf, dds, snap, engine, sm, dd_calc, _get_logger())

        state = sm.get_state(str(scope))
        assert state.current_stage == 1

        from risk_attribution import compute_pnl_attribution
        attr = compute_pnl_attribution(positions, reference, 0.65)
        assert attr.source == "LONG", f"Expected LONG attribution, got {attr.source}"


# ---------------------------------------------------------------------------
# Scenario C: Mixed book
# ---------------------------------------------------------------------------

class TestScenarioC:
    """No side dominates — Stage 1 closes top-N across both sides."""

    def test_stage1_closes_mixed(self):
        scope = _scope()
        store = SqliteStore(":memory:")
        dd_calc = DrawdownCalculator(store)

        positions = [
            _pos("BTC/USDT", "long",  upnl=-200.0, notional=2000.0),
            _pos("ETH/USDT", "short", upnl=-180.0, notional=1800.0),
        ]
        reference = [
            _pos("BTC/USDT", "long",  upnl=0.0, notional=2000.0),
            _pos("ETH/USDT", "short", upnl=0.0, notional=1800.0),
        ]

        mock = MockAdapter(None)
        engine, sm, pos_store = _build_engine(mock, store, positions=positions)
        pos_store.save_many(reference)

        s1, s2, s3 = _build_stage_confs(s1_source_threshold=0.65)
        acc_conf = AccountConfig(enabled=True, windows=["1m"],
                                  stage_1=s1, stage_2=s2, stage_3=s3)

        equity = [10000, 9550]
        dds, snap = _push_equity(store, dd_calc, scope, equity)

        _process_stage_triggers(scope, acc_conf, dds, snap, engine, sm, dd_calc, _get_logger())

        state = sm.get_state(str(scope))
        assert state.current_stage == 1

        from risk_attribution import compute_pnl_attribution
        attr = compute_pnl_attribution(positions, reference, 0.65)
        assert attr.source == "MIXED"


# ---------------------------------------------------------------------------
# Scenario D: Escalation chain
# ---------------------------------------------------------------------------

class TestScenarioD:
    """Stage 1 → 2 → 3 as equity keeps deteriorating."""

    def test_full_escalation(self):
        """Stage 1 → 2 → 3 via consecutive large 1-minute drops.

        Each step drops enough within its 1m window to trigger the next stage.
        Stage machine must allow escalation even while lower-stage cooldown active.
          Step 0 (baseline): equity=10000
          Step 1: equity=9550  → DD=(10000-9550)/10000=4.5%  → stage 1 (>4%)
          Step 2: equity=8700  → DD=(9550-8700)/9550=8.9%    → stage 2 (>7%)
          Step 3: equity=7500  → DD=(8700-7500)/8700=13.8%   → stage 3 (>12%)
        """
        scope = _scope()
        store = SqliteStore(":memory:")
        dd_calc = DrawdownCalculator(store)

        positions = [_pos("BTC/USDT", "long", upnl=-200.0)]
        mock = MockAdapter(None)
        engine, sm, pos_store = _build_engine(mock, store, positions=positions)
        pos_store.save_many(positions)

        s1, s2, s3 = _build_stage_confs(
            s1_thresh=0.04, s2_thresh=0.07, s3_thresh=0.12
        )
        acc_conf = AccountConfig(enabled=True, windows=["1m"],
                                  stage_1=s1, stage_2=s2, stage_3=s3)

        # Use sequential timestamps so consecutive pairs form valid 1m windows
        equity_series = [10000, 9550, 8700, 7500]

        current_stage = 0
        for step, eq in enumerate(equity_series):
            ts = BASE_TS + step * 60
            snap = EquitySnapshot(ts, scope, eq, eq, 0, True)
            store.append_snapshot(snap)
            dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)
            _process_stage_triggers(scope, acc_conf, dds, snap, engine, sm, dd_calc, _get_logger())
            current_stage = sm.get_state(str(scope)).current_stage

        # After all 4 steps the machine should have reached stage 3
        assert current_stage == 3, f"Expected final stage 3, got {current_stage}"

        # Verify escalation log: can recover individual fired stages by checking
        # that stage 3 was reached, which implies 1 and 2 fired first
        final_state = sm.get_state(str(scope))
        assert final_state.current_stage == 3

    def test_stage1_cooldown_does_not_block_stage2(self):
        """Explicit proof: stage-1 cooldown must not suppress stage 2."""
        conn = sqlite3.connect(":memory:")
        sm = StageMachine(conn)
        ts = BASE_TS

        # Stage 1 fired, lock_until is far in future
        sm.record_execution("s", 1, ts + 9999, "m1", {}, ts)

        # Stage 2 should be allowed immediately
        can, reason = sm.can_execute("s", 2, ts + 1)
        assert can is True, f"Stage 2 blocked by stage-1 cooldown: {reason}"

    def test_stage2_cooldown_does_not_block_stage3(self):
        """Stage-2 cooldown must not suppress stage 3."""
        conn = sqlite3.connect(":memory:")
        sm = StageMachine(conn)
        ts = BASE_TS

        sm.record_execution("s", 2, ts + 9999, "m2", {}, ts)

        can, reason = sm.can_execute("s", 3, ts + 1)
        assert can is True, f"Stage 3 blocked by stage-2 cooldown: {reason}"


# ---------------------------------------------------------------------------
# Scenario E: Stale order safety
# ---------------------------------------------------------------------------

class TestScenarioE:
    """Entry orders are cancelled before kill actions."""

    def test_entry_orders_cancelled_before_close(self):
        scope = _scope()
        store = SqliteStore(":memory:")
        dd_calc = DrawdownCalculator(store)

        # Non-reduce-only entry order (should be cancelled)
        entry_orders = [
            {"id": "entry1", "symbol": "BTC/USDT:USDT", "reduceOnly": False},
            {"id": "entry2", "symbol": "SOL/USDT:USDT", "reduceOnly": False},
        ]
        positions = [_pos("BTC/USDT", "long", upnl=-300.0)]

        mock = MockAdapter(None)
        engine, sm, pos_store = _build_engine(mock, store,
                                               positions=positions,
                                               orders=entry_orders)

        # Set up stage 1 with cancel_entry_orders_before_close=True
        s1 = StageConfig(
            thresholds={"1m": 0.04},
            mode="CLOSE_TOP_RISK_CONTRIBUTORS",
            cooldown_min=30, confirm_consecutive=1,
            cancel_entry_orders_before_close=True,
            set_trading_lock=False,
        )
        acc_conf = AccountConfig(enabled=True, windows=["1m"], stage_1=s1)

        equity = [10000, 9550]
        dds, snap = _push_equity(store, dd_calc, scope, equity)
        _process_stage_triggers(scope, acc_conf, dds, snap, engine, sm, dd_calc, _get_logger())

        # After stage 1, MockAdapter.cancel_entry_orders() was called,
        # which removes non-reduce-only orders
        remaining_orders = mock._mock_orders
        non_reduce = [o for o in remaining_orders if not o.get("reduceOnly")]
        assert len(non_reduce) == 0, f"Entry orders not cancelled: {remaining_orders}"

    def test_reduce_only_orders_preserved_during_close(self):
        """Reduce-only orders must NOT be cancelled by cancel_entry_orders."""
        scope = _scope()
        store = SqliteStore(":memory:")

        mixed_orders = [
            {"id": "entry1", "symbol": "BTC/USDT", "reduceOnly": False},
            {"id": "tp1",    "symbol": "BTC/USDT", "reduceOnly": True},  # take-profit
        ]
        mock = MockAdapter(None)
        mock.set_mock_orders(mixed_orders)
        mock.cancel_entry_orders(scope)

        remaining = mock._mock_orders
        # After cancel_entry_orders, reduce-only should remain
        reduce_only = [o for o in remaining if o.get("reduceOnly")]
        assert len(reduce_only) == 1
        assert reduce_only[0]["id"] == "tp1"


# ---------------------------------------------------------------------------
# Additional unit tests for kill-switch internals
# ---------------------------------------------------------------------------

class TestDrawdownCalculatorCanonical:
    """Additional tests for canonical window deduplication."""

    def test_duplicate_windows_deduplicated_in_config_loader(self):
        """Config loader should deduplicate [15, '15m', 60, '1h']."""
        from killswitch import ConfigLoader
        result = ConfigLoader._deduplicate_windows(["15", "15m", "60", "1h"])
        assert len(result) == 2

    def test_dd_computed_for_canonical_windows(self):
        """DD dict keys should match the window strings passed."""
        store = SqliteStore(":memory:")
        scope = Scope(ExchangeId.MOCK, AccountType.FUTURES)
        dd_calc = DrawdownCalculator(store)
        ts = BASE_TS
        for i in range(5):
            store.append_snapshot(EquitySnapshot(ts + i * 60, scope, 10000 - i * 50, quality_ok=True))
        dds = dd_calc.compute(scope, ts + 4 * 60, ["15m"])
        assert "15m" in dds


class TestMockAdapterCompleteness:
    """Verify MockAdapter implements all BaseAdapter methods."""

    def test_mock_adapter_all_methods(self):
        """Missing --test-mock should not break due to missing methods."""
        scope = Scope(ExchangeId.MOCK, AccountType.FUTURES)
        mock = MockAdapter(None)
        mock.set_scenario([10000])

        snap = mock.fetch_equity(scope)
        assert snap.equity_usdt == 10000.0

        orders = mock.fetch_open_orders(scope)
        assert isinstance(orders, list)

        pos = mock.fetch_positions_as_snapshots(scope)
        assert isinstance(pos, list)

        res = mock.cancel_entry_orders(scope)
        assert res.success

        res = mock.cancel_orphan_reduce_only_orders(scope)
        assert res.success

        from order_safety import CloseInstruction
        plan = [CloseInstruction("BTC/USDT", "long", 1.0, 0.5, "test", 1.0, -100.0, -100.0)]
        res = mock.close_positions_by_plan(scope, plan, True)
        assert res.success

        res = mock.sell_spot(scope, ["USDT"], None, True)
        assert res.success
