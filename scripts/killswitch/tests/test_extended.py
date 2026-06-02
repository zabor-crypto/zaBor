#!/usr/bin/env python3
"""
Extended Test Coverage for Killswitch
Additional edge cases, stress tests, and advanced scenarios.
"""

import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from killswitch import (
    DrawdownCalculator, SqliteStore, Scope, ExchangeId, AccountType,
    EquitySnapshot, TierConfig
)
from tests.fakes import FakeCCXTClient, create_futures_scenario


# ==============================================================================
# EXTENDED CONFIRMATION TESTS
# ==============================================================================

def test_is_confirmed_with_3_consecutive():
    """Test confirmation with 3 consecutive breaches"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Create sustained drop
    for i in range(15):
        equity = 10000 if i < 5 else 9500
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 14*60
    threshold = 0.04  # 4%
    window = 10
    consecutive = 3
    
    confirmed = dd_calc.is_confirmed(scope, threshold, window, now, consecutive)
    assert confirmed is True


def test_is_confirmed_with_recovery_in_middle():
    """Test that recovery in middle breaks consecutive streak"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Pattern: drop, recover briefly, drop again
    pattern = [10000]*5 + [9500]*3 + [9900] + [9500]*6
    for i, equity in enumerate(pattern):
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + (len(pattern)-1)*60
    threshold = 0.04
    window = 10
    consecutive = 3
    
    # Should NOT confirm due to recovery spike
    confirmed = dd_calc.is_confirmed(scope, threshold, window, now, consecutive)
    assert confirmed is False


def test_is_confirmed_edge_case_exact_threshold():
    """Test behavior when DD exactly equals threshold"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Create exactly 5% drop
    for i in range(10):
        equity = 10000 if i < 5 else 9500  # Exactly 5% drop
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 9*60
    threshold = 0.05  # Exactly 5%
    window = 10
    consecutive = 2
    
    # Should confirm (>= threshold)
    confirmed = dd_calc.is_confirmed(scope, threshold, window, now, consecutive)
    assert confirmed is True


# ==============================================================================
# TIER SELECTION EDGE CASES
# ==============================================================================

def test_tier_selection_multiple_windows_same_ratio():
    """Test tie-breaking when multiple windows have same DD/threshold ratio"""
    dd_calc = DrawdownCalculator(SqliteStore(":memory:"))
    
    # Both windows have exactly 2x ratio
    dds = {
        "5": 0.10,   # 10% DD
        "15": 0.04   # 4% DD
    }
    
    thresholds = {
        "5": 0.05,   # Ratio = 2.0
        "15": 0.02   # Ratio = 2.0
    }
    
    result = dd_calc.pick_trigger_window(dds, thresholds)
    
    # Should pick one (implementation dependent, but must be consistent)
    assert result is not None
    score, window, thr = result
    assert score == 2.0


def test_tier_selection_zero_threshold():
    """Test handling of zero or negative thresholds"""
    dd_calc = DrawdownCalculator(SqliteStore(":memory:"))
    
    dds = {"5": 0.10}
    
    # Zero threshold
    thresholds = {"5": 0.0}
    result = dd_calc.pick_trigger_window(dds, thresholds)
    # Should not trigger (divide by zero protection)
    assert result is None or result[0] == float('inf')


# ==============================================================================
# COOLDOWN EDGE CASES
# ==============================================================================

def test_cooldown_concurrent_tiers():
    """Test that multiple tiers can be in cooldown simultaneously"""
    store = SqliteStore(":memory:")
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    now = int(time.time())
    
    # Set cooldowns for multiple tiers
    assert store.try_set_cooldown(scope, "A", now + 3600) is True
    assert store.try_set_cooldown(scope, "B", now + 7200) is True
    assert store.try_set_cooldown(scope, "C", now + 1800) is True
    
    # All should be in cooldown
    assert store.in_cooldown(scope, "A", now + 100) is True
    assert store.in_cooldown(scope, "B", now + 100) is True
    assert store.in_cooldown(scope, "C", now + 100) is True


def test_cooldown_expiry_granular():
    """Test cooldown expiry at exact boundary"""
    store = SqliteStore(":memory:")
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    
    base_time = 1000000
    until = base_time + 60
    
    store.try_set_cooldown(scope, "A", until)
    
    # At expiry boundary (exclusive)
    assert store.in_cooldown(scope, "A", until) is False
    
    # Just before expiry
    assert store.in_cooldown(scope, "A", until - 1) is True
    
    # After expiry
    assert store.in_cooldown(scope, "A", until + 1) is False


# ==============================================================================
# STRESS TESTS
# ==============================================================================

def test_dd_compute_with_large_dataset():
    """Test DD calculation with 1000+ snapshots"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Insert 1000 snapshots over ~16 hours
    for i in range(1000):
        equity = 10000 - (i * 5)  # Gradual decline
        snap = EquitySnapshot(base_ts + i*60, scope, max(100, equity), quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 999*60
    
    # Compute DD for various windows
    dds = dd_calc.compute(scope, now, ["5", "15", "60", "240"])
    
    # Should complete without error
    assert "5" in dds
    assert "15" in dds
    assert "60" in dds
    assert "240" in dds
    
    # Longer windows should capture more DD
    assert dds["240"] > dds["60"]


def test_multiple_scopes_isolation():
    """Test that different scopes don't interfere"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    
    scope1 = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    scope2 = Scope(ExchangeId.BYBIT, AccountType.FUTURES)
    scope3 = Scope(ExchangeId.BINANCE, AccountType.SPOT)
    
    base_ts = int(time.time())
    
    # Insert different data for each scope
    for i in range(10):
        snap1 = EquitySnapshot(base_ts + i*60, scope1, 10000 - i*100, quality_ok=True)
        snap2 = EquitySnapshot(base_ts + i*60, scope2, 5000 - i*50, quality_ok=True)
        snap3 = EquitySnapshot(base_ts + i*60, scope3, 15000 - i*150, quality_ok=True)
        
        store.append_snapshot(snap1)
        store.append_snapshot(snap2)
        store.append_snapshot(snap3)
    
    now = base_ts + 540
    
    # Compute DD for each scope
    dds1 = dd_calc.compute(scope1, now, ["10"])
    dds2 = dd_calc.compute(scope2, now, ["10"])
    dds3 = dd_calc.compute(scope3, now, ["10"])
    
    # Each should have different values
    assert dds1["10"] != dds2["10"]
    assert dds2["10"] != dds3["10"]
    assert dds1["10"] != dds3["10"]


# ==============================================================================
# REAL-WORLD SCENARIOS
# ==============================================================================

def test_flash_crash_scenario():
    """Test handling of flash crash (rapid drop and recovery)"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Pattern: stable → flash crash → recovery
    pattern = (
        [10000] * 10 +      # Stable at 10k
        [9000, 8000, 7000] +  # Flash crash
        [9500, 9800, 9900] +  # Recovery
        [10000] * 5          # Back to normal
    )
    
    for i, equity in enumerate(pattern):
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    # Check different points in time
    crash_time = base_ts + 12*60  # During crash
    recovery_time = base_ts + 18*60  # After recovery
    
    dds_crash = dd_calc.compute(scope, crash_time, ["5"])
    dds_recovery = dd_calc.compute(scope, recovery_time, ["5"])
    
    # During crash, DD should be significant
    assert dds_crash["5"] >= 0.20  # At least 20%
    
    # After recovery, recent DD should be lower
    assert dds_recovery["5"] < dds_crash["5"]


def test_gradual_bleed_scenario():
    """Test slow gradual equity decline"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Very gradual decline: -0.5% per hour for 20 hours
    for i in range(1200):  # 20 hours * 60 minutes
        equity = 10000 * (0.995 ** (i // 60))  # -0.5% each hour
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 1199*60
    
    # Check various window DDs
    dds = dd_calc.compute(scope, now, ["60", "240", "720"])
    
    # 1-hour window: ~0.5%
    assert 0.004 <= dds["60"] <= 0.006
    
    # 4-hour window: ~2%
    assert 0.018 <= dds["240"] <= 0.022
    
    # 12-hour window: ~6%
    assert 0.055 <= dds["720"] <= 0.065


# ==============================================================================
# PRECISION AND EDGE CASES
# ==============================================================================

def test_very_small_equity_values():
    """Test with very small equity values (cents)"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Very small values (in cents range)
    for i in range(10):
        equity = 1.00 - (i * 0.05)  # Drop from $1 to $0.50
        snap = EquitySnapshot(base_ts + i*60, scope, max(0.01, equity), quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 540
    dds = dd_calc.compute(scope, now, ["10"])
    
    # Should still calculate DD correctly
    # (1.0 - 0.55) / 1.0 = 0.45 = 45%
    assert dds["10"] >= 0.40
    assert dds["10"] <= 0.50


def test_very_large_equity_values():
    """Test with very large equity values (millions)"""
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    scope = Scope(ExchangeId.BINANCE, AccountType.FUTURES)
    base_ts = int(time.time())
    
    # Large values (millions)
    for i in range(10):
        equity = 1_000_000 - (i * 50_000)  # Drop from 1M to 550K
        snap = EquitySnapshot(base_ts + i*60, scope, equity, quality_ok=True)
        store.append_snapshot(snap)
    
    now = base_ts + 540
    dds = dd_calc.compute(scope, now, ["10"])
    
    # (1M - 550K) / 1M = 0.45 = 45%
    assert dds["10"] >= 0.40
    assert dds["10"] <= 0.50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
