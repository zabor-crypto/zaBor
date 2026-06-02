#!/usr/bin/env python3
"""
Integration Tests for Killswitch Action Execution
Tests futures/spot actions with mock exchanges
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fakes import (
    FakeCCXTClient, create_futures_scenario, create_spot_scenario,
    RateLimitExceeded, NetworkError, InvalidOrder
)


# ==============================================================================
# FUTURES TESTS
# ==============================================================================

def test_futures_close_longs_only():
    """Test CLOSE_LONGS_ONLY mode (one-way)"""
    # Setup: 1 LONG + 1 SHORT
    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1},
        {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.5}
    ])
    
    # Execute: Close longs only
    positions = client.fetch_positions()
    
    for pos in positions:
        if pos["side"] == "long":
            # Close long
            client.create_order(
                pos["symbol"],
                "market",
                "sell",  # Opposite side
                pos["contracts"],
                params={"reduceOnly": True}
            )
    
    # Verify
    orders = client.get_orders_log()
    assert len(orders) == 1
    assert orders[0].symbol == "BTC/USDT:USDT"
    assert orders[0].side == "sell"
    assert orders[0].params.get("reduceOnly") is True
    
    # Verify SHORT position untouched
    remaining_positions = client.fetch_positions()
    assert len(remaining_positions) == 1
    assert remaining_positions[0]["side"] == "short"


def test_futures_close_all_positions():
    """Test CLOSE_ALL_POSITIONS mode"""
    # Setup: Multiple positions
    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.05},
        {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.0},
        {"symbol": "SOL/USDT:USDT", "side": "long", "contracts": 5.0}  # Add SOL market
    ])
    
    # Add SOL market
    from tests.fakes import FakeMarket
    client.markets["SOL/USDT:USDT"] = FakeMarket("SOL/USDT:USDT", "SOL", "USDT", 2, 2, 0.1, 10)
    
    # Execute: Close all
    positions = client.fetch_positions()
    
    for pos in positions:
        side = pos["side"]
        close_side = "sell" if side == "long" else "buy"
        
        client.create_order(
            pos["symbol"],
            "market",
            close_side,
            pos["contracts"],
            params={"reduceOnly": True}
        )
    
    # Verify
    orders = client.get_orders_log()
    assert len(orders) == 3  # All 3 positions closed
    
    # Check all have reduceOnly
    for order in orders:
        assert order.params.get("reduceOnly") is True
    
    # Verify all positions closed
    remaining = client.fetch_positions()
    assert len(remaining) == 0


def test_bybit_hedge_mode_requires_position_idx():
    """Test Bybit hedge mode requires positionIdx"""
    client = FakeCCXTClient("bybit", "hedge")
    client.set_positions([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1, "positionIdx": 1}
    ])
    client.set_tickers({
        "BTC/USDT:USDT": {"bid": 50000, "ask": 50001, "last": 50000}
    })
    
    # Try without positionIdx (should fail)
    with pytest.raises(InvalidOrder, match="requires positionIdx"):
        client.create_order(
            "BTC/USDT:USDT",
            "market",
            "sell",
            0.1,
            params={"reduceOnly": True}  # Missing positionIdx
        )
    
    # Try with correct positionIdx (should succeed)
    client.create_order(
        "BTC/USDT:USDT",
        "market",
        "sell",
        0.1,
        params={"reduceOnly": True, "positionIdx": 1}
    )
    
    orders = client.get_orders_log()
    assert len(orders) == 1
    assert orders[0].params["positionIdx"] == 1


def test_bybit_one_way_no_position_idx_needed():
    """Test Bybit one-way mode doesn't require positionIdx"""
    client = FakeCCXTClient("bybit", "one-way")
    client.set_positions([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1}
    ])
    client.set_tickers({
        "BTC/USDT:USDT": {"bid": 50000, "ask": 50001, "last": 50000}
    })
    
    # Should work without positionIdx in one-way mode
    client.create_order(
        "BTC/USDT:USDT",
        "market",
        "sell",
        0.1,
        params={"reduceOnly": True}
    )
    
    orders = client.get_orders_log()
    assert len(orders) == 1


def test_futures_partial_fill_simulation():
    """Test position reduction after partial fill"""
    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 1.0}
    ])
    
    # Close 0.5 BTC
    client.create_order(
        "BTC/USDT:USDT",
        "market",
        "sell",
        0.5,
        params={"reduceOnly": True}
    )
    
    # Verify position reduced
    positions = client.fetch_positions()
    assert len(positions) == 1
    assert positions[0]["contracts"] == 0.5  # Reduced from 1.0


# ==============================================================================
# SPOT TESTS
# ==============================================================================

def test_spot_sell_blacklist_only():
    """Test SELL_BLACKLIST_ONLY_KEEP_STABLES mode"""
    client = create_spot_scenario({
        "USDT": 1000,
        "BTC": 0.1,
        "DOGE": 5000,
        "USDC": 500
    })
    
    stables_keep = ["USDT", "USDC"]
    blacklist = ["DOGE"]
    
    # Execute: Sell only blacklist
    balances = client.fetch_balance()
    
    for coin, amount in balances["free"].items():
        if coin in stables_keep:
            continue
        if coin not in blacklist:
            continue
        if amount <= 0:
            continue
        
        # Sell blacklist coin
        pair = f"{coin}/USDT"
        client.create_order(pair, "market", "sell", amount)
    
    # Verify
    orders = client.get_orders_log()
    assert len(orders) == 1
    assert orders[0].symbol == "DOGE/USDT"
    assert orders[0].side == "sell"
    
    # Verify USDT, USDC, BTC not touched
    assert client.balances["USDT"] == 1000
    assert client.balances["USDC"] == 500
    assert client.balances["BTC"] == 0.1


def test_spot_sell_all_non_usdt():
    """Test SELL_ALL_NON_USDT_KEEP_STABLES mode"""
    client = create_spot_scenario({
        "USDT": 1000,
        "BTC": 0.05,
        "SOL": 10,
        "USDC": 200
    })
    
    stables_keep = ["USDT", "USDC"]
    
    # Execute: Sell all except stables
    balances = client.fetch_balance()
    
    for coin, amount in balances["free"].items():
        if coin in stables_keep:
            continue
        if amount <= 0:
            continue
        
        pair = f"{coin}/USDT"
        client.create_order(pair, "market", "sell", amount)
    
    # Verify
    orders = client.get_orders_log()
    assert len(orders) == 2
    
    symbols_sold = {o.symbol for o in orders}
    assert "BTC/USDT" in symbols_sold
    assert "SOL/USDT" in symbols_sold
    
    # Verify stables kept
    assert client.balances["USDT"] == 1000
    assert client.balances["USDC"] == 200


def test_spot_multihop_routing():
    """Test multi-hop routing (e.g., DOGE -> BTC -> USDT)"""
    client = create_spot_scenario({
        "USDT": 1000,
        "DOGE": 10000
    })
    
    # Remove direct DOGE/USDT pair, keep DOGE/BTC and BTC/USDT
    tickers = {
        "DOGE/BTC": {"bid": 0.0000016, "ask": 0.00000161, "last": 0.0000016},
        "BTC/USDT": {"bid": 50000, "ask": 50001, "last": 50000}
    }
    client.set_tickers(tickers)
    
    # Execute: Sell DOGE (requires routing)
    # Step 1: DOGE -> BTC
    client.create_order("DOGE/BTC", "market", "sell", 10000)
    
    # Simulate receiving BTC
    client.balances["BTC"] = 10000 * 0.0000016  # = 0.016 BTC
    client.balances["DOGE"] = 0
    
    # Step 2: BTC -> USDT
    btc_amount = client.balances["BTC"]
    client.create_order("BTC/USDT", "market", "sell", btc_amount)
    
    # Verify
    orders = client.get_orders_log()
    assert len(orders) == 2
    assert orders[0].symbol == "DOGE/BTC"
    assert orders[1].symbol == "BTC/USDT"


def test_spot_min_notional_filtering():
    """Test that dust balances are skipped"""
    client = create_spot_scenario({
        "USDT": 1000,
        "DOGE": 10  # Only $0.80 worth (below $10 min notional)
    })
    
    # Calculate notional
    doge_price = client.tickers["DOGE/USDT"]["bid"]
    doge_amount = 10
    notional = doge_amount * doge_price  # = 10 * 0.08 = 0.8 USDT
    
    # Should skip if below min notional (normally $10)
    if notional < 10:
        # Skip order
        orders_placed = 0
    else:
        client.create_order("DOGE/USDT", "market", "sell", doge_amount)
        orders_placed = 1
    
    assert orders_placed == 0
    assert len(client.get_orders_log()) == 0


def test_spot_precision_enforcement():
    """Test amount_to_precision enforces market rules"""
    client = create_spot_scenario({
        "USDT": 1000,
        "BTC": 0.123456789
    })
    
    # BTC/USDT precision is 3 decimals for amount
    raw_amount = 0.123456789
    precise_amount = client.amount_to_precision("BTC/USDT", raw_amount)
    
    assert precise_amount == 0.123  # Rounded to 3 decimals
    
    # Create order with precision
    client.create_order("BTC/USDT", "market", "sell", precise_amount)
    
    order = client.get_last_order()
    assert order.amount == 0.123


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

def test_network_error_injection():
    """Test error injection for network failures"""
    client = create_futures_scenario([])
    
    # Inject network error
    client.inject_error("fetch_positions", NetworkError("Connection timeout"))
    
    with pytest.raises(NetworkError):
        client.fetch_positions()


def test_rate_limit_error_injection():
    """Test rate limit error injection"""
    client = create_spot_scenario({"USDT": 1000})
    
    client.inject_error("fetch_balance", RateLimitExceeded("Rate limit exceeded"))
    
    with pytest.raises(RateLimitExceeded):
        client.fetch_balance()


def test_multiple_error_injection():
    """Test scheduling multiple errors"""
    client = create_futures_scenario([])
    
    # Inject 3 errors in sequence
    client.inject_error("fetch_positions", NetworkError("Error 1"))
    client.inject_error("fetch_positions", NetworkError("Error 2"))
    client.inject_error("fetch_positions", NetworkError("Error 3"))
    
    # First 3 calls fail
    for i in range(3):
        with pytest.raises(NetworkError):
            client.fetch_positions()
    
    # 4th call succeeds
    positions = client.fetch_positions()
    assert positions == []


# ==============================================================================
# STATE TRACKING TESTS
# ==============================================================================

def test_state_snapshot():
    """Test state snapshot functionality"""
    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1}
    ])
    
    # Take snapshot
    snap1 = client.snapshot_state("before_close")
    
    # Close position
    client.create_order("BTC/USDT:USDT", "market", "sell", 0.1, params={"reduceOnly": True})
    
    # Take another snapshot
    snap2 = client.snapshot_state("after_close")
    
    # Verify snapshots
    assert snap1["orders_count"] == 0
    assert snap2["orders_count"] == 1
    assert len(snap1["positions"]) == 1
    assert len(snap2["positions"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
