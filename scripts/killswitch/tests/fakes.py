#!/usr/bin/env python3
"""
Mock CCXT Exchange Clients for Offline Testing
Provides fully controllable fake exchanges with error injection and state tracking.
"""

import time
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ==============================================================================
# FAKE CCXT EXCEPTIONS (mimicking real ccxt errors)
# ==============================================================================

class ExchangeError(Exception):
    """Base exchange error"""
    pass


class RateLimitExceeded(ExchangeError):
    """Rate limit error"""
    pass


class NetworkError(ExchangeError):
    """Network connectivity error"""
    pass


class InvalidOrder(ExchangeError):
    """Invalid order parameters"""
    pass


# ==============================================================================
# FAKE CCXT CLIENT
# ==============================================================================

@dataclass
class FakeMarket:
    """Mock market info"""
    symbol: str
    base: str
    quote: str
    precision_amount: int = 3
    precision_price: int = 2
    min_amount: float = 0.001
    min_notional: float = 10.0


@dataclass
class OrderLog:
    """Logged order entry"""
    ts: int
    symbol: str
    type: str
    side: str
    amount: float
    price: Optional[float]
    params: Dict[str, Any]


class FakeCCXTClient:
    """
    Mock CCXT exchange client with full control over responses.
    Supports:
    - Scenario-driven balances/positions/prices
    - Error injection
    - One-way vs Hedge mode simulation (Bybit)
    - Order logging
    - Precision handling
    """
    
    def __init__(self, exchange_id: str = "mock", mode: str = "one-way"):
        self.id = exchange_id
        self.mode = mode  # "one-way" or "hedge"
        
        # State
        self.balances = {"USDT": 10000.0}
        self.positions = []
        self.tickers = {}
        self.markets = self._init_markets()
        
        # Logs
        self.orders_log: List[OrderLog] = []
        self.state_log: List[Dict] = []
        
        # Error injection
        self.error_schedule: Dict[str, List[Exception]] = defaultdict(list)
        
    def _init_markets(self) -> Dict[str, FakeMarket]:
        """Initialize common markets"""
        return {
            "BTC/USDT": FakeMarket("BTC/USDT", "BTC", "USDT", 3, 2, 0.001, 10),
            "BTC/USDT:USDT": FakeMarket("BTC/USDT:USDT", "BTC", "USDT", 3, 1, 0.001, 10),
            "ETH/USDT": FakeMarket("ETH/USDT", "ETH", "USDT", 3, 2, 0.01, 10),
            "ETH/USDT:USDT": FakeMarket("ETH/USDT:USDT", "ETH", "USDT", 3, 1, 0.01, 10),
            "DOGE/USDT": FakeMarket("DOGE/USDT", "DOGE", "USDT", 1, 5, 10, 10),
            "SOL/USDT": FakeMarket("SOL/USDT", "SOL", "USDT", 2, 2, 0.1, 10),
            "BTC/BTC": FakeMarket("BTC/BTC", "BTC", "BTC", 8, 8, 0.00001, 0.0001),
            "DOGE/BTC": FakeMarket("DOGE/BTC", "DOGE", "BTC", 1, 8, 10, 0.0001),
            "ETH/BTC": FakeMarket("ETH/BTC", "ETH", "BTC", 3, 6, 0.01, 0.0001),
        }
    
    # ==========================================================================
    # ERROR INJECTION
    # ==========================================================================
    
    def inject_error(self, method_name: str, error: Exception):
        """Schedule an error for the next call to method_name"""
        self.error_schedule[method_name].append(error)
    
    def _check_error(self, method_name: str):
        """Check if error should be raised for this method"""
        if self.error_schedule[method_name]:
            error = self.error_schedule[method_name].pop(0)
            raise error
    
    # ==========================================================================
    # SCENARIO LOADING
    # ==========================================================================
    
    def load_scenario(self, scenario: Dict[str, Any]):
        """Load a scenario from JSON"""
        if "balances" in scenario:
            self.balances = scenario["balances"]
        if "positions" in scenario:
            self.positions = scenario["positions"]
        if "tickers" in scenario:
            self.tickers = scenario["tickers"]
    
    def set_balances(self, balances: Dict[str, float]):
        """Set spot balances"""
        self.balances = balances.copy()
    
    def set_positions(self, positions: List[Dict]):
        """Set futures positions"""
        self.positions = positions.copy()
    
    def set_tickers(self, tickers: Dict[str, Dict]):
        """Set price tickers"""
        self.tickers = tickers.copy()
    
    # ==========================================================================
    # CCXT API METHODS
    # ==========================================================================
    
    def load_markets(self):
        """Load market information"""
        self._check_error("load_markets")
        return self.markets
    
    def fetch_balance(self, params: Optional[Dict] = None) -> Dict:
        """Fetch balance (spot or futures)"""
        self._check_error("fetch_balance")
        
        params = params or {}
        
        # Futures balance (similar to real exchanges)
        if params.get("type") == "swap" or params.get("type") == "future":
            # Simulate futures balance response
            total_equity = self.balances.get("USDT", 0)
            return {
                "total": {"USDT": total_equity},
                "free": {"USDT": total_equity},
                "used": {"USDT": 0},
                "info": {
                    "totalWalletBalance": str(total_equity),
                    "totalUnrealizedProfit": "0",
                    # Bybit V5 format
                    "result": {
                        "list": [{
                            "totalEquity": str(total_equity),
                            "totalWalletBalance": str(total_equity),
                            "totalPerpUPL": "0"
                        }]
                    }
                }
            }
        
        # Spot balance
        return {
            "total": self.balances.copy(),
            "free": self.balances.copy(),
            "used": {k: 0 for k in self.balances},
            "info": {}
        }
    
    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch futures positions"""
        self._check_error("fetch_positions")
        
        result = []
        for pos in self.positions:
            # Filter by symbols if provided
            if symbols and pos["symbol"] not in symbols:
                continue
            
            # Build position dict in ccxt format
            result.append({
                "symbol": pos["symbol"],
                "side": pos["side"],
                "contracts": pos.get("contracts", 0),
                "contractSize": 1,
                "info": {
                    "size": str(pos.get("contracts", 0)),
                    "positionIdx": pos.get("positionIdx", 0)
                }
            })
        
        return result
    
    def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """Fetch price tickers"""
        self._check_error("fetch_tickers")
        
        if symbols:
            return {k: v for k, v in self.tickers.items() if k in symbols}
        return self.tickers.copy()
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch single ticker"""
        self._check_error("fetch_ticker")
        return self.tickers.get(symbol, {"bid": 0, "ask": 0, "last": 0})
    
    def create_order(
        self, 
        symbol: str, 
        type: str, 
        side: str, 
        amount: float, 
        price: Optional[float] = None, 
        params: Optional[Dict] = None
    ) -> Dict:
        """Create order (logs order, validates params, simulates execution)"""
        self._check_error("create_order")
        
        params = params or {}
        
        # Validate Bybit hedge mode positionIdx
        if self.id == "bybit" and self.mode == "hedge":
            position_idx = params.get("positionIdx")
            # In hedge mode, positionIdx is required and must be 1 (long) or 2 (short)
            if position_idx is None:
                raise InvalidOrder("Bybit hedge mode requires positionIdx parameter")
            if position_idx not in [0, 1, 2]:
                raise InvalidOrder(f"Invalid positionIdx: {position_idx}")
        
        # Validate reduceOnly for futures
        if ":USDT" in symbol or "PERP" in symbol:
            if not params.get("reduceOnly"):
                # Allow it but warn
                pass
        
        # Log order
        order_log = OrderLog(
            ts=int(time.time()),
            symbol=symbol,
            type=type,
            side=side,
            amount=amount,
            price=price,
            params=params.copy()
        )
        self.orders_log.append(order_log)
        
        # Simulate execution (update state)
        self._execute_order(symbol, side, amount, params)
        
        return {
            "id": f"ORDER_{len(self.orders_log)}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "filled",
            "info": params
        }
    
    def _execute_order(self, symbol: str, side: str, amount: float, params: Dict):
        """Simulate order execution (update positions/balances)"""
        # For reduceOnly orders, reduce position
        if params.get("reduceOnly"):
            # Find matching position
            for i, pos in enumerate(self.positions):
                if pos["symbol"] == symbol:
                    # Check if sides match for closing
                    pos_side = pos["side"]
                    if (pos_side == "long" and side == "sell") or (pos_side == "short" and side == "buy"):
                        # Reduce position
                        current_size = pos.get("contracts", 0)
                        new_size = max(0, current_size - amount)
                        
                        if new_size == 0:
                            # Position closed
                            self.positions.pop(i)
                        else:
                            self.positions[i]["contracts"] = new_size
                        break
        
        # For spot, update balances
        if "/" in symbol and ":" not in symbol:
            base, quote = symbol.split("/")
            if side == "sell":
                # Selling base for quote
                self.balances[base] = self.balances.get(base, 0) - amount
                # Simplified: don't add quote (would need price)
            elif side == "buy":
                # Buying base with quote
                self.balances[base] = self.balances.get(base, 0) + amount
    
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Apply amount precision rules"""
        market = self.markets.get(symbol)
        if not market:
            return round(amount, 3)
        
        precision = market.precision_amount
        return round(amount, precision)
    
    def price_to_precision(self, symbol: str, price: float) -> float:
        """Apply price precision rules"""
        market = self.markets.get(symbol)
        if not market:
            return round(price, 2)
        
        precision = market.precision_price
        return round(price, precision)
    
    # ==========================================================================
    # TEST UTILITIES
    # ==========================================================================
    
    def get_orders_log(self) -> List[OrderLog]:
        """Get all logged orders"""
        return self.orders_log.copy()
    
    def get_last_order(self) -> Optional[OrderLog]:
        """Get last logged order"""
        return self.orders_log[-1] if self.orders_log else None
    
    def clear_logs(self):
        """Clear all logs"""
        self.orders_log.clear()
        self.state_log.clear()
    
    def snapshot_state(self, label: str = ""):
        """Snapshot current state for testing"""
        snapshot = {
            "label": label,
            "ts": int(time.time()),
            "balances": self.balances.copy(),
            "positions": self.positions.copy(),
            "orders_count": len(self.orders_log)
        }
        self.state_log.append(snapshot)
        return snapshot


# ==============================================================================
# SCENARIO LOADER
# ==============================================================================

class ScenarioLoader:
    """Load test scenarios from JSON into FakeCCXTClient"""
    
    @staticmethod
    def load_from_file(path: str, client: FakeCCXTClient):
        """Load scenario from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        client.load_scenario(data)
        return data
    
    @staticmethod
    def load_from_dict(data: Dict, client: FakeCCXTClient):
        """Load scenario from dict"""
        client.load_scenario(data)
        return data


# ==============================================================================
# TEST HELPERS
# ==============================================================================

def create_futures_scenario(
    positions: List[Dict],
    equity: float = 10000.0,
    tickers: Optional[Dict] = None
) -> FakeCCXTClient:
    """Helper: Create futures testing scenario"""
    client = FakeCCXTClient("mock", "one-way")
    client.set_balances({"USDT": equity})
    client.set_positions(positions)
    
    if tickers:
        client.set_tickers(tickers)
    else:
        # Default tickers
        client.set_tickers({
            "BTC/USDT:USDT": {"bid": 50000, "ask": 50001, "last": 50000},
            "ETH/USDT:USDT": {"bid": 3000, "ask": 3001, "last": 3000}
        })
    
    return client


def create_spot_scenario(
    balances: Dict[str, float],
    tickers: Optional[Dict] = None
) -> FakeCCXTClient:
    """Helper: Create spot testing scenario"""
    client = FakeCCXTClient("mock", "one-way")
    client.set_balances(balances)
    
    if tickers:
        client.set_tickers(tickers)
    else:
        # Default tickers
        client.set_tickers({
            "BTC/USDT": {"bid": 50000, "ask": 50001, "last": 50000},
            "ETH/USDT": {"bid": 3000, "ask": 3001, "last": 3000},
            "DOGE/USDT": {"bid": 0.08, "ask": 0.0801, "last": 0.08},
            "SOL/USDT": {"bid": 100, "ask": 100.01, "last": 100},
            "BTC/BTC": {"bid": 1, "ask": 1, "last": 1},
            "ETH/BTC": {"bid": 0.06, "ask": 0.0601, "last": 0.06},
            "DOGE/BTC": {"bid": 0.0000016, "ask": 0.00000161, "last": 0.0000016}
        })
    
    return client
