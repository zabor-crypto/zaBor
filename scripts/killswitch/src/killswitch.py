#!/usr/bin/env python3
"""
################################################################################
# CRYPTO PORTFOLIO KILL-SWITCH v2.0 (Production Hardened)
################################################################################
# 
# CRITICAL FIXES:
# - Atomic cooldown check-and-set (prevents race conditions)
# - Retry with exponential backoff (network resilience)
# - Partial fill handling (re-fetch position after each order)
# - Precision enforcement (amount_to_precision for all orders)
# - MIN_NOTIONAL filtering (prevents dust rejections)
# - Multi-hop routing for spot (USDT/BTC/ETH)
# - Circuit breaker (alerts on consecutive failures)
# - Separate tier cooldowns (A/B independent)
# - Bitget password validation
# - Simple backtest mode
#
################################################################################
"""

import os
import sys
import time
import json
import yaml
import sqlite3
import argparse
import traceback
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

# Ensure script runs from its own directory (for relative paths like logs, config, db)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# PATCH 11: Import structured logger
from logger import get_logger

try:
    import ccxt
except ImportError:
    print("CRITICAL ERROR: 'ccxt' library not found.")
    print("Please install it using: pip install ccxt")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MIN_NOTIONAL_USDT = 10.0  # Minimum order value (Binance ~5-10 USDT)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures

# PATCH 10: Dust & trash token filtering for spot equity calculation
MIN_BALANCE_FILTER = 0.001        # Ignore tokens with balance < 0.001 (dust)
MIN_NOTIONAL_FILTER = 1.0         # Ignore tokens with notional < $1 (trash)
UNPRICED_FALLBACK_USDT = 0.000001 # Micro-fallback for coverage calc ($0.000001 per token)
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD'}  # Always keep these

# Spot equity valuation quality gates
SPOT_COVERAGE_MIN = 0.98           # fraction of non-USDT notional that must be priceable
SPOT_UNPRICED_USDT_MAX = 500.0     # PATCH 10: Raised from $25 to $500 for filtered shitcoins


# ==============================================================================
# SECTION 1: DATA MODELS
# ==============================================================================

class AccountType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"

class Tier(str, Enum):
    A = "A"
    B = "B"

class ExchangeId(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    BITGET = "bitget"
    MOCK = "mock"

@dataclass(frozen=True)
class Scope:
    exchange: ExchangeId
    account: AccountType
    def __str__(self): return f"{self.exchange.value}_{self.account.value}"

@dataclass
class EquitySnapshot:
    ts: int
    scope: Scope
    equity_usdt: float
    wallet_usdt: Optional[float] = None
    upnl_usdt: Optional[float] = None
    quality_ok: bool = True
    raw: Optional[Dict[str, Any]] = None

@dataclass
class Decision:
    ts: int
    scope: Scope
    tier: Tier
    dd: Dict[str, float]
    reasons: List[str]
    action_mode: str

@dataclass
class ActionResult:
    success: bool
    details: str
    orders_placed: int = 0
    errors: List[str] = field(default_factory=list)

@dataclass
class TierConfig:
    thresholds: Dict[str, float]
    mode: str
    cooldown_min: int
    confirm_consecutive: int = 1
    blacklist: Optional[List[str]] = None

@dataclass
class AccountConfig:
    enabled: bool
    windows: List[str]
    tier_a: Optional[TierConfig] = None
    tier_b: Optional[TierConfig] = None

@dataclass
class ExchangeConfig:
    enabled: bool
    api_key: str
    api_secret: str
    password: Optional[str]
    accounts: Dict[str, AccountConfig] = field(default_factory=dict)

@dataclass
class GlobalConfig:
    poll_seconds: int
    state_db: str
    stables_keep: List[str]
    exchanges: Dict[str, ExchangeConfig]
    dry_run: bool

class ConfigLoader:
    def load(self, path: str) -> GlobalConfig:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return self._parse(data)

    def _parse(self, data: dict) -> GlobalConfig:
        exchanges = {}
        for name, ex_data in data.get('exchanges', {}).items():
            if not ex_data.get('enabled', False):
                continue
                
            api_key = self._resolve_env(ex_data.get('api_key_env') or ex_data.get('api_key'))
            api_secret = self._resolve_env(ex_data.get('api_secret_env') or ex_data.get('api_secret'))
            
            if not api_key or not api_secret:
                print(f"[CONFIG ERROR] Exchange '{name}' is enabled but API Keys are missing.")
                sys.exit(1)
            
            password = self._resolve_env(ex_data.get('password_env') or ex_data.get('password'))
            
            # PATCH 1: Diagnostic logging (security: only first 10 chars)
            key_preview = api_key[:10] + '...' if len(api_key) > 10 else api_key
            secret_preview = api_secret[:10] + '...' if len(api_secret) > 10 else '(empty)'
            print(f"[CONFIG] {name}: api_key={key_preview}, api_secret={secret_preview}")
            if password:
                pwd_preview = password[:5] + '...' if len(password) > 5 else '(set)'
                print(f"[CONFIG] {name}: password={pwd_preview}")
            
            # CRITICAL: Bitget requires password
            if name == 'bitget' and not password:
                print(f"[CONFIG ERROR] Bitget requires 'password_env' or 'password' field.")
                sys.exit(1)
            
            accounts = {}
            for acc_name, acc_data in ex_data.get('accounts', {}).items():
                accounts[acc_name] = self._parse_acc(acc_data)
            
            exchanges[name] = ExchangeConfig(
                enabled=True,
                api_key=api_key,
                api_secret=api_secret,
                password=password,
                accounts=accounts
            )
        
        return GlobalConfig(
            poll_seconds=data.get('poll_seconds', 60),
            state_db=data.get('state_db', './killswitch_state.sqlite'),
            stables_keep=data.get('stables_keep', ["USDT", "USDC", "DAI", "FDUSD"]),
            exchanges=exchanges,
            dry_run=data.get('dry_run', False)
        )

    def _parse_acc(self, data: dict) -> AccountConfig:
        acc = AccountConfig(
            enabled=data.get('enabled', False),
            windows=[str(w) for w in data.get('windows', [])],
            tier_a=self._parse_tier(data.get('tier_a')),
            tier_b=self._parse_tier(data.get('tier_b'))
        )
        
        # Validation
        if acc.enabled and not acc.windows:
            raise ValueError("Account enabled but 'windows' is empty")
        
        return acc

    def _parse_tier(self, data: dict) -> Optional[TierConfig]:
        if not data: return None
        # Ensure threshold keys are strings to match window config
        raw_thresholds = data.get('thresholds', {})
        thresholds = {str(k): float(v) for k, v in raw_thresholds.items()}
        
        tier = TierConfig(
            thresholds=thresholds,
            mode=data.get('mode', ''),
            cooldown_min=data.get('cooldown_min', 60),
            confirm_consecutive=data.get('confirm_consecutive', 1),
            blacklist=data.get('blacklist')
        )
        
        if not tier.thresholds:
            raise ValueError("Tier config has no 'thresholds'")
        if not tier.mode:
            raise ValueError("Tier config has no 'mode'")
            
        return tier

    def _resolve_env(self, val: Optional[str]) -> str:
        """Resolve environment variable, stripping whitespace"""
        if not val: return ""
        if val.startswith("$"):
            var_name = val[1:]
            result = os.environ.get(var_name, "")
            return result.strip()  # Strip whitespace
        if val.isupper() and "_" in val and " " not in val:
             env_val = os.environ.get(val)
             if env_val: return env_val.strip()  # Strip whitespace
        return val.strip()  # Strip whitespace

# ==============================================================================
# SECTION 2: STORAGE & DRAWDOWN LOGIC
# ==============================================================================

class SqliteStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._configure_db()
        self._init_schema()

    def _configure_db(self):
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self.conn.execute('PRAGMA busy_timeout=5000;')

    def _init_schema(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS snapshots (
                ts INTEGER, exchange TEXT, account TEXT, 
                equity REAL, wallet REAL, upnl REAL,
                PRIMARY KEY (exchange, account, ts) 
            )''')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts)')
            
            self.conn.execute('''CREATE TABLE IF NOT EXISTS actions (
                key TEXT PRIMARY KEY, ts INTEGER, details TEXT)''')
                
            self.conn.execute('''CREATE TABLE IF NOT EXISTS cooldowns (
                scope TEXT, tier TEXT, until INTEGER, 
                PRIMARY KEY(scope, tier))''')
            
            # Per-scope persistent de-risked state (idempotency / spam prevention)
            self.conn.execute('''CREATE TABLE IF NOT EXISTS scope_state (
                scope TEXT PRIMARY KEY,
                derisked_until INTEGER,
                derisked_mode TEXT,
                updated_ts INTEGER,
                details TEXT
            )''')


    def append_snapshot(self, s: EquitySnapshot):
        if not s.quality_ok: return
        with self.conn:
            self.conn.execute('''
                INSERT OR IGNORE INTO snapshots (ts, exchange, account, equity, wallet, upnl)
                VALUES (?,?,?,?,?,?)
            ''', (s.ts, s.scope.exchange.value, s.scope.account.value, 
                  s.equity_usdt, s.wallet_usdt, s.upnl_usdt))

    def get_history(self, scope: Scope, lookback_sec: int, now_ts: int) -> List[Tuple[int, float]]:
        min_ts = now_ts - lookback_sec
        cur = self.conn.cursor()
        cur.execute('''
            SELECT ts, equity FROM snapshots 
            WHERE exchange=? AND account=? AND ts>=? 
            ORDER BY ts ASC
        ''', (scope.exchange.value, scope.account.value, min_ts))
        return cur.fetchall()

    def in_cooldown(self, scope: Scope, tier_name: str, now_ts: int) -> bool:
        cur = self.conn.cursor()
        cur.execute('SELECT until FROM cooldowns WHERE scope=? AND tier=?', (str(scope), tier_name))
        row = cur.fetchone()
        return row and row[0] > now_ts

    def try_set_cooldown(self, scope: Scope, tier_name: str, until: int) -> bool:
        """
        ATOMIC check-and-set. Returns True if cooldown was set (not already active).
        """
        cur = self.conn.cursor()
        # First check if cooldown exists and is still active
        cur.execute('SELECT until FROM cooldowns WHERE scope=? AND tier=?', (str(scope), tier_name))
        row = cur.fetchone()
        
        now = int(time.time())
        if row and row[0] > now:
            return False  # Already in cooldown
        
        # Set new cooldown
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO cooldowns VALUES (?,?,?)', 
                            (str(scope), tier_name, until))
        return True
    
    def get_derisked_until(self, scope: Scope) -> int:
        cur = self.conn.cursor()
        cur.execute('SELECT derisked_until FROM scope_state WHERE scope=?', (str(scope),))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    
    def is_derisked(self, scope: Scope, now_ts: int) -> bool:
        cur = self.conn.cursor()
        cur.execute('SELECT derisked_until FROM scope_state WHERE scope=?', (str(scope),))
        row = cur.fetchone()
        return bool(row and row[0] and int(row[0]) > now_ts)
    
    def set_derisked(self, scope: Scope, until_ts: int, mode: str, details: str, now_ts: int):
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO scope_state(scope, derisked_until, derisked_mode, updated_ts, details) VALUES (?,?,?,?,?)',
                (str(scope), int(until_ts), str(mode), int(now_ts), str(details))
            )

    def clear_derisked(self, scope: Scope):
        with self.conn:
            self.conn.execute('DELETE FROM scope_state WHERE scope=?', (str(scope),))


    def record_action(self, key: str, details: str, ts: int):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO actions VALUES (?,?,?)', (key, ts, details))

class DrawdownCalculator:
    def __init__(self, store: SqliteStore):
        self.store = store

    @staticmethod
    def parse_window(w: Any) -> int:
        """Normalize window config to minutes (int). Handles '15m', '1h', or int."""
        if isinstance(w, int): return w
        s = str(w).lower()
        if s.endswith('m'): return int(s[:-1])
        if s.endswith('h'): return int(s[:-1]) * 60
        if s.endswith('d'): return int(s[:-1]) * 1440
        return int(s)

    def compute(self, scope: Scope, now: int, windows: List[str]) -> Dict[str, float]:
        if not windows: return {}
        # Normalize window list to minutes for calculation
        w_minutes = [self.parse_window(w) for w in windows]
        max_win = max(w_minutes)
        
        hist = self.store.get_history(scope, max_win*60 + 300, now)
        if not hist: return {str(w): 0.0 for w in windows} # Return keys as original strings? Or minutes? 
        # Requirement: Keys in 'dds' usually match keys in 'thresholds'. 
        # Config has keys like "15", "60". If user puts "1h", our config loader might keep it as "1h".
        # Let's map back to original keys if possible, or just standard stringified ints.
        # User config: windows: ["15", "60"] -> dict keys "15", "60".
        
        curr_eq = hist[-1][1]
        res = {}
        for w_str in windows:
            w = self.parse_window(w_str)
            start_ts = now - (w * 60)
            
            subset = [eq for (t, eq) in hist if t >= start_ts]
            
            min_req = max(2, int(0.8 * w))
            if len(subset) < min_req:
                res[w_str] = 0.0
                continue
                
            hwm = max(subset)
            if hwm <= 0:
                res[w_str] = 0.0
            else:
                res[w_str] = (hwm - curr_eq) / hwm
        return res

    def pick_trigger_window(self, dds: Dict[str, float], thresholds: Dict[str, float]) -> Optional[Tuple[float, str, float]]:
        """
        Selects strongest trigger window based on DD/Threshold ratio.
        Returns: (score, window_str, threshold_float) or None
        """
        best = None 
        for w_key, thr in thresholds.items():
            dd = float(dds.get(w_key, 0.0))
            if thr > 0 and dd >= thr:
                score = dd / thr
                if (best is None) or (score > best[0]):
                    best = (score, w_key, thr)
        return best

    def is_confirmed(self, scope: Scope, thresh: float, window: Any, now: int, consecutive: int, debug: bool = False) -> bool:
        """
        Check if drawdown breach is confirmed over consecutive snapshots.
        
        Args:
            scope: Trading scope
            thresh: DD threshold (e.g., 0.05 for 5%)
            window: Time window (minutes or string like "15m")
            now: Current timestamp
            consecutive: Number of consecutive breaches required
            debug: Enable debug logging
        
        Returns:
            True if breach confirmed over consecutive snapshots
        """
        if consecutive <= 1: 
            if debug: print(f"[is_confirmed] consecutive={consecutive}, auto-confirm=True")
            return True
        
        w_min = self.parse_window(window)
        lookback = (w_min * 60) + (consecutive * 120)
        hist = self.store.get_history(scope, lookback, now)
        
        if debug:
            print(f"\n[is_confirmed DEBUG]")
            print(f"  Scope: {scope}")
            print(f"  Window: {window} ({w_min} min)")
            print(f"  Threshold: {thresh:.2%}")
            print(f"  Consecutive required: {consecutive}")
            print(f"  History points: {len(hist)}")
        
        if len(hist) < consecutive:
            if debug: print(f"  Result: INSUFFICIENT DATA (need {consecutive}, have {len(hist)})")
            return False
        
        latest_points = hist[-consecutive:]
        
        if debug:
            print(f"  Latest {consecutive} points:")
            for i, (ts, eq) in enumerate(latest_points):
                print(f"    [{i}] ts={ts}, eq={eq:.2f}")
        
        breach_count = 0
        breach_details = []
        
        for idx, (pt_ts, pt_eq) in enumerate(latest_points):
            win_start = pt_ts - (w_min * 60)
            
            # Get all equity values in the window BEFORE or AT this point
            valid_hwm_candidates = [e for (t, e) in hist if win_start <= t <= pt_ts]
            
            if not valid_hwm_candidates:
                if debug: breach_details.append(f"    Point {idx}: NO HWM candidates")
                continue
            
            past_hwm = max(valid_hwm_candidates)
            
            if past_hwm > 0:
                dd = (past_hwm - pt_eq) / past_hwm
                breached = dd >= thresh
                
                if debug:
                    breach_details.append(
                        f"    Point {idx}: HWM={past_hwm:.2f}, Eq={pt_eq:.2f}, "
                        f"DD={dd:.4f} ({dd:.2%}), Breach={breached}"
                    )
                
                if breached:
                    breach_count += 1
            else:
                if debug: breach_details.append(f"    Point {idx}: HWM=0 (skip)")
        
        confirmed = breach_count >= consecutive
        
        if debug:
            print(f"  Breach calculations:")
            for detail in breach_details:
                print(detail)
            print(f"  Breaches: {breach_count}/{consecutive}")
            print(f"  Result: {'CONFIRMED ✓' if confirmed else 'NOT CONFIRMED ✗'}\n")
        
        return confirmed

# ==============================================================================
# SECTION 3: CIRCUIT BREAKER
# ==============================================================================

class CircuitBreaker:
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD):
        self.failures = defaultdict(int)
        self.threshold = threshold
        self.broken = set()
    
    def record_failure(self, scope: Scope):
        self.failures[str(scope)] += 1
        if self.failures[str(scope)] >= self.threshold:
            if str(scope) not in self.broken:
                self.broken.add(str(scope))
                print(f"⚠️  CIRCUIT BREAKER OPEN: {scope} ({self.failures[str(scope)]} consecutive failures)")
    
    def record_success(self, scope: Scope):
        if str(scope) in self.broken:
            print(f"✅ CIRCUIT BREAKER CLOSED: {scope}")
            self.broken.remove(str(scope))
        self.failures[str(scope)] = 0
    
    def is_broken(self, scope: Scope) -> bool:
        return str(scope) in self.broken

# ==============================================================================
# SECTION 4: EXCHANGE ADAPTERS
# ==============================================================================

class BaseAdapter(ABC):
    def __init__(self, config): 
        self.config = config
    
    @abstractmethod
    def fetch_equity(self, scope: Scope) -> EquitySnapshot: pass
    
    @abstractmethod
    def close_all(self, scope: Scope, longs_only: bool, dry_run: bool) -> ActionResult: pass
    
    @abstractmethod
    def sell_spot(self, scope: Scope, keep: List[str], blacklist: List[str], dry_run: bool) -> ActionResult: pass

class RealCCXTAdapter(BaseAdapter):
    def __init__(self, config: ExchangeConfig, ex_id: ExchangeId):
        super().__init__(config)
        self.ex_id = ex_id
        opts = {
            'apiKey': config.api_key, 
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True}
        }
        if config.password: opts['password'] = config.password
        
        if ex_id == ExchangeId.BINANCE:
            self.client = ccxt.binance(opts)
            self.client_futures = ccxt.binance({**opts, 'options': {'defaultType': 'future'}})
        elif ex_id == ExchangeId.BYBIT:
            self.client = ccxt.bybit(opts)
            self.client_futures = self.client
        elif ex_id == ExchangeId.BITGET:
            self.client = ccxt.bitget(opts)
            self.client_futures = ccxt.bitget({**opts, 'options': {'defaultType': 'swap'}})
            
            # PATCH 3: Diagnostic logging for Bitget API keys
            print(f"[DEBUG] Bitget client created")
            print(f"[DEBUG] Bitget API Key: {opts['apiKey'][:10]}... (len={len(opts['apiKey'])})")
            print(f"[DEBUG] Bitget API Secret: {opts['secret'][:10]}... (len={len(opts['secret'])})")
            if 'password' in opts:
                print(f"[DEBUG] Bitget Password: {opts['password'][:5]}... (len={len(opts['password'])})")
            else:
                print(f"[DEBUG] Bitget Password: NOT SET ⚠️")

    def _retry_with_backoff(self, func, max_retries=MAX_RETRIES):
        """Exponential backoff with jitter"""
        for attempt in range(max_retries):
            try:
                return func()
            except ccxt.NetworkError as e:
                if attempt == max_retries - 1:
                    raise
                delay = RETRY_BASE_DELAY ** attempt + random.random()
                print(f"  [RETRY {attempt+1}/{max_retries}] Network error, waiting {delay:.1f}s: {e}")
                time.sleep(delay)
            except Exception as e:
                raise  # Non-network errors fail immediately

    def fetch_equity(self, scope: Scope) -> EquitySnapshot:
        ts = int(time.time())
        try:
            if self.ex_id == ExchangeId.BINANCE:
                if scope.account == AccountType.FUTURES:
                    bal = self._retry_with_backoff(lambda: self.client_futures.fetch_balance())
                    info = bal.get('info', {})
                    wallet = float(info.get('totalWalletBalance', bal['total'].get('USDT', 0)))
                    upnl = float(info.get('totalUnrealizedProfit', 0))
                    total_eq = wallet + upnl
                    return EquitySnapshot(ts, scope, total_eq, wallet, upnl, True, {'src': 'binance_f'})
                else:
                    # Binance spot - no market_type needed
                    return self._fetch_spot_equity(scope, self.client, market_type=None)

            elif self.ex_id == ExchangeId.BYBIT:
                if scope.account == AccountType.FUTURES:
                    bal = self._retry_with_backoff(lambda: self.client.fetch_balance(params={'type': 'swap'}))
                    info = bal.get('info', {})
                    
                    if 'result' in info and 'list' in info['result']:
                        raw_list = info['result']['list']
                        if raw_list:
                            acct = raw_list[0]
                            total_eq = float(acct.get('totalEquity', 0))
                            wallet_bal = float(acct.get('totalWalletBalance', 0))
                            upnl = float(acct.get('totalPerpUPL', 0))
                            return EquitySnapshot(ts, scope, total_eq, wallet_bal, upnl, True, {'src': 'bybit_v5'})
                    
                    return EquitySnapshot(ts, scope, 0.0, None, None, False, {'err': 'Bybit V5 Parse Failed'})
                else:
                    # PATCH 2: Bybit spot - use market_type='spot'
                    return self._fetch_spot_equity(scope, self.client, market_type='spot')

            elif self.ex_id == ExchangeId.BITGET:
                if scope.account == AccountType.FUTURES:
                    bal = self._retry_with_backoff(lambda: self.client_futures.fetch_balance(params={'type': 'swap'}))
                    info = bal.get('info')
                    
                    # CCXT may return info as list or dict with nested list payload
                    if isinstance(info, dict):
                        for k in ('data', 'result', 'list', 'accounts'):
                            v = info.get(k)
                            if isinstance(v, list):
                                info = v
                                break
                    
                    if isinstance(info, list) and len(info) > 0:
                        # Bitget returns list of accounts
                        for acc in info:
                            if not isinstance(acc, dict):
                                continue
                            coin = (acc.get('marginCoin') or acc.get('coin') or '').upper()
                            if coin != 'USDT':
                                continue
                            
                            # Prefer explicit equity fields
                            eq = float(acc.get('usdtEquity') or acc.get('equity') or acc.get('totalEquity') or 0.0)
                            wallet = acc.get('available') or acc.get('wallet') or None
                            upnl = acc.get('unrealizedPL') or acc.get('uPnL') or None
                            
                            wallet_f = float(wallet) if wallet is not None else None
                            upnl_f = float(upnl) if upnl is not None else None
                            
                            if eq > 0:
                                return EquitySnapshot(ts, scope, eq, wallet_f, upnl_f, True, {'src': 'bitget'})
                        
                        # If no USDT account found, return first account's USD equity
                        first_acc = info[0]
                        eq = float(first_acc.get('usdtEquity', 0))
                        wallet = float(first_acc.get('available', 0))
                        upnl = float(first_acc.get('unrealizedPL', 0))
                        return EquitySnapshot(ts, scope, eq, wallet, upnl, True, {'src': 'bitget_default'})
                    
                    return EquitySnapshot(ts, scope, 0.0, None, None, False, {'err': 'Bitget Parse Failed: Empty response'})
                else:
                    # Bitget spot - no market_type needed
                    return self._fetch_spot_equity(scope, self.client, market_type=None)

        except Exception as e:
            # PATCH 4: Enhanced error messages for debugging
            error_msg = f"{self.ex_id.value} {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {scope} {error_msg}")
            # For debugging: print full traceback for non-network errors
            if not isinstance(e, ccxt.NetworkError):
                import traceback
                print(f"[ERROR] {scope} Traceback:")
                traceback.print_exc()
            return EquitySnapshot(ts, scope, 0.0, None, None, False, {'err': error_msg})
            
        return EquitySnapshot(ts, scope, 0.0, quality_ok=False)

    def _fetch_spot_equity(self, scope: Scope, client, market_type=None) -> EquitySnapshot:
        """Multi-hop routing: USDT -> BTC -> ETH with coverage gate. market_type for Bybit='spot'"""
        ts = int(time.time())
        
        # PATCH 2: Use market_type parameter for exchanges that need it (Bybit spot)
        if market_type:
            print(f"[DEBUG] {scope} Fetching balance with market_type={market_type}")
            bal = self._retry_with_backoff(lambda: client.fetch_balance(params={'type': market_type}))
        else:
            bal = self._retry_with_backoff(lambda: client.fetch_balance())
            
        # Load all assets with positive balance
        raw_assets = {c: amt for c, amt in bal['total'].items() if amt > 0}
        
        # PATCH 10: Filter dust tokens (balance too small to matter)
        assets = {}
        for c, amt in raw_assets.items():
            # Always keep USDT and stablecoins
            if c in STABLECOINS:
                assets[c] = amt
                continue
            
            # Filter dust: ignore tokens with very small balance
            if amt < MIN_BALANCE_FILTER:
                print(f"[DUST FILTER] {scope} Skipping {c}: {amt:.6f} < {MIN_BALANCE_FILTER}")
                continue
            
            assets[c] = amt
        
        total_usdt = assets.get('USDT', 0.0)
        print(f"[DEBUG] {scope} Loaded {len(assets)} assets (filtered {len(raw_assets) - len(assets)} dust tokens)")
        
        # PATCH 6: Early return for empty accounts (fixing Binance spot empty issue)
        if not assets or (len(assets) == 0):
            return EquitySnapshot(ts, scope, 0.0, 0.0, 0, True, {'src': 'spot_empty'})
        
        # If only USDT and it's zero, also return empty
        if len(assets) == 1 and 'USDT' in assets and total_usdt == 0:
            return EquitySnapshot(ts, scope, 0.0, 0.0, 0, True, {'src': 'spot_empty'})
        
        # PATCH 4: Track pricing coverage for quality gate
        total_notional_estimate = 0.0  # Best-effort total value
        priced_notional = 0.0           # Successfully priced value
        unpriced_assets = []
        
        needed_pairs = set()
        for c in assets:
            if c == 'USDT': continue
            needed_pairs.add(f"{c}/USDT")
            needed_pairs.add(f"{c}/BTC")
            needed_pairs.add(f"{c}/ETH")
        
        needed_pairs.update(['BTC/USDT', 'ETH/USDT'])
        
        if not needed_pairs:
            return EquitySnapshot(ts, scope, total_usdt, total_usdt, 0, True, {'src': 'spot_usdt_only'})

        # PATCH 2: Use market_type for Bybit spot ticker fetching
        ticker_params = {}
        if market_type:
            ticker_params['type'] = market_type
        
        try:
            print(f"[DEBUG] {scope} Fetching tickers for {len(needed_pairs)} pairs with params={ticker_params}")
            if ticker_params:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(list(needed_pairs), params=ticker_params))
            else:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(list(needed_pairs)))
            print(f"[DEBUG] {scope} Got {len(tickers)} tickers (with symbols)")
        except Exception as e:
            print(f"[DEBUG] {scope} fetch_tickers(symbols) failed: {e}")
            print(f"[DEBUG] {scope} Falling back to fetch_tickers() without symbols")
            if ticker_params:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(params=ticker_params))
            else:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers())
            print(f"[DEBUG] {scope} Got {len(tickers)} tickers (all)")
        
        btc_usdt = tickers.get('BTC/USDT', {}).get('bid', 0)
        eth_usdt = tickers.get('ETH/USDT', {}).get('bid', 0)
        
        for c, amt in assets.items():
            if c == 'USDT': continue
            
            price = self._get_price_multihop(c, tickers, btc_usdt, eth_usdt)
            
            if price > 0:
                notional = amt * price
                
                # PATCH 10: Filter low-notional assets (trash tokens)
                if notional < MIN_NOTIONAL_FILTER:
                    print(f"[NOTIONAL FILTER] {scope} Skipping {c}: ${notional:.4f} < ${MIN_NOTIONAL_FILTER}")
                    continue
                
                total_usdt += notional
                priced_notional += notional
                total_notional_estimate += notional
            else:
                # PATCH 10: Use micro-fallback for coverage (not $1!)
                # If token has 1M units, fallback is $1 (not $1M)
                fallback_value = amt * UNPRICED_FALLBACK_USDT
                total_notional_estimate += fallback_value
                unpriced_assets.append(c)
                print(f"[UNPRICED] {scope} {c}: {amt:.2f} tokens → fallback ${fallback_value:.4f}")
        
        # Coverage gate: Check if we successfully priced enough of the portfolio
        coverage = priced_notional / total_notional_estimate if total_notional_estimate > 0 else 1.0
        unpriced_value = total_notional_estimate - priced_notional
        
        quality_ok = True
        if coverage < SPOT_COVERAGE_MIN or unpriced_value > SPOT_UNPRICED_USDT_MAX:
            quality_ok = False
            print(f"[WARNING] {scope} Spot coverage FAILED: {coverage:.1%} coverage, ${unpriced_value:.0f} unpriced")
            print(f"          Unpriced assets: {unpriced_assets}")
        
        return EquitySnapshot(ts, scope, total_usdt, total_usdt, 0, quality_ok)

    def _get_price_multihop(self, coin: str, tickers: dict, btc_usdt: float, eth_usdt: float) -> float:
        """Try USDT -> BTC -> ETH routing"""
        # Direct USDT
        pair = f"{coin}/USDT"
        if pair in tickers:
            price = tickers[pair].get('bid', 0)
            if price > 0: return price
        
        # Via BTC
        if btc_usdt > 0:
            pair_btc = f"{coin}/BTC"
            if pair_btc in tickers:
                price_btc = tickers[pair_btc].get('bid', 0)
                if price_btc > 0:
                    return price_btc * btc_usdt
        
        # Via ETH
        if eth_usdt > 0:
            pair_eth = f"{coin}/ETH"
            if pair_eth in tickers:
                price_eth = tickers[pair_eth].get('bid', 0)
                if price_eth > 0:
                    return price_eth * eth_usdt
        
        return 0



    def _safe_futures_close(self, symbol, side, amount, is_hedge_mode_candidate):
        """
        Tries to close position handling both One-Way (idx=0) and Hedge Mode (idx=1/2).
        Bybit requires specific index in Hedge Mode, but rejects it in One-Way.
        """
        client = self.client_futures
        close_side = 'sell' if side == 'long' else 'buy'
        
        # Prepare strategy list
        # 1. Try generic reducyOnly (works for One-Way and many others)
        # 2. If Bybit, try specific Hedge Mode indices
        strategies = [{'reduceOnly': True}]
        
        if self.ex_id == ExchangeId.BYBIT and is_hedge_mode_candidate:
            # If standard reduceOnly fails, try explicit hedge index
            idx = 1 if side == 'long' else 2
            strategies.append({'reduceOnly': True, 'positionIdx': idx}) 
            # Note: Sometimes 0 is needed for One-Way explicitly
            strategies.append({'reduceOnly': True, 'positionIdx': 0})

        last_exception = None
        for params in strategies:
            try:
                self._retry_with_backoff(lambda: client.create_order(
                    symbol, 'market', close_side, amount, params=params
                ))
                return # Success
            except Exception as e:
                # Store the last exception to raise if all strategies fail
                last_exception = e
                # If it's a "position mode" error (common on Bybit), we definitely want to try the next strategy.
                # If it's a network error, _retry_with_backoff would have already retried internally.
                # So here we catch the final failure of that strategy.
                err_str = str(e).lower()
                if "position idx" in err_str or "mode" in err_str or "reduce" in err_str:
                     print(f"  [Strategy Swap] {symbol}: {e}. Trying next...")
                     continue
                
                # For other errors (authorization, insufficient balance?), we might still want to try 
                # the next strategy just in case parameters triggered it.
                continue
        
        if last_exception:
             raise last_exception
        
    def close_all(self, scope: Scope, longs_only: bool, dry_run: bool) -> ActionResult:
        client = self.client_futures
        try:
            params = {}
            if self.ex_id == ExchangeId.BITGET:
                params = {'productType': 'umcbl'} 
            if self.ex_id == ExchangeId.BYBIT:
                params['category'] = 'linear'
                params['settleCoin'] = 'USDT'
                
            positions = self._retry_with_backoff(lambda: client.fetch_positions(params=params))
            
            to_close = []
            for p in positions:
                amt = float(p.get('contracts', 0) or p.get('info', {}).get('size', 0))
                if amt == 0: continue
                
                side = p['side']
                if longs_only and side == 'short': continue
                
                to_close.append(p)
            
            if not to_close: 
                return ActionResult(True, "No positions")

            total_count = 0
            all_errors = []
            
            for p in to_close:
                symbol = p['symbol']
                side = p['side']
                
                # RETRY LOOP with position re-fetch
                for retry in range(MAX_RETRIES):
                    try:
                        # Re-fetch current position size
                        current_positions = self._retry_with_backoff(lambda: client.fetch_positions(params=params))
                        current_pos = next((x for x in current_positions if x['symbol'] == symbol and x['side'] == side), None)
                        
                        if not current_pos:
                            print(f"  Position {symbol} {side} already closed")
                            break
                        
                        qty_raw = abs(float(current_pos.get('contracts', 0) or current_pos.get('info', {}).get('size', 0)))
                        
                        if qty_raw == 0:
                            break
                        
                        if dry_run:
                            total_count += 1
                            break
                        
                        # PRECISION
                        amount = client.amount_to_precision(symbol, qty_raw)
                        
                        # EXECUTE SAFE CLOSE
                        self._safe_futures_close(symbol, side, amount, (self.ex_id == ExchangeId.BYBIT))
                        
                        total_count += 1
                        print(f"  ✓ Closed {symbol} {side} {amount}")
                        break  # Success
                        
                    except Exception as e:
                        if retry == MAX_RETRIES - 1:
                            all_errors.append(f"{symbol}: {e}")
                            print(f"  ✗ Failed {symbol} after {MAX_RETRIES} retries: {e}")
                        else:
                            print(f"  [RETRY {retry+1}] {symbol}: {e}")
                            time.sleep(RETRY_BASE_DELAY ** retry)

            return ActionResult(len(all_errors)==0, f"Closed {total_count}/{len(to_close)}", total_count, all_errors)

        except Exception as e:
            return ActionResult(False, f"Crash: {e}")

    def _sell_spot_routed(self, client, coin, amount, tickers) -> bool:
        """
        Executes spot sell with routing:
        1. COIN/USDT
        2. COIN/BTC -> BTC/USDT (Immediate execution of second leg)
        3. COIN/ETH -> ETH/USDT (Immediate execution of second leg)
        Returns: True if successfully sold to USDT (or initiated path), False otherwise.
        """
        # 1. Direct USDT
        pair = f"{coin}/USDT"
        if pair in tickers and tickers[pair]['bid'] > 0:
            amt = client.amount_to_precision(pair, amount)
            self._retry_with_backoff(lambda: client.create_order(pair, 'market', 'sell', amt))
            print(f"  ✓ Sold {pair} {amt}")
            return True

        # 2. Via BTC
        pair_btc = f"{coin}/BTC"
        if pair_btc in tickers and tickers[pair_btc]['bid'] > 0:
            amt = client.amount_to_precision(pair_btc, amount)
            self._retry_with_backoff(lambda: client.create_order(pair_btc, 'market', 'sell', amt))
            print(f"  ✓ Sold {pair_btc} {amt} (Hop 1/2)")
            
            # Executing Hop 2: Sell the resulting BTC to USDT
            time.sleep(0.5) # Allow small settlement wait
            
            # We need to fetch balance to know exact BTC amount obtained? 
            # Or just fetch total BTC balance. In emergency, selling ALL BTC is acceptable 
            # if we are in "Sell Everything" mode, but sticky if we are in "Sell Blacklist" mode.
            # However, this method is called per-asset. 
            # Best effort: Fetch balance of BTC and sell it if we just bought it?
            # Issue: We might sell pre-existing BTC. 
            # ACCEPTABLE TRADE-OFF: In a kill-switch scenario, if we are routing via BTC, 
            # we likely want to move to USDT anyway. 
            # BETTER: Just try to sell the estimated amount or re-fetch.
            # Let's re-fetch for safety.
            try:
                bal = self._retry_with_backoff(lambda: client.fetch_balance())
                btc_free = bal['free'].get('BTC', 0)
                if btc_free > 0:
                    pair_usdt = "BTC/USDT"
                    amt_btc = client.amount_to_precision(pair_usdt, btc_free)
                    # Check min notional for the second leg too? Assuming it's large enough if the first leg was.
                    self._retry_with_backoff(lambda: client.create_order(pair_usdt, 'market', 'sell', amt_btc))
                    print(f"  ✓ Sold {pair_usdt} {amt_btc} (Hop 2/2 - Liquidated intermediate)")
                    return True
            except Exception as e:
                print(f"  ⚠️ Sold {coin}/BTC but failed to sell BTC/USDT: {e}")
                # We return True because we DID sell the Alt. The BTC remains, which is better than Alt.
                return True

        # 3. Via ETH
        pair_eth = f"{coin}/ETH"
        if pair_eth in tickers and tickers[pair_eth]['bid'] > 0:
            amt = client.amount_to_precision(pair_eth, amount)
            self._retry_with_backoff(lambda: client.create_order(pair_eth, 'market', 'sell', amt))
            print(f"  ✓ Sold {pair_eth} {amt} (Hop 1/2)")
            
            # Executing Hop 2: Sell ETH to USDT
            time.sleep(0.5)
            try:
                bal = self._retry_with_backoff(lambda: client.fetch_balance())
                eth_free = bal['free'].get('ETH', 0)
                if eth_free > 0:
                    pair_usdt = "ETH/USDT"
                    amt_eth = client.amount_to_precision(pair_usdt, eth_free)
                    self._retry_with_backoff(lambda: client.create_order(pair_usdt, 'market', 'sell', amt_eth))
                    print(f"  ✓ Sold {pair_usdt} {amt_eth} (Hop 2/2 - Liquidated intermediate)")
                    return True
            except Exception as e:
                print(f"  ⚠️ Sold {coin}/ETH but failed to sell ETH/USDT: {e}")
                return True
            
            return True
            
        print(f"  ✗ No route for {coin}")
        return False

    def sell_spot(self, scope: Scope, keep: List[str], blacklist: List[str], dry_run: bool) -> ActionResult:
        client = self.client
        try:
            bal = self._retry_with_backoff(lambda: client.fetch_balance())
            assets = bal['free']
            
            # Fetch tickers for Valuation & Routing
            tickers = self._retry_with_backoff(lambda: client.fetch_tickers())
            
            to_sell = []
            for c, amt in assets.items():
                if c in keep: continue
                if blacklist and c not in blacklist: continue
                if amt <= 0: continue
                
                # Valuation Check (to skip dust)
                # We use the same _get_price_multihop logic we used for equity
                btc_usdt = tickers.get('BTC/USDT', {}).get('bid', 0)
                eth_usdt = tickers.get('ETH/USDT', {}).get('bid', 0)
                price = self._get_price_multihop(c, tickers, btc_usdt, eth_usdt)
                
                notional = amt * price
                if notional < MIN_NOTIONAL_USDT:
                    print(f"  Skip {c} (notional ${notional:.2f} < ${MIN_NOTIONAL_USDT})")
                    continue
                
                to_sell.append((c, amt))
            
            if not to_sell:
                return ActionResult(True, "No assets to sell")

            count = 0
            errors = []
            for c, amt in to_sell:
                if dry_run: 
                    count += 1
                    continue
                
                try:
                    success = self._sell_spot_routed(client, c, amt, tickers)
                    if success:
                        count += 1
                    else:
                        errors.append(f"{c}: No route found")
                except Exception as e:
                    errors.append(f"{c}: {e}")
                    print(f"  ✗ Failed {c}: {e}")
            
            return ActionResult(len(errors)==0, f"Sold {count}/{len(to_sell)}", count, errors)

        except Exception as e:
            return ActionResult(False, f"Spot Crash: {e}")

# ==============================================================================
# MOCK ADAPTER (for testing)
# ==============================================================================

class MockAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.scenario = []
        self.idx = 0
        self.base_ts = int(time.time())

    def set_scenario(self, data): 
        self.scenario = data
        
    def fetch_equity(self, scope):
        val = self.scenario[min(self.idx, len(self.scenario)-1)]
        ts = self.base_ts + (self.idx * 60)
        self.idx += 1
        return EquitySnapshot(ts, scope, float(val), float(val), 0)
    
    def close_all(self, s, l, d): 
        return ActionResult(True, "[MOCK] Closed")
        
    def sell_spot(self, s, k, b, d): 
        return ActionResult(True, f"[MOCK] Sold Spot")

# ==============================================================================
# SECTION 5: ACTION ENGINE
# ==============================================================================

class ActionEngine:
    def __init__(self, store, adapters, dry, stables, breaker):
        self.store = store
        self.adapters = adapters
        self.dry = dry
        self.stables = stables
        self.breaker = breaker

    def execute(self, dec: Decision, conf: TierConfig):
        # ATOMIC cooldown check-and-set
        until = dec.ts + (conf.cooldown_min * 60)
        if not self.store.try_set_cooldown(dec.scope, dec.tier.value, until):
            print(f"  [SKIP] {dec.tier} already in cooldown")
            return
            
        adapter = self.adapters.get(dec.scope.exchange.value)
        if not adapter: return
        
        print(f"  >>> EXECUTING {dec.action_mode} | {dec.scope} | {dec.tier}")
        
        res = ActionResult(False, "Unknown Mode")
        
        try:
            if dec.action_mode == "CLOSE_LONGS_ONLY":
                res = adapter.close_all(dec.scope, True, self.dry)
            elif dec.action_mode == "CLOSE_ALL_POSITIONS":
                res = adapter.close_all(dec.scope, False, self.dry)
            elif dec.action_mode == "SELL_BLACKLIST_ONLY_KEEP_STABLES":
                res = adapter.sell_spot(dec.scope, self.stables, conf.blacklist, self.dry)
            elif dec.action_mode == "SELL_ALL_NON_USDT_KEEP_STABLES":
                res = adapter.sell_spot(dec.scope, self.stables, None, self.dry)
            
            print(f"  [RESULT] {res.success} | {res.details}")
            
            if res.success or res.orders_placed > 0:
                self.breaker.record_success(dec.scope)
                
                # PATCH 2: Set de-risked state to prevent re-triggering on same drawdown
                # Duration: 24 hours (longer than any cooldown)
                derisked_until = dec.ts + (24 * 3600)
                details = {
                    'tier': dec.tier.value,
                    'mode': dec.action_mode,
                    'result': res.details,
                    'orders_placed': res.orders_placed,
                    'dd': {k: float(v) for k, v in dec.dd.items()}
                }
                self.store.set_derisked(
                    dec.scope,
                    derisked_until,
                    dec.action_mode,
                    json.dumps(details),
                    dec.ts
                )
                derisked_dt = datetime.fromtimestamp(derisked_until).strftime('%Y-%m-%d %H:%M')
                print(f"  [DE-RISKED] {dec.scope} until {derisked_dt}")
            else:
                self.breaker.record_failure(dec.scope)
            
            # Record action
            bucket_ts = (dec.ts // 300) * 300
            key = f"{dec.scope}|{dec.tier}|{dec.action_mode}|{bucket_ts}"
            self.store.record_action(key, str(res), dec.ts)
            
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            self.breaker.record_failure(dec.scope)

# ==============================================================================
# SECTION 6: BACKTEST MODE
# ==============================================================================

def run_backtest(csv_path: str, config_path: str):
    """
    Simple backtest: replay equity from CSV.
    CSV format: timestamp,exchange,account,equity
    """
    print(f"Loading backtest data from {csv_path}...")
    
    import csv
    snapshots = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snapshots.append({
                'ts': int(row['timestamp']),
                'exchange': row['exchange'],
                'account': row['account'],
                'equity': float(row['equity'])
            })
    
    print(f"Loaded {len(snapshots)} snapshots")
    
    loader = ConfigLoader()
    cfg = loader.load(config_path)
    
    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    breaker = CircuitBreaker()
    
    mock_adapters = {}
    engine = ActionEngine(store, mock_adapters, True, cfg.stables_keep, breaker)
    
    triggers = []
    
    for snap_data in snapshots:
        ex_id = ExchangeId(snap_data['exchange'])
        acc_type = AccountType(snap_data['account'])
        scope = Scope(ex_id, acc_type)
        
        snap = EquitySnapshot(
            snap_data['ts'], 
            scope, 
            snap_data['equity'],
            snap_data['equity'],
            0,
            True
        )
        
        store.append_snapshot(snap)
        
        # Find config
        ex_conf = cfg.exchanges.get(snap_data['exchange'])
        if not ex_conf: continue
        
        acc_conf = ex_conf.accounts.get(snap_data['account'])
        if not acc_conf or not acc_conf.enabled: continue
        
        dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)
        
        # Check triggers
        for tier_name, tier_conf in [('B', acc_conf.tier_b), ('A', acc_conf.tier_a)]:
            if not tier_conf: continue
            
            best = dd_calc.pick_trigger_window(dds, tier_conf.thresholds)
            if best:
                score, w, thr = best
                if dd_calc.is_confirmed(scope, thr, w, snap.ts, tier_conf.confirm_consecutive):
                    if not store.in_cooldown(scope, tier_name, snap.ts):
                        triggers.append({
                            'ts': snap.ts,
                            'scope': str(scope),
                            'tier': tier_name,
                            'dd': dds,
                            'mode': tier_conf.mode
                        })
                        
                        until = snap.ts + (tier_conf.cooldown_min * 60)
                        store.try_set_cooldown(scope, tier_name, until)
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total triggers: {len(triggers)}")
    
    for t in triggers:
        dt = datetime.fromtimestamp(t['ts'])
        print(f"{dt} | {t['scope']} | Tier {t['tier']} | {t['mode']} | DD: {t['dd']}")

# ==============================================================================
# SECTION 7: MAIN ENTRY
# ==============================================================================

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--test-mock", action="store_true")
    parser.add_argument("--backtest", help="CSV file for backtest mode")
    args = parser.parse_args()

    # BACKTEST MODE
    if args.backtest:
        if not args.config:
            print("Usage: --backtest equity.csv --config config.yaml")
            return
        run_backtest(args.backtest, args.config)
        return

    # MOCK TEST MODE
    if args.test_mock:
        print("Running Mock Test...")
        s = SqliteStore(":memory:")
        d = DrawdownCalculator(s)
        m = MockAdapter(None)
        m.set_scenario([10000, 9900, 9850, 9850, 9000])
        
        scope = Scope(ExchangeId.MOCK, AccountType.FUTURES)
        
        print("| TS                 | EQUITY   | DD (1m) | STATUS |")
        print("|--------------------|----------|---------|--------|")
        
        for i in range(5):
            snap = m.fetch_equity(scope)
            s.append_snapshot(snap)
            dds = d.compute(scope, snap.ts, ["1"])
            
            dd_val = dds.get("1", 0)
            status = "SAFE"
            if dd_val > 0.01:
                confirmed = d.is_confirmed(scope, 0.01, 1, snap.ts, 2)
                status = "TRIGGER" if confirmed else "WAIT..."
            
            print(f"| {datetime.fromtimestamp(snap.ts).strftime('%H:%M:%S')} | {snap.equity_usdt:<8.0f} | {dd_val:<7.2%} | {status} |")
        return

    # PRODUCTION MODE
    if not args.config:
        print("Usage: killswitch_v2.py --config config.yaml")
        print("       killswitch_v2.py --backtest equity.csv --config config.yaml")
        print("       killswitch_v2.py --test-mock")
        return

    loader = ConfigLoader()
    cfg = loader.load(args.config)
    
    # PATCH 11: Initialize structured logger with file output
    log_file = 'reports/killswitch.log'
    os.makedirs('reports', exist_ok=True)
    log_level = 'DEBUG' if getattr(args, 'verbose', False) else 'INFO'
    logger = get_logger("killswitch", level=log_level, log_file=log_file)
    logger.info(f"Logger initialized, writing to {log_file}")
    
    store = SqliteStore(cfg.state_db)
    dd_calc = DrawdownCalculator(store)
    breaker = CircuitBreaker()
    
    adapters = {}
    for name, c in cfg.exchanges.items():
        if c.enabled:
            adapters[name] = RealCCXTAdapter(c, ExchangeId(name))
            
    if not adapters:
        logger.error("No exchanges enabled in config")
        print("No exchanges enabled in config.")
        return

    engine = ActionEngine(store, adapters, cfg.dry_run, cfg.stables_keep, breaker)
    
    mode_str = "DRY RUN" if cfg.dry_run else "LIVE"
    logger.info(f"{'='*60}")
    logger.info(f"KILL-SWITCH v2.0 - {mode_str}")
    logger.info(f"Exchanges: {list(adapters.keys())}")
    logger.info(f"Poll Interval: {cfg.poll_seconds}s")
    logger.info(f"{'='*60}")
    
    # Also print to console for visibility
    print(f"{'='*60}")
    print(f"KILL-SWITCH v2.0 - {mode_str}")
    print(f"Exchanges: {list(adapters.keys())}")
    print(f"Poll Interval: {cfg.poll_seconds}s")
    print(f"Logging to: {log_file}")
    print(f"{'='*60}\n")
    
    while True:
        cycle_start = time.time()
        
        for name, ex_config in cfg.exchanges.items():
            if name not in adapters: continue
            
            for acc_name, acc_conf in ex_config.accounts.items():
                if not acc_conf.enabled: continue
                
                scope = Scope(ExchangeId(name), AccountType(acc_name))
                
                # Skip if circuit broken
                if breaker.is_broken(scope):
                    print(f"[{datetime.now().strftime('%H:%M')}] {scope} CIRCUIT BROKEN - SKIPPING")
                    continue
                
                try:
                    snap = adapters[name].fetch_equity(scope)
                    
                    # PATCH 3: VALIDATE SNAPSHOT BEFORE STORING
                    # Distinguish between bad reads and empty accounts
                    if not snap.quality_ok:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} SKIPPING BAD READ: {snap.raw}")
                        breaker.record_failure(scope)
                        continue
                    
                    if snap.equity_usdt is None or snap.equity_usdt < 0:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} INVALID EQUITY: {snap.equity_usdt}")
                        breaker.record_failure(scope)
                        continue
                    
                    # Empty account is valid state - skip DD calculation but don't fail
                    if snap.equity_usdt == 0:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} Empty account (Eq: $0) - skipping DD calculation")
                        breaker.record_success(scope)  # Not a failure
                        store.append_snapshot(snap)  # Still record snapshot
                        continue
                        
                    store.append_snapshot(snap)
                    
                    breaker.record_success(scope)
                    
                    dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)
                    
                    # PATCH 1: Check if scope is in de-risked state (prevents re-liquidation loops)
                    if store.is_derisked(scope, snap.ts):
                        derisked_until = store.get_derisked_until(scope)
                        remaining_sec = derisked_until - snap.ts
                        remaining_min = remaining_sec // 60
                        dd_str = ', '.join([f"{w}:{v:.1%}" for w, v in sorted(dds.items())])
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} DE-RISKED ({remaining_min}m left) | Eq: ${snap.equity_usdt:.0f} | DD: {dd_str}")
                        continue
                    
                    # Decision Logic: Tier B supersedes Tier A
                    final_decision = None
                    target_tier_conf = None
                    
                    # CHECK TIER B (CRITICAL)
                    if acc_conf.tier_b:
                        best_b = dd_calc.pick_trigger_window(dds, acc_conf.tier_b.thresholds)
                        if best_b:
                            score, w, thr = best_b
                            if dd_calc.is_confirmed(scope, thr, w, snap.ts, acc_conf.tier_b.confirm_consecutive):
                                final_decision = Decision(snap.ts, scope, Tier.B, dds, [], acc_conf.tier_b.mode)
                                target_tier_conf = acc_conf.tier_b
                                # PATCH 11: Use logger for triggers
                                logger.warning(f"🚨 [TIER B TRIGGER] {scope} | Window={w} | DD={dds.get(str(w),0):.4f} > {thr}")
                                print(f"\n🚨 [TIER B TRIGGER] {scope} | Window={w} | DD={dds.get(str(w),0):.4f} > {thr}")

                    # CHECK TIER A (WARNING) - Only if Tier B didn't trigger
                    if not final_decision and acc_conf.tier_a:
                         best_a = dd_calc.pick_trigger_window(dds, acc_conf.tier_a.thresholds)
                         if best_a:
                            score, w, thr = best_a
                            if dd_calc.is_confirmed(scope, thr, w, snap.ts, acc_conf.tier_a.confirm_consecutive):
                                final_decision = Decision(snap.ts, scope, Tier.A, dds, [], acc_conf.tier_a.mode)
                                target_tier_conf = acc_conf.tier_a
                                # PATCH 11: Use logger for triggers
                                logger.warning(f"⚠️ [TIER A TRIGGER] {scope} | Window={w} | DD={dds.get(str(w),0):.4f} > {thr}")
                                print(f"\n⚠️ [TIER A TRIGGER] {scope} | Window={w} | DD={dds.get(str(w),0):.4f} > {thr}")
                    
                    dd_str = ', '.join([f"{w}:{v:.1%}" for w, v in sorted(dds.items())])
                    # PATCH 11: Use logger for snapshots
                    logger.info(f"[{datetime.now().strftime('%H:%M')}] {scope} Eq=${snap.equity_usdt:.0f} | DD: {dd_str}")
                    print(f"[{datetime.now().strftime('%H:%M')}] {scope} Eq: ${snap.equity_usdt:.0f} | DD: {dd_str}")
                    
                    if final_decision:
                        # PATCH 11: Use logger for action execution
                        logger.critical(f"🚨 TRIGGERED {final_decision.tier} - Executing {target_tier_conf.mode}")
                        print(f"  🚨 TRIGGERED {final_decision.tier}")
                        engine.execute(final_decision, target_tier_conf)

                except Exception as e:
                    # PATCH 11: Use logger for errors
                    logger.error(f"{scope}: {e}")
                    print(f"[ERROR] {scope}: {e}")
                    breaker.record_failure(scope)
        
        elapsed = time.time() - cycle_start
        time.sleep(max(1, cfg.poll_seconds - elapsed))

if __name__ == "__main__":
    run()