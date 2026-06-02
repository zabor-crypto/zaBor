#!/usr/bin/env python3
"""
################################################################################
# CRYPTO PORTFOLIO KILL-SWITCH v3.0 — Stateful Risk-Attribution Engine
################################################################################
#
# Architecture:
#   equity drawdown → position snapshot → PnL attribution → risk-ranked
#   liquidation plan → staged escalation → audited execution
#
# Key improvements over v2.0:
#   - Stage machine (0-4) replaces binary de-risked flag; higher stages always
#     override lower-stage cooldowns — escalation is never suppressed.
#   - CLOSE_TOP_RISK_CONTRIBUTORS (Stage 1): closes only the positions that
#     caused the drawdown, based on PnL delta attribution.
#   - CLOSE_DOMINANT_LOSS_DIRECTION (Stage 2): closes the losing direction only.
#   - CLOSE_ALL_POSITIONS (Stage 3): full futures kill with order cancellation
#     and flat-state verification.
#   - Spot multi-hop routing sells only the intermediate BTC/ETH delta, never
#     pre-existing holdings.
#   - Non-reduce-only entry orders are cancelled before every futures close.
#   - File-based trading lock for external bot coordination.
#   - Backward compatible with tier_a / tier_b config (maps to stage_1 / stage_3).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from logger import get_logger

try:
    import ccxt
except ImportError:
    print("CRITICAL ERROR: 'ccxt' library not found. Run: pip install ccxt")
    sys.exit(1)

from risk_attribution import (
    PositionSnapshot,
    AttributionResult,
    compute_pnl_attribution,
    rank_positions_by_risk,
)
from stage_machine import StageMachine, StageState, STAGE_NAMES, STAGE_ALL_FUTURES_CLOSED
from order_safety import CloseInstruction
from trading_lock import TradingLock
from position_store import PositionStore

# ==============================================================================
# CONSTANTS
# ==============================================================================

MIN_NOTIONAL_USDT = 10.0
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
CIRCUIT_BREAKER_THRESHOLD = 5

MIN_BALANCE_FILTER = 0.001
MIN_NOTIONAL_FILTER = 1.0
UNPRICED_FALLBACK_USDT = 0.000001
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD'}

SPOT_COVERAGE_MIN = 0.98
SPOT_UNPRICED_USDT_MAX = 500.0

# Attribution lookback for position reference snapshots (seconds)
ATTRIBUTION_LOOKBACK_SEC = 3600

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
    """Legacy tier-based decision (kept for backward compat)."""
    ts: int
    scope: Scope
    tier: Tier
    dd: Dict[str, float]
    reasons: List[str]
    action_mode: str

@dataclass
class StageDecision:
    """New stage-based decision."""
    ts: int
    scope: Scope
    stage: int
    dd: Dict[str, float]
    action_mode: str

@dataclass
class ActionResult:
    success: bool
    details: str
    orders_placed: int = 0
    errors: List[str] = field(default_factory=list)

@dataclass
class StageConfig:
    """Configuration for one stage (or legacy tier). Superset of old TierConfig."""
    thresholds: Dict[str, float]
    mode: str
    cooldown_min: int
    confirm_consecutive: int = 1
    blacklist: Optional[List[str]] = None
    # Attribution / surgical close fields
    source_mode: str = "AUTO"         # AUTO | LONG | SHORT | MIXED
    source_threshold: float = 0.65
    top_n: int = 3
    close_fraction: float = 0.50
    full_close_if_liq_distance_below_pct: float = 0.03
    cancel_entry_orders_before_close: bool = True
    set_trading_lock: bool = True

# Backward-compat alias
TierConfig = StageConfig

@dataclass
class AccountConfig:
    enabled: bool
    windows: List[str]
    # New stage-based config (preferred)
    stage_1: Optional[StageConfig] = None
    stage_2: Optional[StageConfig] = None
    stage_3: Optional[StageConfig] = None
    # Legacy tier config (kept for external code that reads it)
    tier_a: Optional[StageConfig] = None
    tier_b: Optional[StageConfig] = None
    account_mode: str = "standard"

    def get_stage(self, n: int) -> Optional[StageConfig]:
        return {1: self.stage_1, 2: self.stage_2, 3: self.stage_3}.get(n)

    @property
    def has_stages(self) -> bool:
        return bool(self.stage_1 or self.stage_2 or self.stage_3)

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
    trading_lock_file: str = "./killswitch_trading_lock.json"
    spot_routing_delta_only: bool = True
    spot_routing_allow_preexisting: bool = False

# ==============================================================================
# SECTION 2: CONFIG LOADER
# ==============================================================================

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

            key_preview = api_key[:10] + '...' if len(api_key) > 10 else api_key
            secret_preview = api_secret[:10] + '...' if len(api_secret) > 10 else '(empty)'
            print(f"[CONFIG] {name}: api_key={key_preview}, api_secret={secret_preview}")
            if password:
                pwd_preview = password[:5] + '...' if len(password) > 5 else '(set)'
                print(f"[CONFIG] {name}: password={pwd_preview}")

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

        spot_routing = data.get('spot_routing', {})
        return GlobalConfig(
            poll_seconds=data.get('poll_seconds', 60),
            state_db=data.get('state_db', './killswitch_state.sqlite'),
            stables_keep=data.get('stables_keep', ["USDT", "USDC", "DAI", "FDUSD"]),
            exchanges=exchanges,
            dry_run=data.get('dry_run', False),
            trading_lock_file=data.get('trading_lock_file', './killswitch_trading_lock.json'),
            spot_routing_delta_only=spot_routing.get('sell_intermediate_delta_only', True),
            spot_routing_allow_preexisting=spot_routing.get('allow_liquidate_preexisting_intermediate', False),
        )

    def _parse_acc(self, data: dict) -> AccountConfig:
        windows = [str(w) for w in data.get('windows', [])]
        windows = self._deduplicate_windows(windows)

        stage_1 = self._parse_stage(data.get('stage_1'))
        stage_2 = self._parse_stage(data.get('stage_2'))
        stage_3 = self._parse_stage(data.get('stage_3'))

        tier_a = self._parse_stage(data.get('tier_a'))
        tier_b = self._parse_stage(data.get('tier_b'))

        # Backward compat: map tier_a -> stage_1, tier_b -> stage_3
        if tier_a and not stage_1:
            if tier_a.mode == 'CLOSE_LONGS_ONLY':
                print(
                    f"[CONFIG WARNING] tier_a mode CLOSE_LONGS_ONLY is unsafe: it may close "
                    f"profitable longs while losses are from shorts. Migrating to stage_1. "
                    f"Update your config to use stage_1 with mode: CLOSE_TOP_RISK_CONTRIBUTORS."
                )
            stage_1 = tier_a

        if tier_b and not stage_3:
            stage_3 = tier_b

        acc = AccountConfig(
            enabled=data.get('enabled', False),
            windows=windows,
            stage_1=stage_1,
            stage_2=stage_2,
            stage_3=stage_3,
            tier_a=tier_a,
            tier_b=tier_b,
            account_mode=data.get('account_mode', 'standard'),
        )

        if acc.enabled and not acc.windows:
            raise ValueError("Account enabled but 'windows' is empty")

        return acc

    def _parse_stage(self, data: Optional[dict]) -> Optional[StageConfig]:
        if not data:
            return None
        raw_thresholds = data.get('thresholds', {})
        thresholds = {str(k): float(v) for k, v in raw_thresholds.items()}
        if not thresholds:
            raise ValueError("Stage/tier config has no 'thresholds'")
        mode = data.get('mode', '')
        if not mode:
            raise ValueError("Stage/tier config has no 'mode'")
        return StageConfig(
            thresholds=thresholds,
            mode=mode,
            cooldown_min=data.get('cooldown_min', 60),
            confirm_consecutive=data.get('confirm_consecutive', 1),
            blacklist=data.get('blacklist'),
            source_mode=data.get('source_mode', 'AUTO'),
            source_threshold=float(data.get('source_threshold', 0.65)),
            top_n=int(data.get('top_n', 3)),
            close_fraction=float(data.get('close_fraction', 0.50)),
            full_close_if_liq_distance_below_pct=float(
                data.get('full_close_if_liq_distance_below_pct', 0.03)
            ),
            cancel_entry_orders_before_close=bool(
                data.get('cancel_entry_orders_before_close', True)
            ),
            set_trading_lock=bool(data.get('set_trading_lock', True)),
        )

    @staticmethod
    def _deduplicate_windows(windows: List[str]) -> List[str]:
        """Remove windows that map to the same number of minutes, keeping first."""
        seen: set = set()
        result = []
        for w in windows:
            try:
                s = str(w).lower()
                if s.endswith('m'):
                    mins = int(s[:-1])
                elif s.endswith('h'):
                    mins = int(s[:-1]) * 60
                elif s.endswith('d'):
                    mins = int(s[:-1]) * 1440
                else:
                    mins = int(s)
            except Exception:
                mins = w  # type: ignore
            if mins not in seen:
                seen.add(mins)
                result.append(str(w))
        return result

    def _resolve_env(self, val: Optional[str]) -> str:
        if not val: return ""
        if val.startswith("$"):
            return os.environ.get(val[1:], "").strip()
        if val.isupper() and "_" in val and " " not in val:
            env_val = os.environ.get(val)
            if env_val: return env_val.strip()
        return val.strip()

# ==============================================================================
# SECTION 3: STORAGE & DRAWDOWN LOGIC
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

            # Legacy de-risked state (kept for backward compat, no longer written by main loop)
            self.conn.execute('''CREATE TABLE IF NOT EXISTS scope_state (
                scope TEXT PRIMARY KEY,
                derisked_until INTEGER,
                derisked_mode TEXT,
                updated_ts INTEGER,
                details TEXT
            )''')

        # New tables via helper modules
        StageMachine(self.conn)._ensure_table()
        PositionStore(self.conn)._ensure_table()

    def append_snapshot(self, s: EquitySnapshot):
        if not s.quality_ok: return
        with self.conn:
            self.conn.execute(
                'INSERT OR IGNORE INTO snapshots (ts, exchange, account, equity, wallet, upnl) '
                'VALUES (?,?,?,?,?,?)',
                (s.ts, s.scope.exchange.value, s.scope.account.value,
                 s.equity_usdt, s.wallet_usdt, s.upnl_usdt)
            )

    def get_history(self, scope: Scope, lookback_sec: int, now_ts: int) -> List[Tuple[int, float]]:
        min_ts = now_ts - lookback_sec
        cur = self.conn.cursor()
        cur.execute(
            'SELECT ts, equity FROM snapshots WHERE exchange=? AND account=? AND ts>=? ORDER BY ts ASC',
            (scope.exchange.value, scope.account.value, min_ts)
        )
        return cur.fetchall()

    def in_cooldown(self, scope: Scope, tier_name: str, now_ts: int) -> bool:
        cur = self.conn.cursor()
        cur.execute('SELECT until FROM cooldowns WHERE scope=? AND tier=?', (str(scope), tier_name))
        row = cur.fetchone()
        return bool(row and row[0] > now_ts)

    def try_set_cooldown(self, scope: Scope, tier_name: str, until: int) -> bool:
        cur = self.conn.cursor()
        cur.execute('SELECT until FROM cooldowns WHERE scope=? AND tier=?', (str(scope), tier_name))
        row = cur.fetchone()
        now = int(time.time())
        if row and row[0] > now:
            return False
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO cooldowns VALUES (?,?,?)',
                              (str(scope), tier_name, until))
        return True

    # Legacy de-risked API (kept so old tests continue to pass)
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
                'INSERT OR REPLACE INTO scope_state(scope, derisked_until, derisked_mode, updated_ts, details) '
                'VALUES (?,?,?,?,?)',
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
        """Normalize window to minutes. Handles '15m', '1h', '1d', or int."""
        if isinstance(w, int): return w
        s = str(w).lower()
        if s.endswith('m'): return int(s[:-1])
        if s.endswith('h'): return int(s[:-1]) * 60
        if s.endswith('d'): return int(s[:-1]) * 1440
        return int(s)

    def compute(self, scope: Scope, now: int, windows: List[str]) -> Dict[str, float]:
        if not windows: return {}
        w_minutes = [self.parse_window(w) for w in windows]
        max_win = max(w_minutes)
        hist = self.store.get_history(scope, max_win * 60 + 300, now)
        if not hist: return {str(w): 0.0 for w in windows}
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
            res[w_str] = (hwm - curr_eq) / hwm if hwm > 0 else 0.0
        return res

    def pick_trigger_window(self, dds: Dict[str, float], thresholds: Dict[str, float]) -> Optional[Tuple[float, str, float]]:
        best = None
        for w_key, thr in thresholds.items():
            dd = float(dds.get(w_key, 0.0))
            if thr > 0 and dd >= thr:
                score = dd / thr
                if best is None or score > best[0]:
                    best = (score, w_key, thr)
        return best

    def is_confirmed(self, scope: Scope, thresh: float, window: Any, now: int,
                     consecutive: int, debug: bool = False) -> bool:
        if consecutive <= 1:
            return True
        w_min = self.parse_window(window)
        lookback = (w_min * 60) + (consecutive * 120)
        hist = self.store.get_history(scope, lookback, now)
        if len(hist) < consecutive:
            return False
        latest_points = hist[-consecutive:]
        breach_count = 0
        for pt_ts, pt_eq in latest_points:
            win_start = pt_ts - (w_min * 60)
            candidates = [e for (t, e) in hist if win_start <= t <= pt_ts]
            if not candidates:
                continue
            past_hwm = max(candidates)
            if past_hwm > 0:
                dd = (past_hwm - pt_eq) / past_hwm
                if dd >= thresh:
                    breach_count += 1
        return breach_count >= consecutive

# ==============================================================================
# SECTION 4: CIRCUIT BREAKER
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
                print(f"[CIRCUIT BREAKER OPEN] {scope} ({self.failures[str(scope)]} failures)")

    def record_success(self, scope: Scope):
        if str(scope) in self.broken:
            print(f"[CIRCUIT BREAKER CLOSED] {scope}")
            self.broken.remove(str(scope))
        self.failures[str(scope)] = 0

    def is_broken(self, scope: Scope) -> bool:
        return str(scope) in self.broken

# ==============================================================================
# SECTION 5: EXCHANGE ADAPTERS
# ==============================================================================

class BaseAdapter(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def fetch_equity(self, scope: Scope) -> EquitySnapshot: pass

    @abstractmethod
    def close_all(self, scope: Scope, longs_only: bool, dry_run: bool) -> ActionResult: pass

    @abstractmethod
    def sell_spot(self, scope: Scope, keep: List[str], blacklist: Optional[List[str]], dry_run: bool) -> ActionResult: pass

    @abstractmethod
    def fetch_positions_as_snapshots(self, scope: Scope) -> List[PositionSnapshot]: pass

    @abstractmethod
    def fetch_open_orders(self, scope: Scope) -> List[dict]: pass

    @abstractmethod
    def cancel_entry_orders(self, scope: Scope, symbols: Optional[List[str]] = None) -> ActionResult: pass

    @abstractmethod
    def cancel_orphan_reduce_only_orders(self, scope: Scope) -> ActionResult: pass

    @abstractmethod
    def close_positions_by_plan(self, scope: Scope, plan: List[CloseInstruction], dry_run: bool) -> ActionResult: pass


class RealCCXTAdapter(BaseAdapter):
    def __init__(self, config: ExchangeConfig, ex_id: ExchangeId,
                 spot_routing_delta_only: bool = True):
        super().__init__(config)
        self.ex_id = ex_id
        self.spot_routing_delta_only = spot_routing_delta_only
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
            print(f"[DEBUG] Bitget client created")
            print(f"[DEBUG] Bitget API Key: {opts['apiKey'][:10]}... (len={len(opts['apiKey'])})")
            if 'password' in opts:
                print(f"[DEBUG] Bitget Password: {opts['password'][:5]}... (len={len(opts['password'])})")

    def _retry_with_backoff(self, func, max_retries=MAX_RETRIES):
        for attempt in range(max_retries):
            try:
                return func()
            except ccxt.NetworkError as e:
                if attempt == max_retries - 1:
                    raise
                delay = RETRY_BASE_DELAY ** attempt + random.random()
                print(f"  [RETRY {attempt+1}/{max_retries}] Network error, waiting {delay:.1f}s: {e}")
                time.sleep(delay)
            except Exception:
                raise

    def _futures_params(self) -> dict:
        params = {}
        if self.ex_id == ExchangeId.BITGET:
            params['productType'] = 'umcbl'
        elif self.ex_id == ExchangeId.BYBIT:
            params['category'] = 'linear'
            params['settleCoin'] = 'USDT'
        return params

    def fetch_equity(self, scope: Scope) -> EquitySnapshot:
        ts = int(time.time())
        try:
            if self.ex_id == ExchangeId.BINANCE:
                if scope.account == AccountType.FUTURES:
                    bal = self._retry_with_backoff(lambda: self.client_futures.fetch_balance())
                    info = bal.get('info', {})
                    wallet = float(info.get('totalWalletBalance', bal['total'].get('USDT', 0)))
                    upnl = float(info.get('totalUnrealizedProfit', 0))
                    return EquitySnapshot(ts, scope, wallet + upnl, wallet, upnl, True, {'src': 'binance_f'})
                else:
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
                    return self._fetch_spot_equity(scope, self.client, market_type='spot')

            elif self.ex_id == ExchangeId.BITGET:
                if scope.account == AccountType.FUTURES:
                    bal = self._retry_with_backoff(lambda: self.client_futures.fetch_balance(params={'type': 'swap'}))
                    info = bal.get('info')
                    if isinstance(info, dict):
                        for k in ('data', 'result', 'list', 'accounts'):
                            v = info.get(k)
                            if isinstance(v, list):
                                info = v
                                break
                    if isinstance(info, list) and len(info) > 0:
                        for acc in info:
                            if not isinstance(acc, dict): continue
                            coin = (acc.get('marginCoin') or acc.get('coin') or '').upper()
                            if coin != 'USDT': continue
                            eq = float(acc.get('usdtEquity') or acc.get('equity') or acc.get('totalEquity') or 0.0)
                            wallet = acc.get('available') or acc.get('wallet') or None
                            upnl = acc.get('unrealizedPL') or acc.get('uPnL') or None
                            if eq > 0:
                                return EquitySnapshot(ts, scope, eq,
                                                      float(wallet) if wallet else None,
                                                      float(upnl) if upnl else None,
                                                      True, {'src': 'bitget'})
                        first_acc = info[0]
                        eq = float(first_acc.get('usdtEquity', 0))
                        return EquitySnapshot(ts, scope, eq,
                                              float(first_acc.get('available', 0)),
                                              float(first_acc.get('unrealizedPL', 0)),
                                              True, {'src': 'bitget_default'})
                    return EquitySnapshot(ts, scope, 0.0, None, None, False, {'err': 'Bitget Parse Failed'})
                else:
                    return self._fetch_spot_equity(scope, self.client, market_type=None)

        except Exception as e:
            error_msg = f"{self.ex_id.value} {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {scope} {error_msg}")
            if not isinstance(e, ccxt.NetworkError):
                traceback.print_exc()
            return EquitySnapshot(ts, scope, 0.0, None, None, False, {'err': error_msg})

        return EquitySnapshot(ts, scope, 0.0, quality_ok=False)

    def _fetch_spot_equity(self, scope: Scope, client, market_type=None) -> EquitySnapshot:
        ts = int(time.time())
        if market_type:
            bal = self._retry_with_backoff(lambda: client.fetch_balance(params={'type': market_type}))
        else:
            bal = self._retry_with_backoff(lambda: client.fetch_balance())

        raw_assets = {c: amt for c, amt in bal['total'].items() if amt > 0}
        assets = {}
        for c, amt in raw_assets.items():
            if c in STABLECOINS:
                assets[c] = amt
                continue
            if amt < MIN_BALANCE_FILTER:
                continue
            assets[c] = amt

        total_usdt = assets.get('USDT', 0.0)
        if not assets or (len(assets) == 1 and 'USDT' in assets and total_usdt == 0):
            return EquitySnapshot(ts, scope, 0.0, 0.0, 0, True, {'src': 'spot_empty'})

        needed_pairs = set()
        for c in assets:
            if c == 'USDT': continue
            needed_pairs.add(f"{c}/USDT")
            needed_pairs.add(f"{c}/BTC")
            needed_pairs.add(f"{c}/ETH")
        needed_pairs.update(['BTC/USDT', 'ETH/USDT'])

        if not needed_pairs:
            return EquitySnapshot(ts, scope, total_usdt, total_usdt, 0, True, {'src': 'spot_usdt_only'})

        ticker_params = {}
        if market_type:
            ticker_params['type'] = market_type

        try:
            if ticker_params:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(list(needed_pairs), params=ticker_params))
            else:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(list(needed_pairs)))
        except Exception as e:
            print(f"[DEBUG] {scope} fetch_tickers(symbols) failed: {e}, falling back")
            if ticker_params:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers(params=ticker_params))
            else:
                tickers = self._retry_with_backoff(lambda: client.fetch_tickers())

        btc_usdt = tickers.get('BTC/USDT', {}).get('bid', 0)
        eth_usdt = tickers.get('ETH/USDT', {}).get('bid', 0)

        total_notional_estimate = 0.0
        priced_notional = 0.0
        unpriced_assets = []

        for c, amt in assets.items():
            if c == 'USDT': continue
            price = self._get_price_multihop(c, tickers, btc_usdt, eth_usdt)
            if price > 0:
                notional = amt * price
                if notional < MIN_NOTIONAL_FILTER:
                    continue
                total_usdt += notional
                priced_notional += notional
                total_notional_estimate += notional
            else:
                fallback_value = amt * UNPRICED_FALLBACK_USDT
                total_notional_estimate += fallback_value
                unpriced_assets.append(c)

        coverage = priced_notional / total_notional_estimate if total_notional_estimate > 0 else 1.0
        unpriced_value = total_notional_estimate - priced_notional
        quality_ok = coverage >= SPOT_COVERAGE_MIN and unpriced_value <= SPOT_UNPRICED_USDT_MAX

        if not quality_ok:
            print(f"[WARNING] {scope} Spot coverage FAILED: {coverage:.1%}, ${unpriced_value:.0f} unpriced")

        return EquitySnapshot(ts, scope, total_usdt, total_usdt, 0, quality_ok)

    def _get_price_multihop(self, coin: str, tickers: dict, btc_usdt: float, eth_usdt: float) -> float:
        pair = f"{coin}/USDT"
        if pair in tickers:
            price = tickers[pair].get('bid', 0)
            if price > 0: return price
        if btc_usdt > 0:
            pair_btc = f"{coin}/BTC"
            if pair_btc in tickers:
                price_btc = tickers[pair_btc].get('bid', 0)
                if price_btc > 0:
                    return price_btc * btc_usdt
        if eth_usdt > 0:
            pair_eth = f"{coin}/ETH"
            if pair_eth in tickers:
                price_eth = tickers[pair_eth].get('bid', 0)
                if price_eth > 0:
                    return price_eth * eth_usdt
        return 0

    def _safe_futures_close(self, symbol, side, amount, is_hedge_mode_candidate):
        client = self.client_futures
        close_side = 'sell' if side == 'long' else 'buy'
        strategies = [{'reduceOnly': True}]
        if self.ex_id == ExchangeId.BYBIT and is_hedge_mode_candidate:
            idx = 1 if side == 'long' else 2
            strategies.append({'reduceOnly': True, 'positionIdx': idx})
            strategies.append({'reduceOnly': True, 'positionIdx': 0})
        last_exception = None
        for params in strategies:
            try:
                self._retry_with_backoff(lambda: client.create_order(
                    symbol, 'market', close_side, amount, params=params))
                return
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if "position idx" in err_str or "mode" in err_str or "reduce" in err_str:
                    continue
                continue
        if last_exception:
            raise last_exception

    def close_all(self, scope: Scope, longs_only: bool, dry_run: bool) -> ActionResult:
        client = self.client_futures
        try:
            params = self._futures_params()
            positions = self._retry_with_backoff(lambda: client.fetch_positions(params=params))
            to_close = []
            for p in positions:
                amt = float(p.get('contracts', 0) or p.get('info', {}).get('size', 0))
                if amt == 0: continue
                if longs_only and p['side'] == 'short': continue
                to_close.append(p)

            if not to_close:
                return ActionResult(True, "No positions")

            total_count = 0
            all_errors = []
            for p in to_close:
                symbol = p['symbol']
                side = p['side']
                for retry in range(MAX_RETRIES):
                    try:
                        current_positions = self._retry_with_backoff(lambda: client.fetch_positions(params=params))
                        current_pos = next((x for x in current_positions if x['symbol'] == symbol and x['side'] == side), None)
                        if not current_pos:
                            break
                        qty_raw = abs(float(current_pos.get('contracts', 0) or current_pos.get('info', {}).get('size', 0)))
                        if qty_raw == 0:
                            break
                        if dry_run:
                            total_count += 1
                            break
                        amount = client.amount_to_precision(symbol, qty_raw)
                        self._safe_futures_close(symbol, side, amount, self.ex_id == ExchangeId.BYBIT)
                        total_count += 1
                        print(f"  Closed {symbol} {side} {amount}")
                        break
                    except Exception as e:
                        if retry == MAX_RETRIES - 1:
                            all_errors.append(f"{symbol}: {e}")
                        else:
                            time.sleep(RETRY_BASE_DELAY ** retry)

            return ActionResult(len(all_errors) == 0, f"Closed {total_count}/{len(to_close)}", total_count, all_errors)
        except Exception as e:
            return ActionResult(False, f"Crash: {e}")

    def fetch_positions_as_snapshots(self, scope: Scope) -> List[PositionSnapshot]:
        ts = int(time.time())
        try:
            params = self._futures_params()
            positions = self._retry_with_backoff(
                lambda: self.client_futures.fetch_positions(params=params))
            result = []
            for p in positions:
                amt = float(p.get('contracts', 0) or p.get('info', {}).get('size', 0) or 0)
                if amt == 0:
                    continue
                symbol = p.get('symbol', '')
                side = p.get('side', '')
                entry = p.get('entryPrice') or p.get('info', {}).get('entryPrice')
                mark = p.get('markPrice') or p.get('info', {}).get('markPrice')
                liq = p.get('liquidationPrice') or p.get('info', {}).get('liqPrice')
                notional = float(p.get('notional', 0) or (abs(amt) * float(mark or 0)))
                margin = (p.get('initialMargin')
                          or p.get('info', {}).get('positionInitialMargin')
                          or p.get('info', {}).get('initialMargin'))
                leverage = p.get('leverage') or p.get('info', {}).get('leverage')
                upnl = float(p.get('unrealizedPnl', 0) or p.get('info', {}).get('unrealizedPnl', 0) or 0)
                pnl_pct = None
                if margin and float(margin) > 0:
                    pnl_pct = upnl / float(margin)
                result.append(PositionSnapshot(
                    ts=ts, scope=scope, symbol=symbol, side=side,
                    contracts=amt, notional_usdt=notional,
                    entry_price=float(entry) if entry else None,
                    mark_price=float(mark) if mark else None,
                    liquidation_price=float(liq) if liq else None,
                    margin_usdt=float(margin) if margin else None,
                    leverage=float(leverage) if leverage else None,
                    unrealized_pnl_usdt=upnl,
                    pnl_pct_on_margin=pnl_pct,
                    raw=p.get('info', {}),
                ))
            return result
        except Exception as e:
            print(f"[ERROR] fetch_positions_as_snapshots {scope}: {e}")
            return []

    def fetch_open_orders(self, scope: Scope) -> List[dict]:
        try:
            client = self.client_futures if scope.account == AccountType.FUTURES else self.client
            return self._retry_with_backoff(lambda: client.fetch_open_orders()) or []
        except Exception as e:
            print(f"[ERROR] fetch_open_orders {scope}: {e}")
            return []

    def cancel_entry_orders(self, scope: Scope, symbols: Optional[List[str]] = None) -> ActionResult:
        """Cancel open non-reduce-only orders (entry orders) before a kill action."""
        try:
            client = self.client_futures if scope.account == AccountType.FUTURES else self.client
            orders = self._retry_with_backoff(lambda: client.fetch_open_orders()) or []
            cancelled = 0
            errors = []
            for order in orders:
                # Skip reduce-only orders — they are position-closing orders
                if order.get('reduceOnly') or order.get('info', {}).get('reduceOnly'):
                    continue
                if symbols and order.get('symbol') not in symbols:
                    continue
                try:
                    self._retry_with_backoff(lambda: client.cancel_order(order['id'], order['symbol']))
                    cancelled += 1
                    print(f"  Cancelled entry order {order['id']} {order['symbol']}")
                except Exception as e:
                    errors.append(f"{order.get('id', '?')}: {e}")
            return ActionResult(len(errors) == 0, f"Cancelled {cancelled} entry orders", cancelled, errors)
        except Exception as e:
            return ActionResult(False, f"cancel_entry_orders crash: {e}")

    def cancel_orphan_reduce_only_orders(self, scope: Scope) -> ActionResult:
        """Cancel reduce-only orders that remain after positions are closed."""
        try:
            client = self.client_futures
            orders = self._retry_with_backoff(lambda: client.fetch_open_orders()) or []
            cancelled = 0
            errors = []
            for order in orders:
                if not (order.get('reduceOnly') or order.get('info', {}).get('reduceOnly')):
                    continue
                try:
                    self._retry_with_backoff(lambda: client.cancel_order(order['id'], order['symbol']))
                    cancelled += 1
                except Exception as e:
                    errors.append(f"{order.get('id', '?')}: {e}")
            return ActionResult(len(errors) == 0, f"Cancelled {cancelled} orphan orders", cancelled, errors)
        except Exception as e:
            return ActionResult(False, f"cancel_orphan crash: {e}")

    def close_positions_by_plan(self, scope: Scope, plan: List[CloseInstruction], dry_run: bool) -> ActionResult:
        """Execute a risk-ranked close plan with reduce-only market orders."""
        client = self.client_futures
        params = self._futures_params()
        total_count = 0
        all_errors = []

        for instr in plan:
            try:
                current_positions = self._retry_with_backoff(
                    lambda: client.fetch_positions(params=params))
                current_pos = next(
                    (x for x in current_positions
                     if x['symbol'] == instr.symbol and x['side'] == instr.side), None)
                if not current_pos:
                    print(f"  {instr.symbol} {instr.side} already closed")
                    continue

                qty_raw = abs(float(
                    current_pos.get('contracts', 0)
                    or current_pos.get('info', {}).get('size', 0)
                ))
                if qty_raw == 0:
                    continue

                qty_to_close = qty_raw * instr.close_fraction

                if dry_run:
                    total_count += 1
                    print(
                        f"  [DRY RUN] {instr.symbol} {instr.side} "
                        f"{qty_to_close:.4f} x{instr.close_fraction:.0%} "
                        f"score={instr.risk_score:.2f} delta={instr.pnl_delta_usdt:.2f} | {instr.reason}"
                    )
                    continue

                amount = client.amount_to_precision(instr.symbol, qty_to_close)
                self._safe_futures_close(instr.symbol, instr.side, amount, self.ex_id == ExchangeId.BYBIT)
                total_count += 1
                print(f"  Closed {instr.symbol} {instr.side} {amount} | {instr.reason}")

            except Exception as e:
                all_errors.append(f"{instr.symbol}: {e}")
                print(f"  Failed {instr.symbol}: {e}")

        return ActionResult(
            len(all_errors) == 0,
            f"Executed {total_count}/{len(plan)} instructions",
            total_count,
            all_errors,
        )

    def _sell_spot_routed(self, client, coin, amount, tickers) -> bool:
        """Sell coin to USDT via direct pair, BTC intermediate, or ETH intermediate.

        Uses delta-only mode by default: only the newly-acquired BTC/ETH from
        hop-1 is sold, never pre-existing holdings.
        """
        delta_only = self.spot_routing_delta_only

        # 1. Direct USDT pair
        pair = f"{coin}/USDT"
        if pair in tickers and tickers[pair]['bid'] > 0:
            amt = client.amount_to_precision(pair, amount)
            self._retry_with_backoff(lambda: client.create_order(pair, 'market', 'sell', amt))
            print(f"  Sold {pair} {amt}")
            return True

        # 2. Via BTC
        pair_btc = f"{coin}/BTC"
        if pair_btc in tickers and tickers[pair_btc]['bid'] > 0:
            btc_before = 0.0
            if delta_only:
                try:
                    b = self._retry_with_backoff(lambda: client.fetch_balance())
                    btc_before = float(b.get('free', {}).get('BTC', 0) or 0)
                except Exception:
                    pass

            amt = client.amount_to_precision(pair_btc, amount)
            self._retry_with_backoff(lambda: client.create_order(pair_btc, 'market', 'sell', amt))
            print(f"  Sold {pair_btc} {amt} (hop 1/2)")
            time.sleep(0.5)

            try:
                b_after = self._retry_with_backoff(lambda: client.fetch_balance())
                btc_after = float(b_after.get('free', {}).get('BTC', 0) or 0)
                btc_to_sell = max(0.0, btc_after - btc_before) if delta_only else btc_after
                if btc_to_sell <= 0:
                    print(f"  BTC delta is zero — no hop-2 needed")
                    return True
                btc_usdt_bid = tickers.get('BTC/USDT', {}).get('bid', 0)
                notional = btc_to_sell * btc_usdt_bid
                if notional < MIN_NOTIONAL_USDT:
                    print(f"  BTC delta ${notional:.2f} < min notional — leaving as dust")
                    return True
                amt_btc = client.amount_to_precision('BTC/USDT', btc_to_sell)
                self._retry_with_backoff(lambda: client.create_order('BTC/USDT', 'market', 'sell', amt_btc))
                print(f"  Sold BTC/USDT {amt_btc} (hop 2/2 delta={delta_only})")
                return True
            except Exception as e:
                print(f"  Sold {coin}/BTC but failed BTC/USDT: {e}")
                return True

        # 3. Via ETH
        pair_eth = f"{coin}/ETH"
        if pair_eth in tickers and tickers[pair_eth]['bid'] > 0:
            eth_before = 0.0
            if delta_only:
                try:
                    b = self._retry_with_backoff(lambda: client.fetch_balance())
                    eth_before = float(b.get('free', {}).get('ETH', 0) or 0)
                except Exception:
                    pass

            amt = client.amount_to_precision(pair_eth, amount)
            self._retry_with_backoff(lambda: client.create_order(pair_eth, 'market', 'sell', amt))
            print(f"  Sold {pair_eth} {amt} (hop 1/2)")
            time.sleep(0.5)

            try:
                b_after = self._retry_with_backoff(lambda: client.fetch_balance())
                eth_after = float(b_after.get('free', {}).get('ETH', 0) or 0)
                eth_to_sell = max(0.0, eth_after - eth_before) if delta_only else eth_after
                if eth_to_sell <= 0:
                    print(f"  ETH delta is zero — no hop-2 needed")
                    return True
                eth_usdt_bid = tickers.get('ETH/USDT', {}).get('bid', 0)
                notional = eth_to_sell * eth_usdt_bid
                if notional < MIN_NOTIONAL_USDT:
                    print(f"  ETH delta ${notional:.2f} < min notional — leaving as dust")
                    return True
                amt_eth = client.amount_to_precision('ETH/USDT', eth_to_sell)
                self._retry_with_backoff(lambda: client.create_order('ETH/USDT', 'market', 'sell', amt_eth))
                print(f"  Sold ETH/USDT {amt_eth} (hop 2/2 delta={delta_only})")
                return True
            except Exception as e:
                print(f"  Sold {coin}/ETH but failed ETH/USDT: {e}")
                return True

        print(f"  No route for {coin}")
        return False

    def sell_spot(self, scope: Scope, keep: List[str], blacklist: Optional[List[str]], dry_run: bool) -> ActionResult:
        client = self.client
        try:
            bal = self._retry_with_backoff(lambda: client.fetch_balance())
            assets = bal['free']
            tickers = self._retry_with_backoff(lambda: client.fetch_tickers())
            to_sell = []
            for c, amt in assets.items():
                if c in keep: continue
                if blacklist and c not in blacklist: continue
                if amt <= 0: continue
                btc_usdt = tickers.get('BTC/USDT', {}).get('bid', 0)
                eth_usdt = tickers.get('ETH/USDT', {}).get('bid', 0)
                price = self._get_price_multihop(c, tickers, btc_usdt, eth_usdt)
                notional = amt * price
                if notional < MIN_NOTIONAL_USDT:
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
                    if self._sell_spot_routed(client, c, amt, tickers):
                        count += 1
                    else:
                        errors.append(f"{c}: No route found")
                except Exception as e:
                    errors.append(f"{c}: {e}")

            return ActionResult(len(errors) == 0, f"Sold {count}/{len(to_sell)}", count, errors)
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
        self._mock_positions: List[PositionSnapshot] = []
        self._mock_orders: List[dict] = []

    def set_scenario(self, data):
        self.scenario = data

    def set_mock_positions(self, positions: List[PositionSnapshot]):
        self._mock_positions = positions

    def set_mock_orders(self, orders: List[dict]):
        self._mock_orders = orders

    def fetch_equity(self, scope: Scope) -> EquitySnapshot:
        val = self.scenario[min(self.idx, len(self.scenario) - 1)]
        ts = self.base_ts + (self.idx * 60)
        self.idx += 1
        return EquitySnapshot(ts, scope, float(val), float(val), 0)

    def close_all(self, scope: Scope, longs_only: bool, dry_run: bool) -> ActionResult:
        return ActionResult(True, "[MOCK] close_all")

    def sell_spot(self, scope: Scope, keep: List[str], blacklist: Optional[List[str]], dry_run: bool) -> ActionResult:
        return ActionResult(True, "[MOCK] sell_spot")

    def fetch_positions_as_snapshots(self, scope: Scope) -> List[PositionSnapshot]:
        return list(self._mock_positions)

    def fetch_open_orders(self, scope: Scope) -> List[dict]:
        return list(self._mock_orders)

    def cancel_entry_orders(self, scope: Scope, symbols: Optional[List[str]] = None) -> ActionResult:
        self._mock_orders = [o for o in self._mock_orders
                             if o.get('reduceOnly') or (symbols and o.get('symbol') not in symbols)]
        return ActionResult(True, "[MOCK] cancel_entry_orders")

    def cancel_orphan_reduce_only_orders(self, scope: Scope) -> ActionResult:
        return ActionResult(True, "[MOCK] cancel_orphan_reduce_only_orders")

    def close_positions_by_plan(self, scope: Scope, plan: List[CloseInstruction], dry_run: bool) -> ActionResult:
        return ActionResult(True, f"[MOCK] close_positions_by_plan {len(plan)} instructions")

# ==============================================================================
# SECTION 6: ACTION ENGINE
# ==============================================================================

class ActionEngine:
    def __init__(self, store: SqliteStore, adapters: dict, dry: bool,
                 stables: List[str], breaker: CircuitBreaker,
                 stage_machine: Optional[StageMachine] = None,
                 position_store: Optional[PositionStore] = None,
                 trading_lock: Optional[TradingLock] = None):
        self.store = store
        self.adapters = adapters
        self.dry = dry
        self.stables = stables
        self.breaker = breaker
        self.stage_machine = stage_machine
        self.position_store = position_store
        self.trading_lock = trading_lock

    # ------------------------------------------------------------------
    # Stage-based execution (new)
    # ------------------------------------------------------------------

    def execute_stage(self, decision: StageDecision, conf: StageConfig,
                      stage_machine: StageMachine) -> bool:
        """Execute one stage action atomically via stage machine. Returns True if executed."""
        scope = decision.scope
        stage = decision.stage
        now_ts = decision.ts

        can, reason = stage_machine.can_execute(str(scope), stage, now_ts)
        if not can:
            print(f"  [BLOCKED] Stage {stage}: {reason}")
            return False

        adapter = self.adapters.get(scope.exchange.value)
        if not adapter:
            print(f"  [ERROR] No adapter for {scope.exchange.value}")
            return False

        print(f"  >>> Stage {stage} | {conf.mode} | {scope}")

        res = ActionResult(False, "Unknown mode")
        try:
            # Cancel entry orders before futures close actions
            if (conf.cancel_entry_orders_before_close
                    and scope.account == AccountType.FUTURES
                    and conf.mode in ("CLOSE_TOP_RISK_CONTRIBUTORS",
                                      "CLOSE_DOMINANT_LOSS_DIRECTION",
                                      "CLOSE_ALL_POSITIONS",
                                      "CLOSE_LONGS_ONLY")):
                cancel_res = adapter.cancel_entry_orders(scope)
                print(f"  Entry orders: {cancel_res.details}")

            if conf.mode == "CLOSE_TOP_RISK_CONTRIBUTORS":
                res = self._exec_close_top_risk(scope, conf, adapter, now_ts)
            elif conf.mode == "CLOSE_DOMINANT_LOSS_DIRECTION":
                res = self._exec_close_dominant(scope, conf, adapter, now_ts)
            elif conf.mode == "CLOSE_ALL_POSITIONS":
                res = self._exec_close_all(scope, conf, adapter)
            elif conf.mode == "CLOSE_LONGS_ONLY":
                print(f"  [WARNING] CLOSE_LONGS_ONLY is deprecated — "
                      f"consider migrating to CLOSE_TOP_RISK_CONTRIBUTORS")
                res = adapter.close_all(scope, True, self.dry)
            elif conf.mode == "SELL_BLACKLIST_ONLY_KEEP_STABLES":
                res = adapter.sell_spot(scope, self.stables, conf.blacklist, self.dry)
            elif conf.mode == "SELL_ALL_NON_USDT_KEEP_STABLES":
                res = adapter.sell_spot(scope, self.stables, None, self.dry)

            print(f"  [RESULT] success={res.success} | {res.details}")

            if res.success or res.orders_placed > 0:
                self.breaker.record_success(scope)
                lock_until = now_ts + (conf.cooldown_min * 60)
                details = {
                    'mode': conf.mode,
                    'result': res.details,
                    'dd': {k: float(v) for k, v in decision.dd.items()},
                }
                stage_machine.record_execution(
                    str(scope), stage, lock_until, conf.mode, details, now_ts)

                if conf.set_trading_lock and self.trading_lock and scope.account == AccountType.FUTURES:
                    reason_str = f"killswitch_stage_{stage}_{conf.mode.lower()}"
                    self.trading_lock.set_lock(
                        str(scope), stage, reason_str, conf.cooldown_min * 60, details)
                    print(f"  [TRADING LOCK] Set for {scope} stage {stage}")
            else:
                self.breaker.record_failure(scope)

            bucket_ts = (now_ts // 300) * 300
            key = f"{scope}|stage{stage}|{conf.mode}|{bucket_ts}"
            self.store.record_action(key, str(res), now_ts)

            return bool(res.success or res.orders_placed > 0)

        except Exception as e:
            print(f"  [ERROR] Stage {stage} failed: {e}")
            traceback.print_exc()
            self.breaker.record_failure(scope)
            return False

    def _exec_close_top_risk(self, scope: Scope, conf: StageConfig,
                             adapter: BaseAdapter, now_ts: int) -> ActionResult:
        current = adapter.fetch_positions_as_snapshots(scope)
        if not current:
            return ActionResult(True, "No positions")

        ref_dict = (
            self.position_store.get_reference_snapshots(scope, ATTRIBUTION_LOOKBACK_SEC, now_ts)
            if self.position_store else {}
        )
        reference = list(ref_dict.values())

        attr = compute_pnl_attribution(current, reference, conf.source_threshold)
        print(f"  Attribution: source={attr.source} "
              f"long_neg={attr.long_negative_delta:.2f} "
              f"short_neg={attr.short_negative_delta:.2f}")

        hist = self.store.get_history(scope, 300, now_ts)
        equity = hist[-1][1] if hist else 10000.0

        ranked = rank_positions_by_risk(current, reference, equity, attr.source, conf.source_mode)
        top_n = min(conf.top_n, len(ranked))
        plan = []

        for pos, pnl_delta, risk_score in ranked[:top_n]:
            close_frac = conf.close_fraction
            if pos.mark_price and pos.liquidation_price and pos.mark_price > 0:
                if pos.side == "long":
                    dist_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price
                else:
                    dist_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price
                if dist_pct <= conf.full_close_if_liq_distance_below_pct:
                    close_frac = 1.0
                    print(f"  {pos.symbol} liq distance {dist_pct:.1%} -> full close")

            plan.append(CloseInstruction(
                symbol=pos.symbol,
                side=pos.side,
                contracts=pos.contracts,
                close_fraction=close_frac,
                reason=f"stage1_source_{attr.source.lower()}",
                risk_score=risk_score,
                pnl_delta_usdt=pnl_delta,
                unrealized_pnl_usdt=pos.unrealized_pnl_usdt,
            ))
            print(f"  Plan: {pos.symbol} {pos.side} {pos.contracts} "
                  f"x{close_frac:.0%} score={risk_score:.2f} delta={pnl_delta:.2f}")

        if not plan:
            return ActionResult(True, "No candidates")

        return adapter.close_positions_by_plan(scope, plan, self.dry)

    def _exec_close_dominant(self, scope: Scope, conf: StageConfig,
                             adapter: BaseAdapter, now_ts: int) -> ActionResult:
        current = adapter.fetch_positions_as_snapshots(scope)
        if not current:
            return ActionResult(True, "No positions")

        ref_dict = (
            self.position_store.get_reference_snapshots(scope, ATTRIBUTION_LOOKBACK_SEC, now_ts)
            if self.position_store else {}
        )
        reference = list(ref_dict.values())

        attr = compute_pnl_attribution(current, reference, conf.source_threshold)
        print(f"  Attribution: source={attr.source} "
              f"long_neg={attr.long_negative_delta:.2f} "
              f"short_neg={attr.short_negative_delta:.2f}")

        if attr.source in ("LONG", "SHORT"):
            target_side = "long" if attr.source == "LONG" else "short"
            plan = [
                CloseInstruction(
                    symbol=p.symbol, side=p.side, contracts=p.contracts,
                    close_fraction=1.0,
                    reason=f"stage2_dominant_{attr.source.lower()}",
                    risk_score=0.0, pnl_delta_usdt=0.0,
                    unrealized_pnl_usdt=p.unrealized_pnl_usdt,
                )
                for p in current if p.side == target_side
            ]
        else:
            hist = self.store.get_history(scope, 300, now_ts)
            equity = hist[-1][1] if hist else 10000.0
            ranked = rank_positions_by_risk(current, reference, equity, "MIXED", "AUTO")
            n = max(1, int(len(ranked) * 0.60))
            plan = [
                CloseInstruction(
                    symbol=pos.symbol, side=pos.side, contracts=pos.contracts,
                    close_fraction=1.0, reason="stage2_mixed_worst60pct",
                    risk_score=risk_score, pnl_delta_usdt=pnl_delta,
                    unrealized_pnl_usdt=pos.unrealized_pnl_usdt,
                )
                for pos, pnl_delta, risk_score in ranked[:n]
            ]

        if not plan:
            return ActionResult(True, "No candidates")

        return adapter.close_positions_by_plan(scope, plan, self.dry)

    def _exec_close_all(self, scope: Scope, conf: StageConfig,
                        adapter: BaseAdapter) -> ActionResult:
        # Cancel non-reduce-only orders first
        c_res = adapter.cancel_entry_orders(scope)
        print(f"  Entry orders cancelled: {c_res.details}")

        # Close all positions
        res = adapter.close_all(scope, False, self.dry)

        # Cancel orphan reduce-only orders after positions are flat
        if not self.dry:
            o_res = adapter.cancel_orphan_reduce_only_orders(scope)
            print(f"  Orphan orders cancelled: {o_res.details}")

        return res

    # ------------------------------------------------------------------
    # Legacy tier-based execution (kept for backward compat)
    # ------------------------------------------------------------------

    def execute(self, dec: Decision, conf: StageConfig):
        until = dec.ts + (conf.cooldown_min * 60)
        if not self.store.try_set_cooldown(dec.scope, dec.tier.value, until):
            print(f"  [SKIP] {dec.tier} already in cooldown")
            return

        adapter = self.adapters.get(dec.scope.exchange.value)
        if not adapter: return

        print(f"  >>> EXECUTING {dec.action_mode} | {dec.scope} | {dec.tier}")

        res = ActionResult(False, "Unknown mode")
        try:
            if dec.action_mode == "CLOSE_LONGS_ONLY":
                print(f"  [WARNING] CLOSE_LONGS_ONLY is deprecated — "
                      f"migrate to stage_1 with CLOSE_TOP_RISK_CONTRIBUTORS")
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
            else:
                self.breaker.record_failure(dec.scope)

            bucket_ts = (dec.ts // 300) * 300
            key = f"{dec.scope}|{dec.tier}|{dec.action_mode}|{bucket_ts}"
            self.store.record_action(key, str(res), dec.ts)

        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            self.breaker.record_failure(dec.scope)

# ==============================================================================
# SECTION 7: STAGE TRIGGER PROCESSING
# ==============================================================================

def _process_stage_triggers(
    scope: Scope,
    acc_conf: AccountConfig,
    dds: Dict[str, float],
    snap: EquitySnapshot,
    engine: ActionEngine,
    stage_machine: StageMachine,
    dd_calc: DrawdownCalculator,
    logger,
) -> None:
    """Find and execute the highest-triggered stage for this scope."""
    dd_str = ', '.join(f"{w}:{v:.1%}" for w, v in sorted(dds.items()))
    executed = False

    for stage_num in [3, 2, 1]:
        conf = acc_conf.get_stage(stage_num)
        if not conf:
            continue

        best = dd_calc.pick_trigger_window(dds, conf.thresholds)
        if not best:
            continue

        score, w, thr = best
        if not dd_calc.is_confirmed(scope, thr, w, snap.ts, conf.confirm_consecutive):
            continue

        # Quick pre-check (not atomic — prevents expensive position fetch)
        can_exec, block_reason = stage_machine.can_execute(str(scope), stage_num, snap.ts)
        if not can_exec:
            dd_val = dds.get(w, 0.0)
            logger.info(
                f"Stage {stage_num} blocked for {scope}: {block_reason} "
                f"| DD {w}:{dd_val:.4f}"
            )
            print(
                f"[{datetime.now().strftime('%H:%M')}] {scope} "
                f"[STAGE {stage_num} BLOCKED] {block_reason}"
            )
            continue

        dd_val = dds.get(w, 0.0)
        logger.warning(
            f"STAGE {stage_num} TRIGGER | {scope} | {conf.mode} | "
            f"Window={w} | DD={dd_val:.4f} > {thr}"
        )
        print(
            f"\n[STAGE {stage_num} TRIGGER] {scope} | {conf.mode} | "
            f"Window={w} | DD={dd_val:.4f} > {thr}"
        )

        decision = StageDecision(snap.ts, scope, stage_num, dds, conf.mode)
        executed = engine.execute_stage(decision, conf, stage_machine)

        if executed:
            break  # Execute only the highest triggered stage per cycle

    if not executed:
        state = stage_machine.get_state(str(scope))
        stage_label = STAGE_NAMES.get(state.current_stage, f"stage_{state.current_stage}")
        print(
            f"[{datetime.now().strftime('%H:%M')}] {scope} "
            f"[{stage_label}] Eq: ${snap.equity_usdt:.0f} | DD: {dd_str}"
        )

# ==============================================================================
# SECTION 8: BACKTEST MODE
# ==============================================================================

def run_backtest(csv_path: str, config_path: str):
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
    stage_machine = StageMachine(store.conn)
    engine = ActionEngine(store, {}, True, cfg.stables_keep, breaker, stage_machine)
    triggers = []

    for snap_data in snapshots:
        ex_id = ExchangeId(snap_data['exchange'])
        acc_type = AccountType(snap_data['account'])
        scope = Scope(ex_id, acc_type)
        snap = EquitySnapshot(snap_data['ts'], scope, snap_data['equity'],
                              snap_data['equity'], 0, True)
        store.append_snapshot(snap)

        ex_conf = cfg.exchanges.get(snap_data['exchange'])
        if not ex_conf: continue
        acc_conf = ex_conf.accounts.get(snap_data['account'])
        if not acc_conf or not acc_conf.enabled: continue

        dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)

        if acc_conf.has_stages:
            for stage_num in [3, 2, 1]:
                conf = acc_conf.get_stage(stage_num)
                if not conf: continue
                best = dd_calc.pick_trigger_window(dds, conf.thresholds)
                if not best: continue
                score, w, thr = best
                if not dd_calc.is_confirmed(scope, thr, w, snap.ts, conf.confirm_consecutive):
                    continue
                can, _ = stage_machine.can_execute(str(scope), stage_num, snap.ts)
                if not can: continue
                triggers.append({'ts': snap.ts, 'scope': str(scope),
                                  'stage': stage_num, 'dd': dds, 'mode': conf.mode})
                lock_until = snap.ts + (conf.cooldown_min * 60)
                stage_machine.record_execution(str(scope), stage_num, lock_until,
                                               conf.mode, {}, snap.ts)
                break
        else:
            for tier_name, tier_conf in [('B', acc_conf.tier_b), ('A', acc_conf.tier_a)]:
                if not tier_conf: continue
                best = dd_calc.pick_trigger_window(dds, tier_conf.thresholds)
                if not best: continue
                score, w, thr = best
                if not dd_calc.is_confirmed(scope, thr, w, snap.ts, tier_conf.confirm_consecutive):
                    continue
                if not store.in_cooldown(scope, tier_name, snap.ts):
                    triggers.append({'ts': snap.ts, 'scope': str(scope),
                                     'tier': tier_name, 'dd': dds, 'mode': tier_conf.mode})
                    until = snap.ts + (tier_conf.cooldown_min * 60)
                    store.try_set_cooldown(scope, tier_name, until)
                    break

    print(f"\n{'='*60}\nBACKTEST RESULTS\n{'='*60}")
    print(f"Total triggers: {len(triggers)}")
    for t in triggers:
        dt = datetime.fromtimestamp(t['ts'])
        stage_or_tier = f"Stage {t.get('stage', '?')}" if 'stage' in t else f"Tier {t.get('tier', '?')}"
        print(f"{dt} | {t['scope']} | {stage_or_tier} | {t['mode']} | DD: {t['dd']}")

# ==============================================================================
# SECTION 9: MAIN ENTRY
# ==============================================================================

def run():
    parser = argparse.ArgumentParser(description="Crypto Portfolio Kill-Switch v3.0")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--test-mock", action="store_true", help="Run self-test with mock data")
    parser.add_argument("--backtest", help="CSV file for backtest mode")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.backtest:
        if not args.config:
            print("Usage: --backtest equity.csv --config config.yaml")
            return
        run_backtest(args.backtest, args.config)
        return

    if args.test_mock:
        _run_test_mock()
        return

    if not args.config:
        print("Usage: killswitch.py --config config.yaml")
        print("       killswitch.py --backtest equity.csv --config config.yaml")
        print("       killswitch.py --test-mock")
        return

    loader = ConfigLoader()
    cfg = loader.load(args.config)

    log_file = 'reports/killswitch.log'
    os.makedirs('reports', exist_ok=True)
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = get_logger("killswitch", level=log_level, log_file=log_file)
    logger.info(f"Logger initialized, writing to {log_file}")

    store = SqliteStore(cfg.state_db)
    dd_calc = DrawdownCalculator(store)
    breaker = CircuitBreaker()
    stage_machine = StageMachine(store.conn)
    pos_store = PositionStore(store.conn)
    trading_lock = TradingLock(cfg.trading_lock_file)

    adapters = {}
    for name, c in cfg.exchanges.items():
        if c.enabled:
            adapters[name] = RealCCXTAdapter(
                c, ExchangeId(name),
                spot_routing_delta_only=cfg.spot_routing_delta_only,
            )

    if not adapters:
        logger.error("No exchanges enabled in config")
        print("No exchanges enabled in config.")
        return

    engine = ActionEngine(store, adapters, cfg.dry_run, cfg.stables_keep, breaker,
                          stage_machine=stage_machine,
                          position_store=pos_store,
                          trading_lock=trading_lock)

    mode_str = "DRY RUN" if cfg.dry_run else "LIVE"
    logger.info(f"{'='*60}")
    logger.info(f"KILL-SWITCH v3.0 — {mode_str}")
    logger.info(f"Exchanges: {list(adapters.keys())}")
    logger.info(f"Poll: {cfg.poll_seconds}s | DB: {cfg.state_db}")
    logger.info(f"Spot routing delta-only: {cfg.spot_routing_delta_only}")
    logger.info(f"{'='*60}")
    print(f"{'='*60}")
    print(f"KILL-SWITCH v3.0 — {mode_str}")
    print(f"Exchanges: {list(adapters.keys())}")
    print(f"Poll: {cfg.poll_seconds}s | Lock: {cfg.trading_lock_file}")
    print(f"{'='*60}\n")

    while True:
        cycle_start = time.time()

        for name, ex_config in cfg.exchanges.items():
            if name not in adapters: continue

            for acc_name, acc_conf in ex_config.accounts.items():
                if not acc_conf.enabled: continue

                scope = Scope(ExchangeId(name), AccountType(acc_name))

                if breaker.is_broken(scope):
                    print(f"[{datetime.now().strftime('%H:%M')}] {scope} CIRCUIT BROKEN — SKIPPING")
                    continue

                try:
                    snap = adapters[name].fetch_equity(scope)

                    if not snap.quality_ok:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} BAD READ: {snap.raw}")
                        breaker.record_failure(scope)
                        continue

                    if snap.equity_usdt is None or snap.equity_usdt < 0:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} INVALID EQUITY: {snap.equity_usdt}")
                        breaker.record_failure(scope)
                        continue

                    if snap.equity_usdt == 0:
                        print(f"[{datetime.now().strftime('%H:%M')}] {scope} Empty account — skipping DD")
                        breaker.record_success(scope)
                        store.append_snapshot(snap)
                        continue

                    store.append_snapshot(snap)

                    # Fetch and store position snapshots for attribution lookback
                    if scope.account == AccountType.FUTURES:
                        try:
                            pos_snaps = adapters[name].fetch_positions_as_snapshots(scope)
                            if pos_snaps:
                                pos_store.save_many(pos_snaps)
                        except Exception as pe:
                            logger.error(f"Position snapshot failed for {scope}: {pe}")

                    breaker.record_success(scope)
                    dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)

                    if acc_conf.has_stages:
                        _process_stage_triggers(scope, acc_conf, dds, snap,
                                                engine, stage_machine, dd_calc, logger)
                    else:
                        # Legacy tier logic — still uses old is_derisked guard
                        if store.is_derisked(scope, snap.ts):
                            derisked_until = store.get_derisked_until(scope)
                            remaining_min = (derisked_until - snap.ts) // 60
                            dd_str = ', '.join(f"{w}:{v:.1%}" for w, v in sorted(dds.items()))
                            print(
                                f"[{datetime.now().strftime('%H:%M')}] {scope} "
                                f"DE-RISKED ({remaining_min}m left) | "
                                f"Eq: ${snap.equity_usdt:.0f} | DD: {dd_str}"
                            )
                            continue

                        final_decision = None
                        target_tier_conf = None

                        if acc_conf.tier_b:
                            best_b = dd_calc.pick_trigger_window(dds, acc_conf.tier_b.thresholds)
                            if best_b:
                                score, w, thr = best_b
                                if dd_calc.is_confirmed(scope, thr, w, snap.ts,
                                                        acc_conf.tier_b.confirm_consecutive):
                                    final_decision = Decision(snap.ts, scope, Tier.B, dds, [],
                                                              acc_conf.tier_b.mode)
                                    target_tier_conf = acc_conf.tier_b
                                    logger.warning(
                                        f"TIER B TRIGGER | {scope} | Window={w} | "
                                        f"DD={dds.get(w, 0):.4f} > {thr}"
                                    )

                        if not final_decision and acc_conf.tier_a:
                            best_a = dd_calc.pick_trigger_window(dds, acc_conf.tier_a.thresholds)
                            if best_a:
                                score, w, thr = best_a
                                if dd_calc.is_confirmed(scope, thr, w, snap.ts,
                                                        acc_conf.tier_a.confirm_consecutive):
                                    final_decision = Decision(snap.ts, scope, Tier.A, dds, [],
                                                              acc_conf.tier_a.mode)
                                    target_tier_conf = acc_conf.tier_a
                                    logger.warning(
                                        f"TIER A TRIGGER | {scope} | Window={w} | "
                                        f"DD={dds.get(w, 0):.4f} > {thr}"
                                    )

                        dd_str = ', '.join(f"{w}:{v:.1%}" for w, v in sorted(dds.items()))
                        logger.info(f"[{scope}] Eq=${snap.equity_usdt:.0f} | DD: {dd_str}")
                        print(
                            f"[{datetime.now().strftime('%H:%M')}] {scope} "
                            f"Eq: ${snap.equity_usdt:.0f} | DD: {dd_str}"
                        )

                        if final_decision:
                            logger.critical(
                                f"TRIGGERED {final_decision.tier} — {target_tier_conf.mode}"
                            )
                            engine.execute(final_decision, target_tier_conf)

                except Exception as e:
                    logger.error(f"{scope}: {e}")
                    print(f"[ERROR] {scope}: {e}")
                    breaker.record_failure(scope)

        elapsed = time.time() - cycle_start
        time.sleep(max(1, cfg.poll_seconds - elapsed))


def _run_test_mock():
    """Self-test with mock adapter — verifies stage machine, attribution, and dry-run plan."""
    print("=" * 60)
    print("KILL-SWITCH v3.0 — MOCK TEST")
    print("=" * 60)

    store = SqliteStore(":memory:")
    dd_calc = DrawdownCalculator(store)
    stage_machine = StageMachine(store.conn)
    pos_store = PositionStore(store.conn)
    mock_adapter = MockAdapter(None)
    breaker = CircuitBreaker()

    # Equity scenario: starts at 10000, drops steadily to trigger stage 1
    scenario = [10000, 9900, 9800, 9700, 9600, 9500, 9200, 8800, 8400, 8000]
    mock_adapter.set_scenario(scenario)

    # Mock positions: one profitable short (source of drawdown) and one long (OK)
    scope = Scope(ExchangeId.MOCK, AccountType.FUTURES)
    base_ts = int(time.time())

    mock_positions = [
        PositionSnapshot(
            ts=base_ts, scope=scope, symbol="SOL/USDT:USDT", side="short",
            contracts=10.0, notional_usdt=1200.0,
            entry_price=100.0, mark_price=130.0, liquidation_price=200.0,
            margin_usdt=300.0, leverage=4.0,
            unrealized_pnl_usdt=-300.0,  # short SOL, now at -300
            pnl_pct_on_margin=-1.0, raw={},
        ),
        PositionSnapshot(
            ts=base_ts, scope=scope, symbol="BTC/USDT:USDT", side="long",
            contracts=0.01, notional_usdt=600.0,
            entry_price=55000.0, mark_price=62000.0, liquidation_price=40000.0,
            margin_usdt=150.0, leverage=4.0,
            unrealized_pnl_usdt=70.0,  # long BTC, profitable
            pnl_pct_on_margin=0.47, raw={},
        ),
    ]
    mock_adapter.set_mock_positions(mock_positions)
    mock_adapter.set_mock_orders([
        {"id": "e1", "symbol": "SOL/USDT:USDT", "reduceOnly": False},
    ])

    adapters = {"mock": mock_adapter}
    trading_lock = TradingLock("/tmp/killswitch_test_lock.json")
    engine = ActionEngine(
        store, adapters, True,
        ["USDT", "USDC"],
        breaker,
        stage_machine=stage_machine,
        position_store=pos_store,
        trading_lock=trading_lock,
    )

    stage_1_conf = StageConfig(
        thresholds={"1": 0.04},   # 4% in 1-minute window
        mode="CLOSE_TOP_RISK_CONTRIBUTORS",
        cooldown_min=30,
        confirm_consecutive=2,
        source_mode="AUTO",
        source_threshold=0.65,
        top_n=2,
        close_fraction=0.50,
        full_close_if_liq_distance_below_pct=0.03,
        cancel_entry_orders_before_close=True,
        set_trading_lock=True,
    )
    stage_3_conf = StageConfig(
        thresholds={"1": 0.18},
        mode="CLOSE_ALL_POSITIONS",
        cooldown_min=360,
        confirm_consecutive=1,
        cancel_entry_orders_before_close=True,
        set_trading_lock=True,
    )
    acc_conf = AccountConfig(
        enabled=True, windows=["1"],
        stage_1=stage_1_conf, stage_3=stage_3_conf,
    )

    print(f"\n{'Stage':8} {'TS':10} {'Equity':>10} {'DD 1m':>8} {'Status'}")
    print("-" * 60)

    import logging
    dummy_logger = get_logger("test", level="WARNING")  # suppress info noise

    stage_1_fired = False
    stage_3_fired = False

    for i in range(len(scenario)):
        snap = mock_adapter.fetch_equity(scope)
        store.append_snapshot(snap)
        pos_store.save_many(mock_positions)

        dds = dd_calc.compute(scope, snap.ts, acc_conf.windows)
        dd_val = dds.get("1", 0.0)

        state = stage_machine.get_state(str(scope))
        stage_label = STAGE_NAMES.get(state.current_stage, "?")

        _process_stage_triggers(scope, acc_conf, dds, snap,
                                 engine, stage_machine, dd_calc, dummy_logger)

        new_state = stage_machine.get_state(str(scope))
        if new_state.current_stage == 1 and not stage_1_fired:
            stage_1_fired = True
            print(f"  *** Stage 1 fired at step {i} ***")
        if new_state.current_stage == 3 and not stage_3_fired:
            stage_3_fired = True
            print(f"  *** Stage 3 fired at step {i} ***")

        print(
            f"{'step'+str(i):8} {snap.ts:<10} {snap.equity_usdt:>10.0f} "
            f"{dd_val:>8.2%} [{STAGE_NAMES.get(new_state.current_stage,'?')}]"
        )

    # Verify stage machine behavior
    final_state = stage_machine.get_state(str(scope))
    print(f"\nFinal stage: {final_state.current_stage} ({STAGE_NAMES.get(final_state.current_stage, '?')})")

    # Attribution test
    from risk_attribution import compute_pnl_attribution
    earlier = [
        PositionSnapshot(
            ts=base_ts - 300, scope=scope, symbol="SOL/USDT:USDT", side="short",
            contracts=10.0, notional_usdt=1200.0,
            entry_price=100.0, mark_price=100.0, liquidation_price=200.0,
            margin_usdt=300.0, leverage=4.0, unrealized_pnl_usdt=0.0,
            pnl_pct_on_margin=0.0, raw={},
        ),
    ]
    attr = compute_pnl_attribution(mock_positions[:1], earlier, 0.65)
    assert attr.source == "SHORT", f"Expected SHORT attribution, got {attr.source}"
    print(f"\nAttribution test: source={attr.source} (expected SHORT) — OK")

    # Stage escalation test
    store2 = SqliteStore(":memory:")
    sm2 = StageMachine(store2.conn)
    sm2.record_execution("test", 1, base_ts + 3600, "test", {}, base_ts)
    can2, _ = sm2.can_execute("test", 2, base_ts + 60)
    assert can2, "Stage 2 should be allowed after Stage 1"
    can3, _ = sm2.can_execute("test", 3, base_ts + 60)
    assert can3, "Stage 3 should be allowed after Stage 1"
    can1_blocked, reason1 = sm2.can_execute("test", 1, base_ts + 60)
    assert not can1_blocked, "Stage 1 should be blocked (same-stage cooldown)"
    print(f"Stage escalation test: 2 after 1 = OK, 3 after 1 = OK, 1 re-blocked = OK")

    print(f"\n{'='*60}")
    print(f"MOCK TEST COMPLETE — all assertions passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
