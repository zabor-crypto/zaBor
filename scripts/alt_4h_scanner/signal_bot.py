#!/usr/bin/env python3
"""
ALT 4H Pullback Scanner
=======================
Scans Binance spot altcoin pairs at every 4H candle close.
Detects accumulation breakouts and generates structured limit-entry
signals for buying pullbacks into the first impulse.

Signal logic (layered filters, all backtested):
  1. EMA55 trend filter  — price above EMA55
  2. Ichimoku cloud filter — price above Kumo cloud top
  3. 100-bar breakout     — current candle sets new 100-bar high with confirming close
  4. 40H momentum filter  — close >= 8% above close 10 x 4H candles ago
  5. Bullish-body filter  — breakout candle must be bullish (close > open)
  6. Prev-candle quality  — candle before breakout closes in upper 40% of its range
  7. 1H RSI filter        — rejects overbought signals (1H RSI > 80)
  8. BTC regime adaptation — risk range tightens when BTC is below EMA55
  9. Liquidity filter     — min median daily volume $300k USDT

Entry ladder: 5 limit levels stepping from breakout close down to EMA55/cloud support.
TP levels: R-multiples from entry0 (0.3R, 0.6R, 1.0R, 1.5R, 2.5R).
Stop-loss: EMA55 on 4H close.

Position monitoring (V12): tracks active signals, recalculates TP from avg_entry
when entry-3 is reached, sends Telegram update.

Backtest results (636-trade dataset, 3 months):
  - Full filter stack (N=41): PF 11.81, WR 95.7%, MaxDD -18.7%
  - Relaxed (N=70):           PF 7.06,  WR 90.0%, MaxDD -21.1%

All parameters configurable via environment variables. See README.md.
"""

import os
import sys
import time
import json
import csv
import hashlib
import random
import asyncio
import math
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, ClientError
try:
    import fcntl
except Exception:
    fcntl = None

# ─── COMPATIBILITY LAYER ────────────────────────────────────────────────────
try:
    from binance import AsyncClient
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False
    class AsyncClient: pass
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass

# ─── CONFIGURATION & CONSTANTS ──────────────────────────────────────────────

# Environment
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET  = os.getenv("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID")

# Strategy Params
EMA_PERIOD        = 55
ICHIMOKU_CONV     = 9
ICHIMOKU_BASE     = 30
ICHIMOKU_SPAN_B   = 60
ATR_PERIOD        = 14
KLIMIT            = 100
FETCH_LIMIT       = KLIMIT + ICHIMOKU_SPAN_B + 2

# Signal Tolerances
WICK_TOL_PCT       = 0.003
BREAKOUT_CLOSE_TOL = float(os.getenv("BREAKOUT_CLOSE_TOL", "0.002"))
CLOUD_TOL_PCT      = 0.003

# Liquidity
LIQ_LOOKBACK_DAYS    = int(os.getenv("LIQ_LOOKBACK_DAYS", "7"))
LIQ_SPIKE_MULTIPLIER = float(os.getenv("LIQ_SPIKE_MULTIPLIER", "5.0"))
LIQ_MIN_MEDIAN_USDT  = float(os.getenv("LIQ_MIN_MEDIAN_USDT", "300_000"))

# ATR / Risk Logic (ATR used for ladder spacing only now)
ATR_CAP_PCT     = float(os.getenv("ATR_CAP_PCT", "0.12"))
MIN_STOP_PCT    = float(os.getenv("MIN_STOP_PCT", "0.006"))

# V8 R_MULTS: quicker targets [0.3, 0.6, 1.0, 1.5, 2.5]R
# Backtested: higher hit rate, PF 1.48→1.59 combined with trailing stop
r_str = os.getenv("R_MULTS", "0.3,0.6,1.0,1.5,2.5")
try:
    R_MULTS = [float(x.strip()) for x in r_str.split(',')]
except ValueError:
    R_MULTS = [0.3, 0.6, 1.0, 1.5, 2.5]

# V8 Signal quality filters (all backtested)
# Symbols to exclude: non-altcoin assets that behave differently from crypto alts
_excl_env = os.getenv("EXCLUDE_SYMBOLS", "XAUTUSDT,PAXGUSDT,EURUSDT,EURTUSDT")
EXCLUDE_SYMBOLS: set = {s.strip() for s in _excl_env.split(',') if s.strip()}

# Risk distance filter: (entry0 - SL) / entry0 must be in [MIN_RISK_PCT, MAX_RISK_PCT]
# Backtest: <10% risk → negative expectancy; 10-20% → PF 2.19, DD -70%
MIN_RISK_PCT = float(os.getenv("MIN_RISK_PCT", "0.10"))   # 10%
MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.20"))   # 20%

# BTC regime adaptation: when BTC < EMA55 (downtrend), tighten to [BTC_DOWN_MIN, BTC_DOWN_MAX]
# Backtest: BTC DOWN 12-20% → PF 2.56, WR 82.8% vs BTC UP 10-20% → PF 1.01
BTC_REGIME_ENABLED   = os.getenv("BTC_REGIME_ENABLED", "true").lower() == "true"
# V9: Wider risk range validated by backtest (E4 result)
# UP 10-25% DOWN 12-25% → N=187, PF=2.12, Total=498% (vs N=146 at 10-20%)
BTC_UP_MIN_RISK_PCT  = float(os.getenv("BTC_UP_MIN_RISK_PCT",   "0.10"))  # 10%
BTC_UP_MAX_RISK_PCT  = float(os.getenv("BTC_UP_MAX_RISK_PCT",   "0.25"))  # 25% (was 20%)
BTC_DOWN_MIN_RISK_PCT= float(os.getenv("BTC_DOWN_MIN_RISK_PCT", "0.12"))  # 12%
BTC_DOWN_MAX_RISK_PCT= float(os.getenv("BTC_DOWN_MAX_RISK_PCT", "0.25"))  # 25% (was 20%)
BTC_SYMBOL           = os.getenv("BTC_SYMBOL", "BTCUSDT")

# Momentum filter: current close must be ≥ MOMENTUM_MIN_PCT% above 10-candle-ago close
# Backtest: >=8% → PF 1.59 vs 1.48 baseline; reduces signal count by ~22%, improves WR 72→74%
MOMENTUM_MIN_PCT     = float(os.getenv("MOMENTUM_MIN_PCT", "8.0"))
MOMENTUM_LOOKBACK    = int(os.getenv("MOMENTUM_LOOKBACK", "10"))  # candles (10×4H = 40H)

# V9 Position split: 60/20/10/7/3 (aggressive front-load, backtest: PF 2.22, DD -59.5%)
# With bullish-body filter: PF 4.28, WR 88.9%, DD -32.8%
POSITION_SPLIT_PCT   = [
    int(os.getenv("SPLIT_TP1", "60")),
    int(os.getenv("SPLIT_TP2", "20")),
    int(os.getenv("SPLIT_TP3", "10")),
    int(os.getenv("SPLIT_TP4",  "7")),
    int(os.getenv("SPLIT_TP5",  "3")),
]

# Trailing stop guidance after TP1 (communicated in signal message)
TRAIL_PCT            = float(os.getenv("TRAIL_PCT", "6.0"))  # % below highest 4H high

# V9 Early breakeven: move SL to avg_entry when unrealized gain ≥ EARLY_BE_R × risk
# Backtest: 0.5R trigger → PF 2.39→5.51 (with bullish body), DD -59.5→-21.1%
EARLY_BE_R_MULT      = float(os.getenv("EARLY_BE_R", "0.5"))  # 0 to disable

# V9 Bullish-body filter: breakout candle must have close > open
# Backtest: PF 2.02→4.09 (N=81 vs 187), WR 78%→89%, DD -72%→-30%
# Set to "false" to disable and keep all signals (N=187, PF=2.39)
REQUIRE_BULLISH_BODY = os.getenv("REQUIRE_BULLISH_BODY", "true").lower() == "true"

# V10 Prev-candle close-position filter
# The candle BEFORE the breakout candle must close in the upper X% of its range.
# ClosePos = (close - low) / (high - low); ≥ 0.60 = top 40% of candle range.
# Backtest: PF 5.51→6.87, WR 88.9%→90.0%, N 81→70 (lose 14% signals, +25% PF)
# Set PREV_CLOSE_POS=0 to disable (get full V9 at N=81, PF=5.51)
PREV_CLOSE_POS_MIN   = float(os.getenv("PREV_CLOSE_POS", "0.60"))

# V11 1H RSI overbought filter
# Fetch last 50 1H candles at signal time; reject if RSI(14) > MAX_1H_RSI.
# Backtest (on V10 universe): max_1h_rsi=80 → PF 14.26, DD -8.1%, N=41 of 70
# All V10 signals already have RSI>60; filter removes the overbought ~41%.
# Set MAX_1H_RSI=0 to disable entirely.
MAX_1H_RSI           = float(os.getenv("MAX_1H_RSI", "80"))

# V10 Entry window guidance (hours communicated in signal message, not enforced by bot)
# Backtest: 24H window on V9 universe → WR 93.3%, PF 5.90 (N=60, fewer setups)
# Users should cancel unfilled limit orders after this many hours.
ENTRY_WINDOW_HOURS   = int(os.getenv("ENTRY_WINDOW_HOURS", "24"))

# Network / Scanning
HTTP_TIMEOUT       = 10.0
RETRY_MAX_ATTEMPTS = 3
SCAN_DELAY_SEC     = int(os.getenv("SCAN_DELAY_SEC", "30"))
SCAN_JITTER_MIN    = int(os.getenv("SCAN_JITTER_MIN", "5"))
SCAN_JITTER_MAX    = int(os.getenv("SCAN_JITTER_MAX", "15"))
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "15"))
SIGNAL_COOLDOWN_H  = int(os.getenv("SIGNAL_COOLDOWN_H", "24"))

# V12 Active position monitoring
E3_RECALC_ENABLED    = os.getenv("E3_RECALC_ENABLED", "true").lower() == "true"
_e3m = os.getenv("E3_TP_MULTS", "0.25,0.5,0.8,1.2,2.0")
try:
    E3_TP_MULTS = [float(x.strip()) for x in _e3m.split(',')]
except ValueError:
    E3_TP_MULTS = [0.25, 0.5, 0.8, 1.2, 2.0]
MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "900"))   # 15 min
SIGNAL_EXPIRY_DAYS   = int(os.getenv("SIGNAL_EXPIRY_DAYS", "7"))

# File Paths
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE          = os.path.join(BASE_DIR, "signal_history_v11.json")
ACTIVE_POSITIONS_FILE = os.path.join(BASE_DIR, "active_positions.json")
LOG_FILE              = os.getenv("LOG_FILE", "signal_bot_spot_long_pullback.log")

# ─── LOGGING ────────────────────────────────────────────────────────────────
logger = logging.getLogger("swing_bot_v10")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
formatter.converter = time.gmtime

fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

GLOBAL_SIGNALS_CSV = os.getenv(
    "GLOBAL_SIGNALS_CSV",
    os.path.join(BASE_DIR, "global_signals.csv"),
)
GLOBAL_SCANNER_ID = "signal_bot_long_pullback"


def _extract_signal_tag(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.split()[0]
    return ""


def _extract_signal_symbol(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Pair:"):
            return line.split(":", 1)[1].strip()
    return ""


def _log_global_signal_csv(text: str, exchange: str = "", symbol: str = "") -> None:
    if not text:
        return
    symbol_val = symbol or _extract_signal_symbol(text)
    try:
        os.makedirs(os.path.dirname(GLOBAL_SIGNALS_CSV), exist_ok=True)
        with open(GLOBAL_SIGNALS_CSV, "a+", encoding="utf-8", newline="") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0, os.SEEK_END)
            needs_header = fh.tell() == 0
            writer = csv.writer(fh)
            if needs_header:
                writer.writerow([
                    "ts_utc", "scanner", "script", "exchange", "symbol",
                    "tag", "message_hash", "message_text",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                GLOBAL_SCANNER_ID,
                os.path.basename(__file__),
                exchange,
                symbol_val,
                _extract_signal_tag(text),
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                text.replace("\r", ""),
            ])
            fh.flush()
            os.fsync(fh.fileno())
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Global signal CSV write failed: {e}")

# ─── HELPER: RETRY DECORATOR ───────────────────────────────────────────────
async def retry_api_call(func: Callable, *args, retries=3, **kwargs):
    for i in range(retries):
        try:
            return await func(*args, **kwargs)
        except (ClientError, asyncio.TimeoutError, RuntimeError, 
                BinanceAPIException, BinanceRequestException) as e:
            
            if i == retries - 1: raise e
            wait = (2 ** i) + random.uniform(0, 1)
            await asyncio.sleep(wait)
    return None

# ─── DATA MODELS ────────────────────────────────────────────────────────────
@dataclass
class Signal:
    exchange: str
    symbol: str
    entries: List[float]
    sl: float
    tps: List[float]
    atr: float
    base: float
    risk_pct: float
    btc_regime: str = "unknown"  # 'up' | 'down' | 'unknown'

    def to_telegram_str(self, tick: float) -> str:
        es = ", ".join([fmt(x, tick) for x in self.entries])
        sl_str = fmt(self.sl, tick)
        tp_points = ", ".join(
            f"{fmt(tp, tick)} ({pct}%)"
            for tp, pct in zip(self.tps, POSITION_SPLIT_PCT)
        )

        return (
            "#longalt\n"
            f"Pair: {self.symbol}\n"
            f"Entries: {es}\n"
            f"Targets: {tp_points}\n"
            f"Stop-loss: 4H close below {sl_str}"
        )

# ─── UTIL CLASSEs ───────────────────────────────────────────────────────────
class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, max_tokens: int, period: float):
        self.max_tokens = max_tokens
        self.period     = period
        self.tokens     = float(max_tokens)
        self.last_upd   = time.time()
        self._lock      = asyncio.Lock()

    async def wait(self, cost: int = 1):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_upd
            refill = elapsed * (self.max_tokens / self.period)
            self.tokens = min(self.max_tokens, self.tokens + refill)
            self.last_upd = now
            
            if self.tokens >= cost:
                self.tokens -= cost
                return
            
            deficit = cost - self.tokens
            wait_time = deficit * (self.period / self.max_tokens)
            if wait_time > 0.1: pass 
            await asyncio.sleep(wait_time)
            
            now_after = time.time()
            elapsed = now_after - self.last_upd
            refill = elapsed * (self.max_tokens / self.period)
            self.tokens = min(self.max_tokens, self.tokens + refill)
            self.last_upd = now_after
            self.tokens = max(0.0, self.tokens - cost)

class SignalHistory:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        self.hashes = deque(maxlen=2000)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    raw = json.load(f)
                    self.data = raw.get("cooldowns", {})
                    h_list = raw.get("hashes", [])
                    self.hashes = deque(h_list[-2000:], maxlen=2000)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({
                    "cooldowns": self.data,
                    "hashes": list(self.hashes)
                }, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def prune(self):
        now = time.time()
        cutoff = now - (7 * 86400)
        to_del = [k for k, ts in self.data.items() if ts < cutoff]
        for k in to_del: del self.data[k]
        if to_del: logger.info(f"Pruned {len(to_del)} old cooldown keys.")

    def check_cooldown(self, exchange: str, symbol: str) -> bool:
        key = f"{exchange}:{symbol}"
        last_ts = self.data.get(key, 0)
        return (time.time() - last_ts) < (SIGNAL_COOLDOWN_H * 3600)

    def mark_sent_internal(self, exchange: str, symbol: str, msg_text: str):
        key = f"{exchange}:{symbol}"
        self.data[key] = time.time()
        h = hashlib.sha256(msg_text.encode()).hexdigest()
        if h not in self.hashes:
            self.hashes.append(h)

    def is_duplicate_msg(self, msg_text: str) -> bool:
        h = hashlib.sha256(msg_text.encode()).hexdigest()
        return h in self.hashes

# ─── ACTIVE POSITIONS TRACKER ────────────────────────────────────────────────
class ActivePositionsTracker:
    """Persists active signals for the monitoring loop.

    Each position is a dict:
      sig_id, exchange, symbol, entries, tps, sl, tick, sent_at,
      e3_recalc_done, expired
    """
    def __init__(self, path: str):
        self.path = path
        self.positions: List[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self.positions = json.load(f).get("positions", [])
            except Exception as e:
                logger.error(f"ActivePositions load error: {e}")
                self.positions = []

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({"positions": self.positions}, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"ActivePositions save error: {e}")

    def add(self, exchange: str, symbol: str, entries: List[float],
            tps: List[float], sl: float, tick: float):
        sig_id = hashlib.sha256(f"{exchange}:{symbol}:{entries[0]}:{sl}".encode()).hexdigest()[:16]
        if any(p['sig_id'] == sig_id for p in self.positions):
            return
        self.positions.append({
            "sig_id": sig_id,
            "exchange": exchange,
            "symbol": symbol,
            "entries": entries,
            "tps": tps,
            "sl": sl,
            "tick": tick,
            "sent_at": time.time(),
            "e3_recalc_done": False,
            "expired": False,
        })
        self.save()

    def active(self) -> List[dict]:
        return [p for p in self.positions if not p['expired'] and not p['e3_recalc_done']]

    def prune(self):
        cutoff = time.time() - SIGNAL_EXPIRY_DAYS * 86400
        before = len(self.positions)
        for p in self.positions:
            if p['sent_at'] < cutoff:
                p['expired'] = True
        # Keep only last 200 entries to avoid unbounded growth
        self.positions = self.positions[-200:]
        pruned = before - len([p for p in self.positions if not p['expired']])
        if pruned > 0:
            logger.info(f"ActivePositions pruned {pruned} old entries.")

# ─── MATH & ROUNDING HELPERS ────────────────────────────────────────────────
def floor_step(val: float, step: float) -> float:
    if step <= 0: return val
    return math.floor(val / step + 1e-10) * step

def ceil_step(val: float, step: float) -> float:
    if step <= 0: return val
    return math.ceil(val / step - 1e-10) * step

def fmt(val: float, step: float) -> str:
    if val < 0.01:
        # Low-sat handling (e.g. PEPE)
        # Using 8 decimals for precision, stripped of trailing stuff
        return f"{val:.8f}".rstrip('0').rstrip('.')
    if step <= 0: return f"{val:.6f}"
    prec = max(0, int(-math.log10(step))) if 0 < step < 1 else 2
    return f"{val:.{prec}f}"

# ─── EXCHANGE ADAPTER INTERFACE ─────────────────────────────────────────────
class ExchangeAdapter(ABC):
    def __init__(self, name: str, rate_limit_tokens: int, concurrent_limit: int):
        self.name = name
        self.symbol_filters: Dict[str, Dict] = {} 
        self.rate_limiter = RateLimiter(rate_limit_tokens, 60) 
        self.sem = asyncio.Semaphore(concurrent_limit)

    @abstractmethod
    async def connect(self): pass
    @abstractmethod
    async def get_server_time(self) -> int: pass
    @abstractmethod
    async def get_exchange_info(self) -> List[str]: pass
    @abstractmethod
    async def get_tickers_24h(self) -> Dict[str, float]: pass
    @abstractmethod
    async def get_daily_quote_volumes(self, symbols: List[str], limit_days: int) -> Dict[str, List[float]]: pass
    @abstractmethod
    async def get_klines_4h(self, symbol: str, limit: int) -> pd.DataFrame: pass
    @abstractmethod
    async def get_klines_1h(self, symbol: str, limit: int) -> pd.DataFrame: pass
    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[float]: pass
    @abstractmethod
    async def close(self): pass

    def get_filter(self, symbol: str) -> Dict[str, float]:
        f = self.symbol_filters.get(symbol, {"tick": 0.0, "step": 0.0, "minNotional": 0.0})
        if f["tick"] <= 0: f["tick"] = 1e-6
        if f["step"] <= 0: f["step"] = 1e-6
        return f

# ─── BINANCE ADAPTER ────────────────────────────────────────────────────────
class BinanceAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__("BINANCE", rate_limit_tokens=1200, concurrent_limit=20)
        self.client: Optional[AsyncClient] = None

    async def connect(self):
        if not HAS_BINANCE: raise ImportError("python-binance missing")
        if BINANCE_API_KEY and BINANCE_API_SECRET:
            self.client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
        else:
            self.client = await AsyncClient.create()

    async def get_server_time(self) -> int:
        res = await retry_api_call(self.client.get_server_time)
        return res['serverTime']

    async def get_exchange_info(self) -> List[str]:
        await self.rate_limiter.wait(10)
        info = await retry_api_call(self.client.get_exchange_info)
        valid = []
        for s in info["symbols"]:
            if s["status"] != "TRADING": continue
            if s["quoteAsset"] not in ["USDT", "FDUSD"]: continue
            if any(x in s["symbol"] for x in ["LEV", "BEAR", "BULL", "UP", "DOWN"]): continue
            
            f_map = {"tick": 0.0, "step": 0.0, "minNotional": 0.0}
            for f in s["filters"]:
                t = f["filterType"]
                if t == "PRICE_FILTER": f_map["tick"] = float(f["tickSize"])
                if t == "LOT_SIZE": f_map["step"] = float(f["stepSize"])
                if t in ["MIN_NOTIONAL", "NOTIONAL"]: 
                    f_map["minNotional"] = float(f.get("minNotional", f.get("notional", 0)))
            self.symbol_filters[s["symbol"]] = f_map
            valid.append(s["symbol"])
        return valid

    async def get_tickers_24h(self) -> Dict[str, float]:
        await self.rate_limiter.wait(5)
        raw = await retry_api_call(self.client.get_ticker)
        return {x['symbol']: float(x['quoteVolume']) for x in raw}

    async def get_daily_quote_volumes(self, symbols: List[str], limit_days: int) -> Dict[str, List[float]]:
        results = {}
        async def fetch(s):
            await self.rate_limiter.wait(1)
            async with self.sem:
                try:
                    k = await retry_api_call(self.client.get_klines, symbol=s, interval=AsyncClient.KLINE_INTERVAL_1DAY, limit=limit_days)
                    return s, [float(x[7]) for x in k]
                except Exception:
                    return s, []

        chunk_size = 20
        for i in range(0, len(symbols), chunk_size):
            tasks = [fetch(s) for s in symbols[i:i+chunk_size]]
            batch_res = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_res:
                if isinstance(res, tuple):
                    results[res[0]] = res[1]
        return results

    async def get_klines_4h(self, symbol: str, limit: int) -> pd.DataFrame:
        await self.rate_limiter.wait(1)
        async with self.sem:
            raw = await retry_api_call(self.client.get_klines, symbol=symbol, interval=AsyncClient.KLINE_INTERVAL_4HOUR, limit=limit)
        df = pd.DataFrame(raw, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_volume", "Trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        return df

    async def get_klines_1h(self, symbol: str, limit: int) -> pd.DataFrame:
        await self.rate_limiter.wait(1)
        async with self.sem:
            raw = await retry_api_call(self.client.get_klines, symbol=symbol, interval=AsyncClient.KLINE_INTERVAL_1HOUR, limit=limit)
        df = pd.DataFrame(raw, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_volume", "Trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        return df

    async def get_price(self, symbol: str) -> Optional[float]:
        try:
            await self.rate_limiter.wait(1)
            async with self.sem:
                r = await retry_api_call(self.client.get_symbol_ticker, symbol=symbol)
            return float(r['price'])
        except Exception:
            return None

    async def close(self):
        if self.client: await self.client.close_connection()

# ─── BITGET ADAPTER ─────────────────────────────────────────────────────────
class BitgetAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__("BITGET", rate_limit_tokens=300, concurrent_limit=8)
        self.base_url = "https://api.bitget.com"
        self.session: Optional[ClientSession] = None

    async def connect(self):
        if self.session is None or self.session.closed:
            self.session = ClientSession(timeout=ClientTimeout(total=HTTP_TIMEOUT))

    async def _req(self, endpoint, params=None):
        if not self.session: await self.connect()
        url = self.base_url + endpoint
        
        async def do_req():
            async with self.session.get(url, params=params) as resp:
                if resp.status == 429:
                    raise RuntimeError("RateLimit 429")
                if resp.status >= 500:
                    raise RuntimeError(f"Server Error {resp.status}")
                if resp.status != 200:
                    t = await resp.text()
                    raise RuntimeError(f"Bitget {resp.status} {t}")
                return await resp.json()

        await self.rate_limiter.wait(1)
        async with self.sem:
            return await retry_api_call(do_req, retries=3)

    async def get_server_time(self) -> int:
        d = await self._req("/api/v2/public/time")
        return int(d['data']['serverTime'])

    async def get_exchange_info(self) -> List[str]:
        d = await self._req("/api/v2/spot/public/symbols")
        valid = []
        for s in d['data']:
            if s['status'] != 'online': continue
            if s['quoteCoin'] != 'USDT': continue
            
            sym = s['symbol']
            tick = s.get('pricePrecision', '0.000001')
            step = s.get('quantityPrecision', '0.000001')
            
            f_map = {"tick": 0.0, "step": 0.0, "minNotional": 5.0}
            def parse_prec(p):
                if '.' in str(p): return float(p)
                return 1.0 / (10 ** float(p))

            f_map['tick'] = parse_prec(tick)
            f_map['step'] = parse_prec(step)
            f_map['minNotional'] = float(s.get('minTradeAmount', 5.0)) 
            self.symbol_filters[sym] = f_map
            valid.append(sym)
        return valid

    async def get_tickers_24h(self) -> Dict[str, float]:
        d = await self._req("/api/v2/spot/market/tickers")
        return {x['symbol']: float(x.get('usdtVolume', 0)) for x in d['data']}

    async def get_daily_quote_volumes(self, symbols: List[str], limit_days: int) -> Dict[str, List[float]]:
        results = {}
        async def fetch(s):
            try:
                d = await self._req("/api/v2/spot/market/candles", 
                                  {"symbol": s, "granularity": "1day", "limit": limit_days})
                if not d.get('data'): return s, []
                rows = d['data']
                rows.sort(key=lambda x: int(x[0]))
                return s, [float(x[6]) for x in rows]
            except Exception:
                return s, []

        chunk_size = 10 
        for i in range(0, len(symbols), chunk_size):
            tasks = [fetch(s) for s in symbols[i:i+chunk_size]]
            batch_res = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_res:
                if isinstance(res, tuple):
                    results[res[0]] = res[1]
        return results

    async def get_klines_4h(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            d = await self._req("/api/v2/spot/market/candles",
                              {"symbol": symbol, "granularity": "4H", "limit": limit})
        except Exception:
            return pd.DataFrame()
        if not d.get('data'):
            logger.warning(f"[BITGET] Empty candles 4h: {symbol}")
            return pd.DataFrame()
        rows = d['data']
        rows.sort(key=lambda x: int(x[0]))
        data = []
        for r in rows:
            ts = int(r[0])
            data.append([ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                         ts + 14400000 - 1, float(r[6]), 0, 0, 0, 0])
        df = pd.DataFrame(data, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_volume", "Trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        return df

    async def get_klines_1h(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            d = await self._req("/api/v2/spot/market/candles",
                              {"symbol": symbol, "granularity": "1H", "limit": limit})
        except Exception:
            return pd.DataFrame()
        if not d.get('data'):
            return pd.DataFrame()
        rows = d['data']
        rows.sort(key=lambda x: int(x[0]))
        data = []
        for r in rows:
            ts = int(r[0])
            data.append([ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                         ts + 3600000 - 1, float(r[6]), 0, 0, 0, 0])
        df = pd.DataFrame(data, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_volume", "Trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        return df

    async def get_price(self, symbol: str) -> Optional[float]:
        try:
            d = await self._req("/api/v2/spot/market/tickers", {"symbol": symbol})
            if d.get('data'):
                return float(d['data'][0]['lastPr'])
        except Exception:
            pass
        return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

# ─── LOGIC & MATH ───────────────────────────────────────────────────────────
def calculate_atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_rsi_14(series: pd.Series, period: int = 14) -> float:
    """Return most recent RSI(period). Returns NaN if insufficient data."""
    clean = series.dropna()
    if len(clean) < period + 1:
        return float("nan")
    delta  = clean.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_l = float(avg_l.iloc[-1])
    if last_l == 0:
        return 100.0
    rs = float(avg_g.iloc[-1]) / last_l
    return float(100 - 100 / (1 + rs))


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["EMA"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    def donchian(h, l, n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    tenkan = donchian(df["High"], df["Low"], ICHIMOKU_CONV)
    kijun  = donchian(df["High"], df["Low"], ICHIMOKU_BASE)
    span_a = ((tenkan + kijun) / 2).shift(ICHIMOKU_BASE)
    h60 = df["High"].rolling(ICHIMOKU_SPAN_B).max()
    l60 = df["Low"].rolling(ICHIMOKU_SPAN_B).min()
    span_b = ((h60 + l60) / 2).shift(ICHIMOKU_BASE)

    df["Cloud_Top"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
    df["ATR"] = calculate_atr_wilder(df, ATR_PERIOD)
    # Momentum: % change over MOMENTUM_LOOKBACK candles (e.g. 10 × 4H = 40H)
    df["Momentum"] = df["Close"].pct_change(MOMENTUM_LOOKBACK) * 100
    # V10 ClosePos: (close - low) / (high - low) — where in the candle did price close
    rng = df["High"] - df["Low"]
    df["ClosePos"] = (df["Close"] - df["Low"]) / rng.replace(0, float("nan"))
    return df

def get_robust_ladder(close: float, base: float, tick: float, atr: float) -> List[float]:
    if close <= base or tick <= 0: return []
    span = close - base
    raw = [base + span * r for r in [0.90, 0.75, 0.55, 0.35, 0.15]]
    raw = [floor_step(x, tick) for x in raw]
    
    # Cap gap to span/6 to avoid rejection
    min_gap = max(0.15 * atr, tick * 5)
    min_gap = min(min_gap, span / 6.0) 
    
    out = []
    for x in raw:
        if not out:
            out.append(x)
        else:
            prev = out[-1]
            candidate = min(x, prev - min_gap)
            candidate = floor_step(candidate, tick)
            if candidate > base:
                out.append(candidate)
            else:
                break 
    return out if len(out) == 5 else []

# ─── BTC REGIME DETECTION ───────────────────────────────────────────────────
async def get_btc_regime(adapters: List) -> str:
    """
    Fetch BTC 4H data and return 'up' if close > EMA55, else 'down'.
    Backtest insight: BTC DOWN regime → PF 2.56, WR 82.8% (12-20% risk range).
                      BTC UP regime   → PF 1.33 (12-20%), PF 1.01 (10-20%).
    Returns 'unknown' on any error (falls back to standard MIN/MAX_RISK_PCT).
    """
    if not BTC_REGIME_ENABLED:
        return "unknown"
    for ad in adapters:
        try:
            df = await ad.get_klines_4h(BTC_SYMBOL, KLIMIT + 10)
            if df.empty:
                continue
            for c in ["Close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()
            last = df.iloc[-1]
            regime = "up" if last["Close"] > last["EMA55"] else "down"
            logger.info(f"BTC regime: {regime} (close={last['Close']:.1f}, EMA55={last['EMA55']:.1f})")
            return regime
        except Exception as e:
            logger.warning(f"BTC regime fetch error on {ad.name}: {e}")
            continue
    logger.warning("BTC regime: could not determine, defaulting to 'unknown'")
    return "unknown"


# ─── SIGNAL ENGINE ──────────────────────────────────────────────────────────
class SignalEngine:
    def __init__(self, adapter: ExchangeAdapter, btc_regime: str = "unknown"):
        self.adapter = adapter
        self.btc_regime = btc_regime
        # Per-cycle rejection counters for monitoring
        self.rej = {
            "excluded_symbol": 0,
            "no_data": 0,
            "ema_cloud": 0,
            "breakout": 0,
            "momentum": 0,
            "prev_candle": 0,
            "rsi_overbought": 0,
            "ladder": 0,
            "risk_range": 0,
            "passed": 0,
        }

    async def analyze(self, symbol: str, cutoff_ms: int) -> Optional[Signal]:
        # V8: exclude non-altcoin / non-crypto assets
        if symbol in EXCLUDE_SYMBOLS:
            self.rej["excluded_symbol"] += 1
            return None

        df = await self.adapter.get_klines_4h(symbol, FETCH_LIMIT)
        if df.empty:
            self.rej["no_data"] += 1
            return None

        df = df[df["Close_time"] < cutoff_ms].copy()
        if len(df) < KLIMIT:
            self.rej["no_data"] += 1
            return None

        df = calculate_indicators(df)
        last = df.iloc[-1]

        c   = last["Close"]
        ema = last["EMA"]
        cloud = last["Cloud_Top"]
        atr = last["ATR"]

        # Trend / cloud filters
        if c <= ema or last["Low"] < ema * (1.0 - WICK_TOL_PCT):
            self.rej["ema_cloud"] += 1
            return None
        if c < cloud * (1.0 - CLOUD_TOL_PCT):
            self.rej["ema_cloud"] += 1
            return None

        # Breakout filter: current candle must set a new 100-bar high with confirming close
        res_high = df["High"].iloc[-KLIMIT:-1].max()
        if last["High"] <= res_high:
            self.rej["breakout"] += 1
            return None
        if c < res_high * (1.0 - BREAKOUT_CLOSE_TOL):
            self.rej["breakout"] += 1
            return None

        # V8 Momentum filter: close ≥ MOMENTUM_MIN_PCT% above close 10 candles ago
        # Backtest result: >=8% → PF 1.59 vs 1.48 baseline, WR 74.6% vs 72.7%
        # NaN-safe: skip filter if not enough history (< MOMENTUM_LOOKBACK candles)
        momentum_val = last["Momentum"] if "Momentum" in last.index else float("nan")
        if not (isinstance(momentum_val, float) and math.isnan(momentum_val)):
            try:
                if float(momentum_val) < MOMENTUM_MIN_PCT:
                    self.rej["momentum"] += 1
                    return None
            except (TypeError, ValueError):
                pass  # skip filter on unparseable value

        # V9 Bullish-body filter: breakout candle close > open
        # Backtest: PF 2.02→4.09, WR 78%→89%, DD -72%→-30% (N=81 vs 146)
        # Disabled via env REQUIRE_BULLISH_BODY=false for higher signal count (N=187, PF=2.39)
        if REQUIRE_BULLISH_BODY and c <= last["Open"]:
            self.rej["breakout"] += 1
            return None

        # V10 Prev-candle close-position filter
        # The candle immediately before the breakout must close in the upper 40% of its range.
        # Backtest: PF 5.51→6.87, WR 88.9%→90.0%, N 81→70 (at threshold 0.60)
        # Disable with PREV_CLOSE_POS=0
        if PREV_CLOSE_POS_MIN > 0 and len(df) >= 2:
            prev = df.iloc[-2]
            prev_cp = prev["ClosePos"] if "ClosePos" in prev.index else float("nan")
            if not (isinstance(prev_cp, float) and math.isnan(prev_cp)):
                try:
                    if float(prev_cp) < PREV_CLOSE_POS_MIN:
                        self.rej["prev_candle"] += 1
                        return None
                except (TypeError, ValueError):
                    pass  # skip filter on unparseable value

        # V11 1H RSI overbought filter
        # Backtest: max_1h_rsi=80 → PF 7.06→14.26, DD -21.1%→-8.1%, N 70→41
        # All V10 signals have 1H RSI > 60 already; RSI > 80 = overbought on 1H.
        # On fetch failure: log warning and continue (don't reject on API error).
        if MAX_1H_RSI > 0:
            try:
                df_1h = await self.adapter.get_klines_1h(symbol, 50)
                if not df_1h.empty:
                    df_1h["Close"] = pd.to_numeric(df_1h["Close"], errors="coerce")
                    rsi_1h = calculate_rsi_14(df_1h["Close"])
                    if not math.isnan(rsi_1h) and rsi_1h > MAX_1H_RSI:
                        self.rej["rsi_overbought"] += 1
                        return None
            except Exception as e:
                logger.warning(f"[{symbol}] 1H RSI filter skipped: {e}")

        base = max(ema, cloud)
        f_info = self.adapter.get_filter(symbol)
        tick = f_info["tick"]

        entries = get_robust_ladder(c, base, tick, atr)
        if not entries:
            self.rej["ladder"] += 1
            return None

        entry0 = entries[0]

        # SL = EMA55 floored to tick
        sl_final = floor_step(ema, tick)
        if sl_final >= entry0:
            self.rej["ladder"] += 1
            return None

        risk = entry0 - sl_final
        if risk <= 0:
            self.rej["ladder"] += 1
            return None

        # V8 Regime-adaptive risk filter
        # Backtest: BTC DOWN 12-20% → PF 2.56, WR 82.8%; BTC UP 10-20% → PF 1.33
        risk_pct = risk / entry0
        if self.btc_regime == "down":
            min_rp, max_rp = BTC_DOWN_MIN_RISK_PCT, BTC_DOWN_MAX_RISK_PCT
        elif self.btc_regime == "up":
            min_rp, max_rp = BTC_UP_MIN_RISK_PCT, BTC_UP_MAX_RISK_PCT
        else:
            min_rp, max_rp = MIN_RISK_PCT, MAX_RISK_PCT

        if not (min_rp <= risk_pct <= max_rp):
            self.rej["risk_range"] += 1
            return None

        tps = []
        prev_tp = entry0
        min_gap = max(0.15 * min(atr, c * 0.1), tick * 5)
        for m in R_MULTS:
            tp = ceil_step(entry0 + risk * m, tick)
            tp = max(tp, prev_tp + min_gap)
            tp = ceil_step(tp, tick)
            tps.append(tp)
            prev_tp = tp

        self.rej["passed"] += 1
        early_be_price = fmt(entry0 + risk * EARLY_BE_R_MULT, tick) if EARLY_BE_R_MULT > 0 else "off"
        logger.info(
            f"[SIGNAL] {symbol} regime={self.btc_regime} "
            f"risk={risk_pct*100:.1f}% momentum={momentum_val:.1f}% "
            f"entry0={fmt(entry0, tick)} sl={fmt(sl_final, tick)} "
            f"early_be_trigger={early_be_price} "
            f"tp1={fmt(tps[0], tick)} tp5={fmt(tps[4], tick)}"
        )
        return Signal(
            exchange=self.adapter.name,
            symbol=symbol,
            entries=entries,
            sl=sl_final,
            tps=tps,
            atr=atr,
            base=base,
            risk_pct=risk_pct,
            btc_regime=self.btc_regime,
        )

# ─── TELEGRAM HELPER ────────────────────────────────────────────────────────
async def send_telegram_message(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        logger.warning("Telegram config missing — cannot send message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        async def _send():
            async with s.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}) as resp:
                if resp.status == 429:
                    try:
                        d = await resp.json()
                        wait = d.get('parameters', {}).get('retry_after', 1)
                        await asyncio.sleep(wait)
                    except Exception:
                        pass
                    raise RuntimeError("TG 429")
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"TG {resp.status}: {body}")
        try:
            await retry_api_call(_send, retries=3)
        except Exception as e:
            logger.error(f"send_telegram_message failed: {e}")

# ─── MONITOR LOOP ────────────────────────────────────────────────────────────
async def monitor_loop(adapters: List[ExchangeAdapter], tracker: ActivePositionsTracker):
    """Runs every MONITOR_INTERVAL_SEC. Checks active signals for entry-3 fill.
    When price ≤ entries[2], recalculates TPs from avg_entry and notifies via Telegram.
    """
    logger.info(f"Monitor loop started (interval={MONITOR_INTERVAL_SEC}s, e3_recalc={'on' if E3_RECALC_ENABLED else 'off'}).")
    while True:
        await asyncio.sleep(MONITOR_INTERVAL_SEC)
        if not E3_RECALC_ENABLED:
            continue
        try:
            tracker.prune()
            positions = tracker.active()
            if not positions:
                continue

            logger.info(f"[MONITOR] Checking {len(positions)} active position(s).")
            for pos in positions:
                symbol   = pos['symbol']
                exchange = pos['exchange']
                entries  = pos['entries']
                sl       = pos['sl']
                tick     = pos['tick']

                # Pick adapter matching exchange (fall back to first available)
                adapter = next(
                    (a for a in adapters if a.name == exchange),
                    adapters[0] if adapters else None
                )
                if adapter is None:
                    continue

                try:
                    price = await adapter.get_price(symbol)
                except Exception as e:
                    logger.warning(f"[MONITOR] {symbol} price fetch error: {e}")
                    continue

                if price is None:
                    continue

                # Expire if below SL
                if price < sl:
                    logger.info(f"[MONITOR] {symbol} price {price} < SL {sl} — marking expired.")
                    pos['expired'] = True
                    continue

                # Check entry-3 trigger (price touched or passed entry3 level)
                if len(entries) >= 3 and price <= entries[2]:
                    avg_entry = sum(entries[:3]) / 3
                    new_risk  = avg_entry - sl
                    if new_risk <= 0:
                        logger.warning(f"[MONITOR] {symbol} new_risk ≤ 0, skipping recalc.")
                        pos['e3_recalc_done'] = True
                        continue

                    new_tps = [avg_entry + new_risk * m for m in E3_TP_MULTS]
                    tp_str  = ", ".join(
                        f"{fmt(tp, tick)} ({pct}%)"
                        for tp, pct in zip(new_tps, POSITION_SPLIT_PCT)
                    )
                    avg_str = fmt(avg_entry, tick)
                    sl_str  = fmt(sl, tick)

                    msg = (
                        f"#longalt update\n"
                        f"Pair: {symbol}\n"
                        f"Entry-3 reached — TP recalculated from avg entry\n"
                        f"Avg entry: {avg_str}\n"
                        f"Targets: {tp_str}\n"
                        f"Stop-loss: unchanged (4H close below {sl_str})"
                    )
                    logger.info(f"[MONITOR] {symbol} e3 triggered — sending recalc notification.")
                    await send_telegram_message(msg)
                    _log_global_signal_csv(msg, exchange=exchange, symbol=symbol)
                    pos['e3_recalc_done'] = True

            tracker.save()
        except Exception as e:
            logger.error(f"[MONITOR] Loop error: {e}", exc_info=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────
async def run_scan_cycle(adapters: List[ExchangeAdapter], history: SignalHistory,
                         cutoff_ms: int, tracker: Optional["ActivePositionsTracker"] = None):
    stats = {
        "scanned": 0, "liq_pass": 0, "signals": 0, "errors": 0,
        "daily_missing": 0, "fail_open_pass": 0
    }

    # V8: Detect BTC regime once per cycle before scanning all symbols
    btc_regime = await get_btc_regime(adapters)
    logger.info(f"Scan cycle BTC regime: {btc_regime}")

    async def scan_exchange(ad: ExchangeAdapter) -> List[tuple]:
        logger.info(f"[{ad.name}] info/tickers...")
        try:
            syms_all = await ad.get_exchange_info()
            tickers = await ad.get_tickers_24h()
        except Exception as e:
            logger.error(f"[{ad.name}] Init fail: {e}")
            return []

        candidates_lvl1 = []
        for s in syms_all:
            if tickers.get(s, 0) > LIQ_MIN_MEDIAN_USDT * 0.5: 
                candidates_lvl1.append(s)
        
        logger.info(f"[{ad.name}] Ticker filter: {len(syms_all)} -> {len(candidates_lvl1)}")
        
        logger.info(f"[{ad.name}] Verifying median liquidity...")
        daily_vols = await ad.get_daily_quote_volumes(candidates_lvl1, LIQ_LOOKBACK_DAYS + 1)
        
        final_candidates = []
        
        for s in candidates_lvl1:
            vols = daily_vols.get(s, [])
            if not vols or len(vols) < 3:
                stats["daily_missing"] += 1
                if tickers.get(s, 0) > LIQ_MIN_MEDIAN_USDT * 2:
                    final_candidates.append(s)
                    stats["fail_open_pass"] += 1
                continue
            
            if len(vols) >= LIQ_LOOKBACK_DAYS:
                recent = vols[-(LIQ_LOOKBACK_DAYS+1):-1]
                med = np.median(recent)
                if med >= LIQ_MIN_MEDIAN_USDT:
                    last_day_vol = vols[-2]
                    if last_day_vol < med * LIQ_SPIKE_MULTIPLIER:
                        final_candidates.append(s)

        stats["liq_pass"] += len(final_candidates)
        logger.info(f"[{ad.name}] Candidates: {len(final_candidates)}")

        # V8: pass BTC regime to engine for adaptive risk filtering
        eng = SignalEngine(ad, btc_regime=btc_regime)
        raw_results = []

        for i in range(0, len(final_candidates), BATCH_SIZE):
            batch = final_candidates[i:i+BATCH_SIZE]
            res = await asyncio.gather(*(eng.analyze(s, cutoff_ms) for s in batch), return_exceptions=True)
            for r in res:
                stats["scanned"] += 1
                if isinstance(r, Signal):
                    stats["signals"] += 1
                    try:
                        tick = ad.get_filter(r.symbol)["tick"]
                        raw_results.append((r, tick))
                    except Exception as e:
                        logger.error(f"Signal err: {e}")
                        stats["errors"] += 1
                elif isinstance(r, Exception):
                    logger.error(f"[{ad.name}] analyze exception: {r}")
                    stats["errors"] += 1

        # Log V8 filter breakdown for monitoring
        logger.info(
            f"[{ad.name}] Filter stats (regime={btc_regime}): "
            f"excluded={eng.rej['excluded_symbol']} no_data={eng.rej['no_data']} "
            f"ema_cloud={eng.rej['ema_cloud']} breakout={eng.rej['breakout']} "
            f"momentum={eng.rej['momentum']} prev_candle={eng.rej['prev_candle']} "
            f"rsi_overbought={eng.rej['rsi_overbought']} "
            f"ladder={eng.rej['ladder']} risk_range={eng.rej['risk_range']} "
            f"passed={eng.rej['passed']}"
        )
        return raw_results

    # 1. Gather all raw signals from all exchanges
    results_list = await asyncio.gather(*(scan_exchange(a) for a in adapters))
    
    flat_signals = []
    for r in results_list: flat_signals.extend(r)
    
    # 2. Deduplicate: Prioritize Binance
    # Group by Symbol
    grouped = {}
    for item in flat_signals:
        sig, tick = item
        if sig.symbol not in grouped: grouped[sig.symbol] = []
        grouped[sig.symbol].append(item)
    
    final_items = []
    for sym, items in grouped.items():
        if len(items) == 1:
            final_items.append(items[0])
        else:
            # If multiple exchange signals, pick BINANCE if available
            binance_item = next((x for x in items if x[0].exchange == "BINANCE"), None)
            if binance_item:
                final_items.append(binance_item)
            else:
                final_items.append(items[0])
    
    # 3. Process Cooldown & History & Format
    all_msgs = []
    for sig, tick in final_items:
        try:
            if history.check_cooldown(sig.exchange, sig.symbol): continue
            
            txt = sig.to_telegram_str(tick)
            if history.is_duplicate_msg(txt): continue

            audit_payload = {
                "scanner": GLOBAL_SCANNER_ID,
                "tag": "#longalt",
                "exchange": sig.exchange,
                "symbol": sig.symbol,
                "direction": "LONG",
                "entry_prices": sig.entries,
                "tp_prices": sig.tps,
                "sl_price": sig.sl,
                "risk_pct": sig.risk_pct,
                "btc_regime": sig.btc_regime,
                "message_hash": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16],
            }
            logger.info(f"[SIGNAL_AUDIT] {json.dumps(audit_payload, ensure_ascii=False)}")
            
            history.mark_sent_internal(sig.exchange, sig.symbol, txt)
            all_msgs.append((txt, sig, tick))
        except Exception as e:
            logger.error(f"Post-proc err {sig.symbol}: {e}")

    history.save()
    history.prune()

    if all_msgs:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            async with ClientSession(timeout=ClientTimeout(total=30)) as s:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

                async def send_one(msg):
                    async with s.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}) as resp:
                        if resp.status == 429:
                             try:
                                 d = await resp.json()
                                 wait = d.get('parameters', {}).get('retry_after', 1)
                                 logger.warning(f"TG 429, wait {wait}s")
                                 await asyncio.sleep(wait)
                             except Exception:
                                 pass
                             raise RuntimeError("TG 429")
                        if resp.status != 200:
                             body = await resp.text()
                             logger.error(f"TG Error ({resp.status}): {body}")
                             raise RuntimeError(f"TG {resp.status}")

                for msg, sig, tick in all_msgs:
                    try:
                        await retry_api_call(send_one, msg, retries=3)
                        _log_global_signal_csv(msg, exchange=sig.exchange, symbol=sig.symbol)
                        if tracker and E3_RECALC_ENABLED:
                            tracker.add(sig.exchange, sig.symbol, sig.entries, sig.tps, sig.sl, tick)
                    except Exception as e:
                        logger.error(f"TG Send Fail: {e}")
        else:
            logger.warning("Telegram Config Missing.")
            
    logger.info(
        f"Cycle Done. regime={btc_regime} signals={len(all_msgs)} "
        f"raw={len(flat_signals)} stats={stats}"
    )

async def main():
    if not TELEGRAM_CHAT_ID:
         logger.error("TELEGRAM_CHAT_ID missing.")
         return

    adapters = []
    if HAS_BINANCE:
        b = BinanceAdapter()
        await b.connect()
        adapters.append(b)
    
    bg = BitgetAdapter()
    await bg.connect()
    adapters.append(bg)
    
    hist    = SignalHistory(HISTORY_FILE)
    tracker = ActivePositionsTracker(ACTIVE_POSITIONS_FILE)
    logger.info("Bot V12 Started. Aligning to 4H grid...")

    async def scan_loop():
        while True:
            try:
                server_ms = 0
                for ad in adapters:
                    try:
                        server_ms = await ad.get_server_time()
                        break
                    except Exception: continue

                if server_ms == 0:
                    logger.error("Time sync failed. Sleep 60s.")
                    await asyncio.sleep(60)
                    continue

                curr_4h_start = (server_ms // 14400000) * 14400000
                next_4h_end   = curr_4h_start + 14400000

                wait_s = (next_4h_end - server_ms) / 1000 + SCAN_DELAY_SEC
                wait_s += random.uniform(SCAN_JITTER_MIN, SCAN_JITTER_MAX)
                if wait_s < 5: wait_s = 5

                logger.info(f"Sleeping {wait_s/60:.1f}m ...")
                await asyncio.sleep(wait_s)

                cutoff = next_4h_end
                logger.info(f"RUNNING SCAN cycle for close: {datetime.fromtimestamp(cutoff/1000, tz=timezone.utc).isoformat()}")
                await run_scan_cycle(adapters, hist, cutoff, tracker=tracker)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Scan Loop Error: {e}", exc_info=True)
                await asyncio.sleep(60)

    scan_task    = asyncio.create_task(scan_loop())
    monitor_task = asyncio.create_task(monitor_loop(adapters, tracker))
    try:
        await asyncio.gather(scan_task, monitor_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        scan_task.cancel()
        monitor_task.cancel()
        await asyncio.gather(scan_task, monitor_task, return_exceptions=True)
        for a in adapters: await a.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
