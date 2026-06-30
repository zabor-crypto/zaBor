#!/usr/bin/env python3
"""
ALT 4H Reversal Scanner
=======================
A LONG-only 4H reversal scanner for USDT-margined perpetual altcoins
(universe = intersection of Binance and Bitget listings).

It runs three independent reversal detectors at every 4H candle close:
  1. squeeze_breakout  — Bollinger squeeze followed by a bullish breakout
  2. failed_breakdown  — intrabar break below a 60-bar low with a close back above (sweep + reclaim)
  3. sweep_reclaim     — multi-bar liquidity sweep with a reclaim and close lift

A filter stack gates every candidate: divergence-quality score, 4H ATR%
volatility floor, day/hour blacklist, liquidity-cohort filter, and an
optional 1H multi-timeframe (MTF) candle confirmation. A volatility/BTC
regime is classified per cycle and used to pick a 5-level R-multiple
take-profit ladder. The stop-loss is referenced to a 2H candle close.

Confirmed signals are written to a rotating log file and (optionally)
delivered to Telegram.

This is a SIGNAL-GENERATION tool only — it does NOT place orders, size
positions, or track live trades.

Author:  Boris Zabavnikov (zabor-crypto)
License: Apache-2.0

Usage:
  export TELEGRAM_BOT_TOKEN="..."   # optional; muted if unset
  export TELEGRAM_CHAT_ID="..."     # optional
  export BINANCE_API_KEY="..."      # optional (public klines work without keys)
  export BINANCE_API_SECRET="..."   # optional
  python signal_bot.py
"""

from __future__ import annotations
import os
import sys
import time
import math
import random
import socket
import hashlib
import asyncio
import logging
import signal
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from aiohttp import ClientSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from binance import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random


from aiohttp.client_exceptions import ClientError as AiohttpClientError
try:
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_ERRORS = (BinanceAPIException, BinanceRequestException)
except Exception:
    BINANCE_ERRORS = tuple()
from logging.handlers import RotatingFileHandler
from abc import ABC, abstractmethod
import hmac
import base64
import json
try:
    import fcntl
except Exception:
    fcntl = None

# ─── ENV CHECK ────────────────────────────────────────────────────────────────
REQUIRED = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
missing = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    print(f"Missing env vars (optional): {missing}. Telegram will be disabled.", file=sys.stderr)

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
LOG_FILE = os.getenv("LOG_FILE", "signal_bot_reversal.log")
LOG_LVL  = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_STREAM_TO_STDERR = os.getenv("LOG_STREAM_TO_STDERR", "1").strip().lower() in ("1", "true", "yes", "on")

logger = logging.getLogger("reversal_scanner")
logger.setLevel(getattr(logging, LOG_LVL, logging.INFO))
logger.propagate = False
for _h in list(logger.handlers):
    logger.removeHandler(_h)
    try:
        _h.close()
    except Exception:
        pass
formatter = logging.Formatter("[%(asctime)s UTC] %(levelname)s %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
formatter.converter = time.gmtime

fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=0, encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)
if LOG_STREAM_TO_STDERR:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ─── STRATEGY CONFIG ──────────────────────────────────────────────────────────
TOP_N = int(os.getenv("TOP_N", "300"))

MIN_VOL_BINANCE = float(os.getenv("MIN_VOL_BINANCE", "5000000"))
MIN_VOL_BITGET  = float(os.getenv("MIN_VOL_BITGET",  "1000000"))
# Universe liquidity floor on 30-day USD volume (per-signal, computed from 4H data).
# Excludes very-thin listings that pass the 24h gate but sit outside the validated
# universe. Lenient by default: validated signals have min vol_30d_usd ~$8M, so $10M
# keeps ~97% of validated signals while dropping the thin tail.
MIN_VOL_30D_USD = float(os.getenv("MIN_VOL_30D_USD", "10000000"))

# 4H gating
RSI_OVERSOLD_MAX = float(os.getenv("RSI_OVERSOLD_MAX", "70"))
MIN_PRICE_MOVE = float(os.getenv("MIN_PRICE_MOVE", "0.007"))       # 0.7%

# Market structure filters (reversal bottom detection)
MIN_DECLINE_FOR_REVERSAL = float(os.getenv("MIN_DECLINE_FOR_REVERSAL", "0.12"))  # 12%
MAX_DIV_POSITION = float(os.getenv("MAX_DIV_POSITION", "0.40"))  # div must be in lower 40% of range
RETEST_ZONE_PCT = float(os.getenv("RETEST_ZONE_PCT", "0.08"))   # pattern within 8% of div_low
STRUCTURE_LOOKBACK_BARS = int(os.getenv("STRUCTURE_LOOKBACK_BARS", "60"))  # 10 days on 4H

# Risk/reward for reversals
SL_BUFFER_ATR = float(os.getenv("SL_BUFFER_ATR", "0.25"))  # SL buffer below pattern low (ATR units)
TP_DERISK_R = float(os.getenv("TP_DERISK_R", "2.5"))       # de-risk reference (deprecated, use adaptive)
TP_SWING_MULT = float(os.getenv("TP_SWING_MULT", "0.85"))  # final TP at 85% to swing high

# Adaptive TP system
USE_ADAPTIVE_TPS = os.getenv("USE_ADAPTIVE_TPS", "1") == "1"
ATR_RATIO_CHOPPY_THRESHOLD = float(os.getenv("ATR_RATIO_CHOPPY_THRESHOLD", "1.3"))
ATR_RATIO_CONSOLIDATE_THRESHOLD = float(os.getenv("ATR_RATIO_CONSOLIDATE_THRESHOLD", "0.8"))
BREAKOUT_STRONG_ATR = float(os.getenv("BREAKOUT_STRONG_ATR", "1.5"))
BREAKOUT_WEAK_ATR = float(os.getenv("BREAKOUT_WEAK_ATR", "0.3"))

DIV_LB = int(os.getenv("DIV_LB", "60"))
PIVOT_L = int(os.getenv("PIVOT_L", "2"))
PIVOT_R = int(os.getenv("PIVOT_R", "2"))
MIN_PIV_DIST = int(os.getenv("MIN_PIV_DIST", "2"))

MIN_RSI_UP_POINTS = float(os.getenv("MIN_RSI_UP_POINTS", "1.0"))  # kept for reference
MAX_SETUP_TTL_H = int(os.getenv("SETUP_TTL_H", "16"))
WAIT_PATTERN_TTL_H = int(os.getenv("WAIT_PATTERN_TTL_H", "10"))
PATTERN_SCAN_BARS = int(os.getenv("PATTERN_SCAN_BARS", "12"))

# 1H arming/confirm
MAX_CONFIRM_BARS = int(os.getenv("MAX_CONFIRM_BARS", "18"))
MAX_PATTERN_DIST = float(os.getenv("MAX_PATTERN_DIST", "0.10"))
CONFIRM_CLOSE_MODE = os.getenv("CONFIRM_CLOSE_MODE", "HIGH")

# Entry ladder
LADDER_LEVELS = int(os.getenv("LADDER_LEVELS", "5"))
LADDER_WEIGHTS = os.getenv("LADDER_WEIGHTS", "0.28,0.24,0.20,0.16,0.12")
LADDER_BUFFER_ATR = float(os.getenv("LADDER_BUFFER_ATR", "0.10"))
LADDER_MIN_GAP_ATR = float(os.getenv("LADDER_MIN_GAP_ATR", "0.20"))
LADDER_FALLBACK_STEP_PCT = float(os.getenv("LADDER_FALLBACK_STEP_PCT", "0.0015"))
SL_ATR4H_MULT = float(os.getenv("SL_ATR4H_MULT", "0.45"))

# Divergence quality gate (backtest-validated)
# Score 0-11: S1=RSI delta(0-3), S2=div position(0-3), S3=decline depth(0-3), S4=volume(0-2)
# backtest: score>=6 -> PF ~4.5, WR ~81% on N=158 signals
MIN_DIV_SCORE = int(os.getenv("MIN_DIV_SCORE", "6"))

# Hard cap on SL distance as % of entry (backtest-validated):
# trades with risk > 4% get SL moved up to entry*(1-MAX_SL_RISK_PCT).
MAX_SL_RISK_PCT = float(os.getenv("MAX_SL_RISK_PCT", "0.04"))

EXIT_LEVELS = int(os.getenv("EXIT_LEVELS", "5"))
# Back-heavy exit weights (backtest-validated): keep more size for runners.
EXIT_WEIGHTS = os.getenv("EXIT_WEIGHTS", "0.10,0.15,0.20,0.25,0.30")

# BTC regime gate (default OFF — backtest shows strong-bear is the best regime
# for these reversal signals). Set BTC_GATE_ENABLED=1 to re-enable.
BTC_GATE_ENABLED      = os.getenv("BTC_GATE_ENABLED", "0") == "1"
BTC_7D_BEAR_THRESHOLD = float(os.getenv("BTC_7D_BEAR_THRESHOLD", "-0.15"))  # -15% = bear trend

SCANNER_VERSION = "1.0"

# Day-of-week filters. Wed gate kept (backtest PF 0.52); Sat gate default OFF.
DAY_FILTER_WED_ENABLED = os.getenv("DAY_FILTER_WED_ENABLED", "1") == "1"
DAY_FILTER_SAT_ENABLED = os.getenv("DAY_FILTER_SAT_ENABLED", "0") == "1"

# (weekday, hour_utc) cell blacklist. Default ON — backtest flagged these cells
# with weak performance.
DOW_HOUR_BLACKLIST_ENABLED = int(os.getenv("DOW_HOUR_BLACKLIST_ENABLED", "1"))
DOW_HOUR_BLACKLIST = {(2, 16), (4, 16), (3, 0), (0, 0)}  # Wed16, Fri16, Thu00, Mon00

# ATR_PCT_4H_MIN volatility-regime gate (default ON).
# Backtest shows monotonic PF rise with atr_pct_4h; threshold 0.04 retains the positive bucket.
ATR_PCT_4H_MIN_ENABLED = os.getenv("ATR_PCT_4H_MIN_ENABLED", "1") == "1"
ATR_PCT_4H_MIN         = float(os.getenv("ATR_PCT_4H_MIN", "0.04"))

# Optional divergence gates.
MIN_DIV_SCORE_ENABLED   = os.getenv("MIN_DIV_SCORE_ENABLED", "1") == "1"
MAX_DIV_POSITION_ENABLED = os.getenv("MAX_DIV_POSITION_ENABLED", "1") == "1"

# Alt detectors — default ON.
SQUEEZE_DETECTOR_ENABLED   = int(os.getenv("SQUEEZE_DETECTOR_ENABLED",   "1"))
FAILED_BD_DETECTOR_ENABLED = int(os.getenv("FAILED_BD_DETECTOR_ENABLED", "1"))
SWEEP_DETECTOR_ENABLED     = int(os.getenv("SWEEP_DETECTOR_ENABLED",     "1"))

# Classic divergence detector — default OFF (backtest PF ~0.98).
CLASSIC_DIV_ENABLED = int(os.getenv("CLASSIC_DIV_ENABLED", "0"))

# Regime-adaptive TP profiles — default ON.
REGIME_ADAPTIVE_TP_ENABLED = int(os.getenv("REGIME_ADAPTIVE_TP_ENABLED", "1"))

# Optional filter stages.
MTF_1H_CONFIRM_ENABLED = int(os.getenv("MTF_1H_CONFIRM_ENABLED", "1"))
COHORT_FILTER_ENABLED  = int(os.getenv("COHORT_FILTER_ENABLED",  "1"))

# Tag constants — sub-tags for logging; Telegram always shows the unified tag.
TAG_CLASSIC_DIV = "#longalt_reversal_classicdiv"
TAG_SQUEEZE     = "#longalt_reversal_squeeze"
TAG_FAILED_BD   = "#longalt_reversal_failedbd"
TAG_SWEEP       = "#longalt_reversal_sweep"

# R-multiple TP profile tables
TP_R_PROFILES: dict[str, tuple] = {
    "R_tight08": (0.5, 1.0, 1.6, 2.5, 4.0),
    "R_tight10": (0.6, 1.2, 2.0, 3.0, 5.0),
    "R_med12":   (0.8, 1.5, 2.5, 4.0, 6.0),
    "R_default": (0.9, 1.6, 2.5, 4.0, 6.0),
    "R_wide14":  (1.0, 2.0, 3.0, 5.0, 8.0),
    "R_wider16": (1.2, 2.4, 3.6, 5.5, 9.0),
}
TP_W_PROFILES: dict[str, tuple] = {
    "W_front":  (0.30, 0.25, 0.20, 0.15, 0.10),
    "W_even":   (0.20, 0.20, 0.20, 0.20, 0.20),
    "W_back":   (0.10, 0.15, 0.20, 0.25, 0.30),
    "W_runner": (0.05, 0.10, 0.20, 0.25, 0.40),
}
# Per-detector fallback profiles when REGIME_ADAPTIVE_TP_ENABLED=0 or cell missing
_ALT_DETECTOR_DEFAULT_PROFILES: dict[str, tuple[str, str]] = {
    "squeeze_breakout": ("R_default", "W_runner"),
    "failed_breakdown": ("R_tight10", "W_back"),
    "sweep_reclaim":    ("R_wide14",  "W_back"),
}

# Liquidity cohort (30d $-vol tercile) thresholds.
COHORT_T3_MAX_USD  = 883_011_180    # T3_micro: <= this
COHORT_T2_MAX_USD  = 2_790_378_790  # T2_mid: (T3_MAX, T2_MAX]; T1_major: > T2_MAX
DROP_COHORT_CELLS: frozenset = frozenset({("sweep_reclaim", "T2_mid")})

# Fallback TP models
TP_ATR_MULTS = os.getenv("TP_ATR_MULTS", "0.6,1.0,1.5,2.2,3.2")
TP_PCT_FALLBACK = os.getenv("TP_PCT_FALLBACK", "0.03,0.05,0.08,0.12,0.18")

# Candle-close safety
CLOSE_BUFFER_MS = int(os.getenv("CLOSE_BUFFER_MS", "8000"))
SAFETY_1H_SEC = int(os.getenv("SAFETY_1H_SEC", "15"))
SAFETY_4H_SEC = int(os.getenv("SAFETY_4H_SEC", "30"))

# Alerts
ALERT_ONLY_A_GRADE = os.getenv("ALERT_ONLY_A_GRADE", "0") == "1"
DEBUG_CHANNEL = os.getenv("DEBUG_CHANNEL", "0") == "1"

# ─── TELEGRAM SETTINGS ────────────────────────────────────────────────────────
TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
GROUP_TELEGRAM_ENABLED = os.getenv("GROUP_TELEGRAM_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
TG_GROUP_CHAT_ID = os.getenv("TELEGRAM_GROUP_CHAT_ID", "") if GROUP_TELEGRAM_ENABLED else ""
TG_GROUP_TOPIC_THREAD_ID_RAW = os.getenv("TELEGRAM_TOPIC_THREAD_ID", "")


def _parse_topic_thread_id(raw_value: str):
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Invalid TELEGRAM_TOPIC_THREAD_ID={raw!r}; group messages will go to main chat."
        )
        return None


def _build_tg_targets(*chat_ids: str) -> list[str]:
    targets: list[str] = []
    for raw in chat_ids:
        chat_id = str(raw or "").strip()
        if chat_id and chat_id not in targets:
            targets.append(chat_id)
    return targets


TG_TARGET_CHAT_IDS = _build_tg_targets(TG_CHAT_ID, TG_GROUP_CHAT_ID)
TG_ENABLED   = bool(TG_BOT_TOKEN and TG_TARGET_CHAT_IDS)
TG_GROUP_TOPIC_THREAD_ID = _parse_topic_thread_id(TG_GROUP_TOPIC_THREAD_ID_RAW)

TG_URL   = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
TG_MAX   = 3900
RETRY_JITTER_MIN_SEC = float(os.getenv("RETRY_JITTER_MIN_SEC", "0.20"))
RETRY_JITTER_MAX_SEC = float(os.getenv("RETRY_JITTER_MAX_SEC", "1.00"))
KLINES_RETRY_ATTEMPTS = max(1, int(os.getenv("KLINES_RETRY_ATTEMPTS", "3")))
KLINES_RETRY_BASE_SEC = float(os.getenv("KLINES_RETRY_BASE_SEC", "0.8"))

_tg_session: ClientSession | None = None
_last_msg_hash: str | None = None

# BTC regime state — updated once per hourly tick.
_btc_regime_ok: bool = True
_btc_regime_ts: float = 0.0
_btc_ret_7d: float | None = None


def _btc_regime_label(ret_7d) -> str:
    """Bucket BTC 7D return into a regime label."""
    if ret_7d is None:
        return "neutral"
    try:
        r = float(ret_7d)
    except (TypeError, ValueError):
        return "neutral"
    if r <= -0.15:  return "strong_bear"
    if r <= -0.05:  return "bear"
    if r <   0.05:  return "neutral"
    if r <   0.15:  return "bull"
    return "strong_bull"


# ─── PID LOCK ─────────────────────────────────────────────────────────────────
PIDFILE = os.getenv("PIDFILE", "/tmp/alt_4h_reversal.pid")


def acquire_pid_lock():
    if os.path.exists(PIDFILE):
        with open(PIDFILE, "r") as f:
            old = f.read().strip()
        try:
            os.kill(int(old), 0)
            raise RuntimeError(f"PIDFILE exists: {PIDFILE} (pid={old}). Refusing to start.")
        except (OSError, ValueError):
            pass
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))


def release_pid_lock():
    try:
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
    except Exception:
        pass


def startup_fingerprint():
    raw = f"{socket.gethostname()}|{os.getpid()}|{time.time_ns()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


# Concurrency & timeouts (env-tunable)
NET_CONCURRENCY   = int(os.getenv("NET_CONCURRENCY", "12"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "8"))
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", "12"))

KLIMIT_4H    = 400
RSI_PERIOD   = 14
STOCH_K, D   = 14, 3

TREND_EMA_PERIOD = 200
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

MAX_TREND_DISTANCE = float(os.getenv("MAX_TREND_DISTANCE", "0.08"))
MAX_RECENT_DROP    = float(os.getenv("MAX_RECENT_DROP", "0.35"))

SETUP_TTL_H = MAX_SETUP_TTL_H

NET_SEM = asyncio.BoundedSemaphore(NET_CONCURRENCY)

MAX_MON_H    = 12
TP_R_MULTS = os.getenv("TP_R_MULTS", "0.7,1.0,1.5,2.2,3.2")
TAG          = "#longalt_reversal"
SENT_SIGNALS_FILE = os.getenv("SENT_SIGNALS_FILE", "sent_signals.json")
MTF_PENDING_FILE  = os.getenv("MTF_PENDING_FILE", "mtf_pending_signals.json")
MTF_MAX_WINDOW_H  = 6   # reject after 6 closed 1H bars with no pattern match
MTF_CLEANUP_H     = 8   # drop entries older than this on startup
MARKET_STATE_FILE = os.getenv("MARKET_STATE_FILE", "market_state.json")


def _telegram_display_tag(internal_tag: str) -> str:
    """Strip the sub-tag suffix for Telegram display."""
    if internal_tag.startswith("#longalt_reversal_"):
        return "#longalt_reversal"
    return internal_tag


# ─── MTF 1H PATTERN FUNCTIONS ────────────────────────────────────────────────
# These candle definitions are what the MTF-confirm backtest used; do not alter.

def _is_bullish_engulfing(prev: dict, curr: dict) -> bool:
    return (prev["close"] < prev["open"]
            and curr["close"] > curr["open"]
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"])


def _is_hammer(bar: dict) -> bool:
    body = abs(bar["close"] - bar["open"])
    rng  = bar["high"] - bar["low"]
    if rng <= 0 or body <= 0:
        return False
    lower_shadow = min(bar["open"], bar["close"]) - bar["low"]
    upper_shadow = bar["high"] - max(bar["open"], bar["close"])
    return lower_shadow >= 2 * body and upper_shadow <= body * 0.5


def _is_piercing(prev: dict, curr: dict) -> bool:
    if prev["close"] >= prev["open"]:
        return False
    if curr["close"] <= curr["open"]:
        return False
    midpoint = (prev["open"] + prev["close"]) / 2.0
    return curr["open"] < prev["close"] and curr["close"] > midpoint


def _is_two_higher_closes(bars: list, i: int) -> bool:
    if i < 2:
        return False
    return bars[i]["close"] > bars[i - 1]["close"] and bars[i - 1]["close"] > bars[i - 2]["close"]


def _check_1h_confirmation(bars_1h: list, max_bars: int = 6):
    """Return (confirmed, pattern, bar_idx)."""
    n = min(len(bars_1h), max_bars)
    for i in range(n):
        b = bars_1h[i]
        if _is_hammer(b):
            return True, "hammer", i
        if i > 0 and _is_bullish_engulfing(bars_1h[i - 1], b):
            return True, "bull_engulfing", i
        if i > 0 and _is_piercing(bars_1h[i - 1], b):
            return True, "piercing", i
        if _is_two_higher_closes(bars_1h, i):
            return True, "two_higher_closes", i
    return False, None, -1


def _classify_cohort(vol_30d_usd) -> str:
    """Classify 30d $-volume into T1_major / T2_mid / T3_micro / unknown."""
    if vol_30d_usd is None:
        return "unknown"
    if vol_30d_usd <= COHORT_T3_MAX_USD:
        return "T3_micro"
    if vol_30d_usd <= COHORT_T2_MAX_USD:
        return "T2_mid"
    return "T1_major"


def _compute_vol_30d_usd(df4: "pd.DataFrame") -> "float | None":
    """Sum close*volume over last 180 4H bars (=30 days). Returns None if <30 bars."""
    if df4 is None or len(df4) < 30:
        return None
    last_180 = df4.tail(180)
    return float((last_180["Close"] * last_180["Volume"]).sum())


# Per-cell regime TP winners (optional external override map; empty by default).
_REGIME_TP_WINNERS: dict = {}

# ─── MTF PENDING STATE ────────────────────────────────────────────────────────
# Keyed "EXCHANGE:SYMBOL". Persisted to MTF_PENDING_FILE on every change.
_mtf_pending: dict = {}


def _load_mtf_pending() -> None:
    global _mtf_pending
    try:
        if os.path.exists(MTF_PENDING_FILE):
            with open(MTF_PENDING_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cutoff = time.time() - MTF_CLEANUP_H * 3600
            _mtf_pending = {k: v for k, v in raw.items() if float(v.get("registered_ts", 0)) >= cutoff}
            dropped = len(raw) - len(_mtf_pending)
            if dropped:
                logger.info(f"[MTF] Loaded {len(_mtf_pending)} pending, dropped {dropped} stale (>{MTF_CLEANUP_H}h)")
            elif _mtf_pending:
                logger.info(f"[MTF] Loaded {len(_mtf_pending)} pending signals from {MTF_PENDING_FILE}")
    except Exception as e:
        logger.warning(f"[MTF] Failed to load {MTF_PENDING_FILE}: {e!r}")
        _mtf_pending = {}


def _save_mtf_pending() -> None:
    try:
        tmp = MTF_PENDING_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_mtf_pending, f, ensure_ascii=False, default=str)
        os.replace(tmp, MTF_PENDING_FILE)
    except Exception as e:
        logger.warning(f"[MTF] Failed to save {MTF_PENDING_FILE}: {e!r}")


SIGNAL_COOLDOWN_H = float(os.getenv("SIGNAL_COOLDOWN_H", "24"))
BINANCE_PRIORITY_WINDOW_SEC = int(os.getenv("BINANCE_PRIORITY_WINDOW_SEC", "900"))  # 15 minutes

# Stablecoins / fiat / wrapped quote bases to exclude
BLACKLIST_BASES = {
    "USDC", "FDUSD", "TUSD", "PAX", "USDP", "DAI", "EUR", "GBP", "JPY", "USD1", "BUSD",
    "USDT", "SUSD", "AUD", "BKRW", "USDS", "GUSD"
}

# Signal-level blacklist — chronic underperformers (backtest-validated).
SIGNAL_BLACKLIST: set[str] = set(os.getenv("SIGNAL_BLACKLIST", "").split(",")) | {
    "COLLECTUSDT",
    "ENSOUSDT",
    "HUMAUSDT",
    "TRUMPUSDT",
    "XRPUSDT",
    "BGBUSDT",
    "BONKUSDT",
    "LATUSDT",
    "NIGHTUSDT",
    "WIFUSDT",
    "ALCHUSDT",
    "BARDUSDT",
    "SEIUSDT",
}
SIGNAL_BLACKLIST.discard("")

# ─── PUBLIC UNIVERSE FETCHERS ─────────────────────────────────────────────────
# Universe = Binance USDT-M futures listings intersected with Bitget USDT futures
# contracts, restricted to symbols that also trade on the spot endpoints used here.
BITGET_FUTURES_CONTRACTS_CACHE_TTL_SEC = max(60, int(os.getenv("BITGET_FUTURES_CONTRACTS_CACHE_TTL_SEC", "900")))
BINANCE_FUTURES_SYMBOLS_CACHE_TTL_SEC = max(60, int(os.getenv("BINANCE_FUTURES_SYMBOLS_CACHE_TTL_SEC", "900")))
UNIVERSE_INTERSECTION_ENABLED = os.getenv("UNIVERSE_INTERSECTION_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")

_BITGET_FUTURES_CACHE = {"ts": 0.0, "symbols": None}
_BINANCE_FUTURES_LISTING_CACHE = {"ts": 0.0, "symbols": None}


def _normalize_symbol_token(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum())


def _is_normal_binance_futures_symbol(symbol: str) -> bool:
    sym = str(symbol or "").upper().strip()
    return bool(sym) and sym.endswith("USDT") and sym.isascii() and sym.isalnum() and 6 <= len(sym) <= 20


def _fetch_bitget_futures_symbols_sync() -> set[str]:
    resp = requests.get(
        "https://api.bitget.com/api/v2/mix/market/contracts",
        params={"productType": "USDT-FUTURES"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or []
    out: set[str] = set()
    for row in data:
        sym = _normalize_symbol_token((row or {}).get("symbol"))
        if sym:
            out.add(sym)
    if not out:
        raise RuntimeError("empty Bitget futures contracts set")
    return out


def _fetch_binance_futures_listing_symbols_sync() -> set[str]:
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("symbols") or []
    out: set[str] = set()
    for row in data:
        raw_symbol = str((row or {}).get("symbol") or "").upper().strip()
        if not _is_normal_binance_futures_symbol(raw_symbol):
            continue
        sym = _normalize_symbol_token(raw_symbol)
        if sym:
            out.add(sym)
    if not out:
        raise RuntimeError("empty Binance futures symbols set")
    return out


async def _get_bitget_futures_symbols() -> set[str] | None:
    now_ts = time.time()
    cached = _BITGET_FUTURES_CACHE
    cached_symbols = cached.get("symbols")
    if isinstance(cached_symbols, set) and (now_ts - float(cached.get("ts") or 0.0)) < BITGET_FUTURES_CONTRACTS_CACHE_TTL_SEC:
        return cached_symbols
    try:
        symbols = await asyncio.to_thread(_fetch_bitget_futures_symbols_sync)
    except Exception as exc:
        logger.warning(f"[UNIVERSE] Failed to refresh Bitget futures symbols: {exc!r}")
        return cached_symbols if isinstance(cached_symbols, set) else None
    cached["ts"] = now_ts
    cached["symbols"] = symbols
    return symbols


async def _get_binance_futures_listing_symbols() -> set[str] | None:
    now_ts = time.time()
    cached = _BINANCE_FUTURES_LISTING_CACHE
    cached_symbols = cached.get("symbols")
    if isinstance(cached_symbols, set) and (now_ts - float(cached.get("ts") or 0.0)) < BINANCE_FUTURES_SYMBOLS_CACHE_TTL_SEC:
        return cached_symbols
    try:
        symbols = await asyncio.to_thread(_fetch_binance_futures_listing_symbols_sync)
    except Exception as exc:
        logger.warning(f"[UNIVERSE] Failed to refresh Binance futures listing symbols: {exc!r}")
        return cached_symbols if isinstance(cached_symbols, set) else None
    cached["ts"] = now_ts
    cached["symbols"] = symbols
    return symbols


async def _filter_universe_intersection(symbols: list[str], source_exchange: str) -> list[str]:
    """Restrict a spot-symbol list to the Binance-futures ∩ Bitget-futures universe."""
    if (not UNIVERSE_INTERSECTION_ENABLED) or (not symbols):
        return symbols

    bitget_syms = await _get_bitget_futures_symbols()
    binance_syms = await _get_binance_futures_listing_symbols()
    if not bitget_syms or not binance_syms:
        logger.warning(f"[{source_exchange}] Universe intersection bypassed: a futures symbol set is unavailable")
        return symbols

    tradable = bitget_syms & binance_syms
    kept: list[str] = []
    dropped: list[str] = []
    for sym in symbols:
        norm = _normalize_symbol_token(sym)
        if norm in tradable:
            kept.append(sym)
        elif len(dropped) < 10:
            dropped.append(sym)

    logger.info(
        f"[{source_exchange}] Universe intersection: {len(symbols)} -> {len(kept)} "
        f"(dropped={len(symbols) - len(kept)})"
    )
    if dropped:
        logger.info(f"[{source_exchange}] Universe dropped sample: {', '.join(dropped)}")
    return kept


# ─── RATE LIMITER ─────────────────────────────────────────────────────────────
_BINANCE_EXINFO_CACHE = {"ts": 0, "data": None}
_BINANCE_TICKER_CACHE = {"ts": 0, "data": None}


class RateLimiter:
    def __init__(self, maxw=1200, period=60.0):
        self.MAXW = maxw
        self.PERIOD = period
        self.tokens = self.MAXW
        self.last   = time.time()
        self._lock  = asyncio.Lock()

    async def wait(self, w=1):
        async with self._lock:
            now = time.time()
            self.tokens = min(self.MAXW,
                              self.tokens + (now - self.last) * (self.MAXW / self.PERIOD))
            self.last = now
            if self.tokens >= w:
                self.tokens -= w
                return
            delay = (w - self.tokens) * (self.PERIOD / self.MAXW)
            if delay > 0:
                logger.debug(f"RateLimiter sleeping {delay:.2f}s")
                await asyncio.sleep(delay)
            self.last = time.time()
            self.tokens = 0.0


# ─── 4H FETCH ─────────────────────────────────────────────────────────────────
_CACHE_STORE = {}


def drop_live_candle(df: pd.DataFrame, now_ms: int, safety_sec: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Close_time" not in df.columns:
        return df
    last_close = int(df["Close_time"].iloc[-1])
    if last_close > now_ms - safety_sec * 1000:
        return df.iloc[:-1].reset_index(drop=True)
    return df


async def fetch_4h(adapter: "ExchangeAdapter", symbol: str, now_ms: int) -> pd.DataFrame:
    cache_key = f"{adapter.name}:{symbol}"

    if cache_key in _CACHE_STORE:
        cached = _CACHE_STORE[cache_key]
        if not cached.empty:
            last_close_ms = int(cached["Close_time"].iloc[-1])
            current_4h_start = (now_ms // (4 * 3600 * 1000)) * (4 * 3600 * 1000)
            if last_close_ms >= (current_4h_start - 1):
                return cached

    raw = await adapter.get_klines(symbol=symbol, interval="4h", limit=KLIMIT_4H)

    df = pd.DataFrame(raw, columns=["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df["Open_time"] = df["Open_time"].astype("int64")
    df["Close_time"] = df["Close_time"].astype("int64")

    df = drop_live_candle(df, now_ms, SAFETY_4H_SEC)

    _CACHE_STORE[cache_key] = df
    return df


# ─── TELEGRAM SENDER ─────────────────────────────────────────────────────────
async def send_telegram(text: str, exchange: str = "", symbol: str = "", log_text: str | None = None):
    """Send a Telegram message (chunked, with retry). Mutes gracefully if unconfigured."""
    global _tg_session, _last_msg_hash

    all_ok = True
    if not TG_ENABLED:
        all_ok = False
        logger.info(f"TG (muted): {text}")
    else:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        if _tg_session is None or _tg_session.closed:
            _tg_session = ClientSession()
        lines = text.split('\n')
        chunk_size = 4000
        current_chunk = ""

        def _build_payload(chat_id: str, chunk_text: str) -> dict:
            payload = {"chat_id": chat_id, "text": chunk_text, "parse_mode": "HTML"}
            if (
                TG_GROUP_CHAT_ID
                and chat_id == TG_GROUP_CHAT_ID
                and TG_GROUP_TOPIC_THREAD_ID is not None
            ):
                payload["message_thread_id"] = TG_GROUP_TOPIC_THREAD_ID
            return payload

        async def _send(payload: dict) -> None:
            nonlocal all_ok
            for attempt in range(1, 4):
                try:
                    async with _tg_session.post(url, json=payload, timeout=10) as resp:
                        body = await resp.text()
                        if resp.status == 200:
                            return
                        if resp.status == 429 and attempt < 3:
                            retry_after = 1.0
                            try:
                                data = json.loads(body)
                                retry_after = float(data.get("parameters", {}).get("retry_after", 1))
                            except Exception:
                                pass
                            wait_s = min(
                                15.0,
                                retry_after + random.uniform(RETRY_JITTER_MIN_SEC, RETRY_JITTER_MAX_SEC),
                            )
                            logger.warning(
                                f"TG 429 (aiohttp attempt {attempt}/3); retry in {wait_s:.2f}s"
                            )
                            await asyncio.sleep(wait_s)
                            continue
                        logger.error(
                            f"TG Error {resp.status} (aiohttp attempt {attempt}/3): {body}"
                        )
                except Exception as e:
                    logger.warning(
                        f"TG aiohttp failed (attempt {attempt}/3, {type(e).__name__}): {e!r}"
                    )
                if attempt < 3:
                    wait_s = min(
                        15.0,
                        (2 ** (attempt - 1)) + random.uniform(RETRY_JITTER_MIN_SEC, RETRY_JITTER_MAX_SEC),
                    )
                    await asyncio.sleep(wait_s)

            for fallback_attempt in range(1, 3):
                try:
                    resp = requests.post(url, json=payload, timeout=10)
                    if resp.status_code == 200:
                        logger.warning(
                            "TG delivery succeeded via requests fallback after aiohttp failures."
                        )
                        return
                    logger.error(
                        "TG requests fallback error "
                        f"{resp.status_code} (attempt {fallback_attempt}/2): {resp.text}"
                    )
                except Exception as e:
                    logger.error(
                        "TG requests fallback failed "
                        f"(attempt {fallback_attempt}/2, {type(e).__name__}): {e!r}"
                    )
                if fallback_attempt < 2:
                    await asyncio.sleep(
                        min(
                            15.0,
                            1 + random.uniform(RETRY_JITTER_MIN_SEC, RETRY_JITTER_MAX_SEC),
                        )
                    )

            all_ok = False

        for line in lines:
            if len(current_chunk) + len(line) + 1 > chunk_size:
                for chat_id in TG_TARGET_CHAT_IDS:
                    payload = _build_payload(chat_id, current_chunk)
                    await _send(payload)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            for chat_id in TG_TARGET_CHAT_IDS:
                payload = _build_payload(chat_id, current_chunk)
                await _send(payload)

    if not all_ok:
        logger.warning("Signal persisted despite Telegram delivery failure/mute")


# ─── ARMED BOOK (MARKET STATE) ───────────────────────────────────────────────
class MarketState:
    def __init__(self):
        self.books = {}
        self._dirty = False

    def update_4h_setup(self, exchange, symbol, price_at_div, ts, atr4h: float | None = None):
        key = f"{exchange}:{symbol}"
        current = self.books.get(key)

        if not current:
            logger.info(f"[{exchange}] {symbol}: NEW 4H SETUP registered (div_pivot={price_at_div:.6f})")
            self.books[key] = {
                "status": "WAITING",
                "div_price": price_at_div,
                "setup_time": ts,
                "atr4h": float(atr4h) if (atr4h is not None and np.isfinite(atr4h)) else 0.0,
                "state_changed": time.time(),
                "armed_info": None,
                "bars_waited": 0
            }
        else:
            if current["status"] == "WAITING":
                current["div_price"] = float(price_at_div)
                current["setup_time"] = int(ts)
                if atr4h is not None and np.isfinite(atr4h):
                    current["atr4h"] = float(atr4h)
                current["state_changed"] = time.time()
                logger.info(f"[{exchange}] {symbol}: REFRESH 4H setup (div_price={price_at_div:.6f})")
            elif current["status"] == "ARMED":
                if float(price_at_div) < float(current.get("div_price", price_at_div)):
                    current["div_price"] = float(price_at_div)
                    logger.info(f"[{exchange}] {symbol}: UPDATED div_price while ARMED -> {price_at_div:.6f}")
                if atr4h is not None and np.isfinite(atr4h):
                    current["atr4h"] = float(atr4h)

        self._dirty = True

    def get_active_keys(self):
        return list(self.books.keys())

    def remove(self, key):
        if key in self.books:
            del self.books[key]
            self._dirty = True

    def flush(self):
        if not self._dirty:
            return
        self._dirty = False
        try:
            tmp = MARKET_STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.books, f, indent=2, default=_json_default)
            os.replace(tmp, MARKET_STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to save state ({type(e).__name__}): {e!r}", exc_info=True)

    def save(self):
        self._dirty = True
        self.flush()

    def load(self):
        if os.path.exists(MARKET_STATE_FILE):
            try:
                with open(MARKET_STATE_FILE, "r") as f:
                    self.books = json.load(f)
                logger.info(f"Loaded {len(self.books)} setups from disk.")
            except Exception as e:
                logger.error(f"Failed to load state ({type(e).__name__}): {e!r}", exc_info=True)


def _json_default(o):
    try:
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


market_state = MarketState()
market_state.load()
_load_mtf_pending()


class SentSignals:
    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, dict] = {}
        self._dirty = False
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load {self.path} ({type(e).__name__}): {e!r}", exc_info=True)
            self.data = {}

    def flush(self):
        if not self._dirty:
            return
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=_json_default)
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception as e:
            logger.error(f"Failed to save {self.path} ({type(e).__name__}): {e!r}", exc_info=True)

    def mark_sent(self, symbol: str, exchange: str, ts_ms: int):
        self.data[symbol] = {"exchange": exchange, "ts_ms": int(ts_ms)}
        self._dirty = True

    def get_last(self, symbol: str):
        return self.data.get(symbol)

    def in_cooldown(self, symbol: str, now_ms: int) -> bool:
        last = self.get_last(symbol)
        if not last:
            return False
        age_ms = now_ms - int(last.get("ts_ms", 0))
        return age_ms < int(SIGNAL_COOLDOWN_H * 3600 * 1000)


sent_signals = SentSignals(SENT_SIGNALS_FILE)


# ─── EXCHANGE ADAPTERS ───────────────────────────────────────────────────────
class ExchangeAdapter(ABC):
    def __init__(self, name):
        self.name = name
        self.sem = asyncio.BoundedSemaphore(int(os.getenv("NET_CONCURRENCY", "12")))
        self.rate_limiter = RateLimiter()

    def _norm_symbol(self, s: str) -> str:
        return s

    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int):
        pass

    @abstractmethod
    async def get_tickers(self):
        pass

    @abstractmethod
    async def get_trading_symbols(self):
        pass

    @abstractmethod
    async def get_server_time(self) -> int:
        pass

    @abstractmethod
    async def get_tick_size(self, symbol: str) -> float:
        pass

    @abstractmethod
    async def close(self):
        pass


class BinanceAdapter(ExchangeAdapter):
    def __init__(self, client: AsyncClient):
        super().__init__("BINANCE")
        self.client = client
        self.C4 = AsyncClient.KLINE_INTERVAL_4HOUR
        self.C1 = AsyncClient.KLINE_INTERVAL_1HOUR
        self.rate_limiter = RateLimiter(maxw=1200, period=60.0)

    async def get_klines(self, symbol, interval, limit):
        b_interval = self.C4 if interval == "4h" else self.C1

        last_err = None
        for attempt in range(1, KLINES_RETRY_ATTEMPTS + 1):
            try:
                await self.rate_limiter.wait(1)
                async with self.sem:
                    raw = await self.client.get_klines(symbol=symbol, interval=b_interval, limit=limit)
                break
            except Exception as e:
                last_err = e
                err_txt = str(e).lower()
                transient = (
                    isinstance(e, (asyncio.TimeoutError, TimeoutError, socket.timeout, AiohttpClientError))
                    or (BINANCE_ERRORS and isinstance(e, BINANCE_ERRORS))
                    or ("timeout" in err_txt)
                    or ("timed out" in err_txt)
                    or ("temporarily unavailable" in err_txt)
                )
                if (not transient) or attempt >= KLINES_RETRY_ATTEMPTS:
                    raise
                wait_s = min(
                    6.0,
                    (KLINES_RETRY_BASE_SEC * (2 ** (attempt - 1)))
                    + random.uniform(RETRY_JITTER_MIN_SEC, RETRY_JITTER_MAX_SEC),
                )
                logger.warning(
                    f"{symbol}: get_klines transient error ({type(e).__name__}) "
                    f"attempt {attempt}/{KLINES_RETRY_ATTEMPTS}; retry in {wait_s:.2f}s"
                )
                await asyncio.sleep(wait_s)
        else:
            raise RuntimeError(f"{symbol}: get_klines exhausted retries ({last_err})")

        # STRICT CONTRACT: return only 7 columns: [ts, o, h, l, c, v, ct]
        normalized = []
        for r in raw:
            if len(r) >= 7:
                normalized.append(r[:7])
            else:
                padded = r + [0] * (7 - len(r))
                normalized.append(padded[:7])
        return normalized

    async def get_tickers(self):
        now = time.time()
        if _BINANCE_TICKER_CACHE["data"] is not None and (now - _BINANCE_TICKER_CACHE["ts"]) < 60:
            return _BINANCE_TICKER_CACHE["data"]
        await self.rate_limiter.wait(10)
        async with self.sem:
            data = await self.client.get_ticker()
        _BINANCE_TICKER_CACHE.update(ts=now, data=data)
        return data

    async def get_trading_symbols(self):
        import re
        _SYMBOL_RE = re.compile(r"^[A-Z0-9]{6,20}$")

        now = time.time()
        if _BINANCE_EXINFO_CACHE["data"] is not None and (now - _BINANCE_EXINFO_CACHE["ts"]) < 6 * 3600:
            info = _BINANCE_EXINFO_CACHE["data"]
        else:
            await self.rate_limiter.wait(20)
            async with self.sem:
                info = await self.client.get_exchange_info()
            _BINANCE_EXINFO_CACHE.update(ts=now, data=info)

        usdt_syms = []
        for s in info["symbols"]:
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
                base = s["baseAsset"]
                if base in BLACKLIST_BASES:
                    continue
                if _SYMBOL_RE.match(s["symbol"]) is not None:
                    if not any(x in s["symbol"] for x in ("UP", "DOWN", "BEAR", "BULL")):
                        usdt_syms.append(s["symbol"])

        logger.info(f"[BINANCE] Symbols total={len(info['symbols'])} usdt_only={len(usdt_syms)}")
        return usdt_syms

    async def get_server_time(self):
        await self.rate_limiter.wait(1)
        async with self.sem:
            st = await self.client.get_server_time()
        return int(st["serverTime"])

    async def get_tick_size(self, symbol: str) -> float:
        info = _BINANCE_EXINFO_CACHE["data"]
        if not info:
            await self.get_trading_symbols()
            info = _BINANCE_EXINFO_CACHE["data"]
        if info:
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            return float(f["tickSize"])
        return 0.0

    async def close(self):
        await self.client.close_connection()


class BitgetSpotAdapter(ExchangeAdapter):
    def __init__(self, api_key=None, secret=None, passphrase=None):
        super().__init__("BITGET")
        self.api_key = api_key or ""
        self.secret = secret or ""
        self.passphrase = passphrase or ""
        self.base_url = "https://api.bitget.com"
        self.session = ClientSession()
        self.rate_limiter = RateLimiter(maxw=300, period=60.0)

    def _map_granularity(self, interval: str) -> str:
        m = {"1h": "1h", "4h": "4h"}
        return m.get(interval, interval)

    def _norm_symbol(self, s: str) -> str:
        return s.replace("_SPBL", "").replace("-", "")

    async def _request_public(self, method, endpoint, params=None):
        url = self.base_url + endpoint
        await self.rate_limiter.wait(1)
        async with self.sem:
            async with self.session.request(method, url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Bitget Public {resp.status}: {text}")
                return await resp.json()

    async def _request(self, method, endpoint, params=None):
        ts = str(int(time.time() * 1000))
        sign_path = endpoint
        body_str = ""

        if method == "GET" and params:
            import urllib.parse
            q = urllib.parse.urlencode(params)
            sign_path += f"?{q}"
            endpoint += f"?{q}"
        elif method == "POST" and params:
            body_str = json.dumps(params)

        sign_str = f"{ts}{method.upper()}{sign_path}{body_str}"
        signature = hmac.new(self.secret.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature_b64,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US"
        }

        await self.rate_limiter.wait(1)
        async with self.sem:
            async with self.session.request(method, self.base_url + endpoint,
                                            headers=headers, data=body_str, timeout=10) as resp:
                if resp.status != 200:
                    try:
                        text = await resp.text()
                    except Exception:
                        text = "err"
                    raise RuntimeError(f"Bitget {resp.status}: {text}")
                return await resp.json()

    async def get_klines(self, symbol, interval, limit):
        gram = self._map_granularity(interval)
        params = {"symbol": symbol, "granularity": gram, "limit": limit}
        res = await self._request_public("GET", "/api/v2/spot/market/candles", params)

        if res.get("code") != "00000":
            sec_map = {"1h": "3600", "4h": "14400"}
            gram2 = sec_map.get(interval)
            if gram2:
                params["granularity"] = gram2
                res = await self._request_public("GET", "/api/v2/spot/market/candles", params)
                if res.get("code") != "00000":
                    logger.warning(f"Bitget kline error: {res}")
                    return []
            else:
                logger.warning(f"Bitget kline error: {res}")
                return []

        data = res.get("data", [])
        if not data:
            return []

        dur = (4 * 3600 * 1000) if interval == "4h" else (3600 * 1000)

        normalized = []
        for c in data:
            ts = int(c[0])
            ct = ts + dur - 1
            normalized.append([ts, c[1], c[2], c[3], c[4], c[5], ct])

        if normalized:
            unique = {row[0]: row for row in normalized}
            normalized = list(unique.values())
            normalized.sort(key=lambda x: x[0])
            times = [r[0] for r in normalized]
            if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
                logger.error(f"Bitget klines not monotonic after sort! {times[:5]}...")
                return []

        return normalized

    async def get_tickers(self):
        res = await self._request_public("GET", "/api/v2/spot/market/tickers")
        data = res.get("data", [])
        for t in data:
            t["symbol"] = self._norm_symbol(t["symbol"])
        return data

    async def get_trading_symbols(self):
        res = await self._request_public("GET", "/api/v2/spot/public/symbols")
        data = res.get("data", [])

        if not hasattr(self, "_symbol_cache"):
            self._symbol_cache = {}

        valid = []
        for s in data:
            sym = self._norm_symbol(s["symbol"])
            ts = 0.0
            if "tickSize" in s:
                try:
                    ts = float(s["tickSize"])
                except Exception:
                    pass
            if ts == 0.0 and "priceStep" in s:
                try:
                    ts = float(s["priceStep"])
                except Exception:
                    pass
            if ts == 0.0:
                try:
                    p = int(s.get("pricePrecision", 6))
                    ts = 10 ** -p
                except Exception:
                    ts = 0.000001

            self._symbol_cache[sym] = ts

            if str(s.get("status")).lower() == "online" and s.get("quoteCoin") == "USDT":
                base = s.get("baseCoin")
                if base in BLACKLIST_BASES:
                    continue
                if "UP" in sym or "DOWN" in sym or "BEAR" in sym or "BULL" in sym:
                    continue
                valid.append(sym)
        return valid

    async def get_tick_size(self, symbol: str) -> float:
        if not hasattr(self, "_symbol_cache") or symbol not in self._symbol_cache:
            await self.get_trading_symbols()
        return self._symbol_cache.get(symbol, 0.0)

    async def get_server_time(self) -> int:
        res = await self._request_public("GET", "/api/v2/public/time")
        data = res.get("data")

        if isinstance(data, (str, int, float)):
            return int(data)
        if isinstance(data, dict):
            for k in ("serverTime", "server_time", "ts", "timestamp", "time"):
                if k in data:
                    return int(data[k])
        for k in ("serverTime", "server_time", "ts", "timestamp", "time"):
            if k in res:
                return int(res[k])

        logger.warning(f"Bitget time parse failed: {res}")
        return int(time.time() * 1000)

    async def close(self):
        await self.session.close()


# ─── VECTORIZED INDICATORS & PATTERNS ────────────────────────────────────────
def rsi(series, p):
    d = series.diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    ag = g.ewm(alpha=1 / p, adjust=False).mean()
    al = l.ewm(alpha=1 / p, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stoch(df, k, d):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    rng = (hi - lo).replace(0, np.nan)
    k_val = 100 * (df["Close"] - lo) / rng
    k_val = k_val.clip(0, 100).ffill()
    return k_val, k_val.rolling(d).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_all_indicators(df):
    """Compute all indicators once and cache them as DataFrame columns."""
    if len(df) < 80:
        return None

    df = df.copy(deep=False)
    df['rsi'] = rsi(df["Close"], RSI_PERIOD)
    df['stoch_k'], df['stoch_d'] = stoch(df, STOCH_K, D)
    df['ema200'] = ema(df["Close"], TREND_EMA_PERIOD) if len(df) >= TREND_EMA_PERIOD else np.nan
    df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['atr4h'] = atr(df, 14)

    return df


def find_pivot_lows(lows: np.ndarray, left: int, right: int) -> list[int]:
    idxs = []
    n = len(lows)
    for i in range(left, n - right):
        window = lows[i - left:i + right + 1]
        if lows[i] == window.min():
            idxs.append(i)
    return idxs


def pick_two_pivots(pivot_idxs: list[int], lows: np.ndarray, min_dist: int) -> tuple[int, int] | None:
    if len(pivot_idxs) < 2:
        return None
    piv = sorted(set(pivot_idxs))
    b = piv[-1]
    for a in reversed(piv[:-1]):
        if (b - a) >= min_dist:
            return (a, b)
    return None


def body(o, c): return abs(c - o)
def upper_wick(o, h, c): return h - max(o, c)
def lower_wick(o, l, c): return min(o, c) - l


def is_bull_engulf(prev, cur):
    po, pc = prev["Open"], prev["Close"]
    co, cc = cur["Open"], cur["Close"]
    return (pc < po) and (cc > co) and (co <= pc) and (cc >= po)


def is_piercing(prev, cur):
    po, pc = prev["Open"], prev["Close"]
    co, cc = cur["Open"], cur["Close"]
    mid = (po + pc) / 2
    return (pc < po) and (cc > mid) and (co < prev["Low"]) and (cc < po)


def is_bull_harami(prev, cur):
    po, pc = prev["Open"], prev["Close"]
    co, cc = cur["Open"], cur["Close"]
    return (pc < po) and (cc > co) and (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])


def is_hammer(cur):
    o, h, l, c = cur["Open"], cur["High"], cur["Low"], cur["Close"]
    b = body(o, c)

    if b < (h - l) * 0.03:
        b_eff = (h - l) * 0.03
    else:
        b_eff = b

    lw = lower_wick(o, l, c)
    uw = upper_wick(o, h, c)

    if lw < 1.7 * b_eff: return False
    if uw > 1.0 * b_eff: return False
    if c < o: return False

    return True


def is_morning_star(a, b, c):
    ao, ac = a["Open"], a["Close"]
    bo, bc = b["Open"], b["Close"]
    co, cc = c["Open"], c["Close"]
    if not (ac < ao and cc > co):
        return False
    if body(bo, bc) > 0.7 * body(ao, ac):
        return False
    return cc > (ao + ac) / 2


PATTERN_FUNCS = [
    ("engulf", is_bull_engulf),
    ("piercing", is_piercing),
    ("harami", is_bull_harami),
    ("hammer", lambda prev, cur: is_hammer(cur)),
    ("morning_star", None),
]


def detect_pattern(df_closed, setup_time_ms: int):
    if len(df_closed) < 3:
        return None, None

    start = max(2, len(df_closed) - max(3, PATTERN_SCAN_BARS))
    for idx in range(len(df_closed) - 1, start - 1, -1):
        cur = df_closed.iloc[idx]
        if int(cur["Close_time"]) < int(setup_time_ms):
            continue
        prev = df_closed.iloc[idx - 1]

        for name, fn in PATTERN_FUNCS:
            if name == "morning_star":
                continue
            if fn(prev, cur):
                return name, idx

        if idx >= 2:
            a = df_closed.iloc[idx - 2]
            b = df_closed.iloc[idx - 1]
            c = df_closed.iloc[idx]
            if is_morning_star(a, b, c):
                return "morning_star", idx

    return None, None


def volume_grade(df_closed, idx: int) -> str:
    if idx < 20:
        return "B"
    v = float(df_closed["Volume"].iloc[idx])
    vma = float(df_closed["Volume"].iloc[idx - 20:idx].mean())
    if np.isnan(vma) or vma == 0:
        return "B"
    return "A" if v >= 1.05 * vma else "B"


def compute_div_score(rsi_delta: float, div_position: float, decline_pct: float, vol_qual: str) -> int:
    """
    Divergence quality score 0-11 (higher = stronger setup).
    backtest: score>=6 -> PF ~4.5, WR ~81% on N=255 trades.

    S1 (0-3): RSI higher-low magnitude
    S2 (0-3): divergence position in price range (closer to bottom = better)
    S3 (0-3): prior decline depth
    S4 (0-2): volume quality at the divergence pivot
    """
    if rsi_delta >= 8:   s1 = 3
    elif rsi_delta >= 4: s1 = 2
    elif rsi_delta >= 1: s1 = 1
    else:                s1 = 0

    if div_position <= 0.12:   s2 = 3
    elif div_position <= 0.25: s2 = 2
    elif div_position <= 0.40: s2 = 1
    else:                      s2 = 0

    if decline_pct > 0.35:   s3 = 3
    elif decline_pct > 0.20: s3 = 2
    elif decline_pct > 0.12: s3 = 1
    else:                    s3 = 0

    if vol_qual == "A":   s4 = 2
    elif vol_qual == "B": s4 = 1
    else:                 s4 = 0

    return s1 + s2 + s3 + s4


# ─── REVERSAL DETECTORS ──────────────────────────────────────────────────────
# Each detector takes a DataFrame with at least 65 closed 4H bars and returns
# (fired: bool, info: dict). Pure functions over the LAST closed bar — emit at
# 4H close. Detector parameters and thresholds are backtest-validated; do not alter.

def detect_squeeze_breakout(df4: pd.DataFrame) -> tuple[bool, dict]:
    """Bollinger squeeze + bullish breakout.
    backtest: 149 signals, PF ~1.45, ROI +146%, exp +0.98% (1m exact replay).
    """
    if len(df4) < 65:
        return False, {"reason": "insufficient_bars"}
    closes = df4["Close"]
    bb_mid = closes.rolling(20).mean()
    bb_std = closes.rolling(20).std()
    bb_width = (bb_std / bb_mid).iloc[-1]
    bb_width_med = (bb_std / bb_mid).iloc[-30:].median()
    if pd.isna(bb_width) or pd.isna(bb_width_med):
        return False, {"reason": "nan"}
    if bb_width > bb_width_med * 0.8:
        return False, {"reason": "no_squeeze", "bb_width": float(bb_width), "bb_width_med": float(bb_width_med)}
    last = df4.iloc[-1]
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    last_close = float(last["Close"])
    if last_close <= float(bb_upper) * 0.998:
        return False, {"reason": "no_breakout", "close": last_close, "upper": float(bb_upper)}
    if last_close <= float(last["Open"]):
        return False, {"reason": "not_green"}
    cur_atr = float(df4["atr4h"].iloc[-1]) if "atr4h" in df4.columns else 0.0
    last_low = float(last["Low"])
    return True, {
        "detector": "squeeze_breakout",
        "tag": TAG_SQUEEZE,
        "score": 2,
        "entry": last_close,
        "sl": last_low - 0.5 * cur_atr,
        "tp_r_multiples": [0.9, 1.6, 2.5, 4.0, 6.0],
        "tp_weights": [0.05, 0.10, 0.20, 0.25, 0.40],
        "features": {
            "bb_width": float(bb_width),
            "bb_width_med": float(bb_width_med),
            "atr": cur_atr,
            "atr_pct_4h": cur_atr / last_close if last_close > 0 else 0.0,
            "rsi_4h_now": float(df4["rsi"].iloc[-1]) if "rsi" in df4.columns else None,
        },
    }


def detect_failed_breakdown(df4: pd.DataFrame) -> tuple[bool, dict]:
    """Last-bar break below the recent 60-bar low intrabar but close above (sweep + reclaim).
    backtest: 218 signals, PF ~1.21, ROI +67%, exp +0.31% with tight TP profile.
    """
    if len(df4) < 65:
        return False, {"reason": "insufficient_bars"}
    last = df4.iloc[-1]
    prior = df4.iloc[-61:-1]
    rec_low = float(prior["Low"].min())
    if float(last["Low"]) > rec_low * 0.998:
        return False, {"reason": "no_breakdown", "last_low": float(last["Low"]), "rec_low": rec_low}
    if float(last["Close"]) <= rec_low:
        return False, {"reason": "no_reclaim"}
    if float(last["Close"]) <= float(last["Open"]):
        return False, {"reason": "not_green"}
    cur_atr = float(df4["atr4h"].iloc[-1]) if "atr4h" in df4.columns else 0.0
    body = float(last["Close"]) - float(last["Open"])
    full_range = float(last["High"]) - float(last["Low"])
    body_ratio = body / full_range if full_range > 0 else 0.0
    last_close = float(last["Close"])
    return True, {
        "detector": "failed_breakdown",
        "tag": TAG_FAILED_BD,
        "score": 1 + (1 if body_ratio > 0.3 else 0) + (1 if body_ratio > 0.5 else 0),
        "entry": last_close,
        "sl": float(last["Low"]) - 0.25 * cur_atr,
        "tp_r_multiples": [0.6, 1.2, 2.0, 3.0, 5.0],
        "tp_weights": [0.10, 0.15, 0.20, 0.25, 0.30],
        "features": {
            "rec_low": rec_low,
            "body_ratio": body_ratio,
            "atr": cur_atr,
            "atr_pct_4h": cur_atr / last_close if last_close > 0 else 0.0,
            "rsi_4h_now": float(df4["rsi"].iloc[-1]) if "rsi" in df4.columns else None,
        },
    }


def detect_sweep_reclaim(df4: pd.DataFrame) -> tuple[bool, dict]:
    """Multi-bar sweep + reclaim (last bar low < 30-bar prior low, close above prior low).
    backtest: 140 signals, PF ~1.34, ROI +65%, exp +0.46% with wide TP profile.
    """
    if len(df4) < 65:
        return False, {"reason": "insufficient_bars"}
    last = df4.iloc[-1]
    prior = df4.iloc[-31:-1]
    prior_low = float(prior["Low"].min())
    if float(last["Low"]) >= prior_low:
        return False, {"reason": "no_sweep"}
    if float(last["Close"]) <= prior_low:
        return False, {"reason": "no_reclaim"}
    if float(last["Close"]) <= float(last["Open"]):
        return False, {"reason": "not_green"}
    prev = df4.iloc[-2]
    if float(last["Close"]) <= float(prev["Close"]):
        return False, {"reason": "no_close_lift"}
    cur_atr = float(df4["atr4h"].iloc[-1]) if "atr4h" in df4.columns else 0.0
    body = float(last["Close"]) - float(last["Open"])
    rng = float(last["High"]) - float(last["Low"])
    body_ratio = body / rng if rng > 0 else 0.0
    sweep_depth = (prior_low - float(last["Low"])) / prior_low if prior_low > 0 else 0.0
    last_close = float(last["Close"])
    return True, {
        "detector": "sweep_reclaim",
        "tag": TAG_SWEEP,
        "score": 1 + (1 if body_ratio > 0.4 else 0) + (1 if sweep_depth > 0.005 else 0),
        "entry": last_close,
        "sl": float(last["Low"]) - 0.25 * cur_atr,
        "tp_r_multiples": [1.0, 2.0, 3.0, 5.0, 8.0],
        "tp_weights": [0.10, 0.15, 0.20, 0.25, 0.30],
        "features": {
            "prior_low": prior_low,
            "body_ratio": body_ratio,
            "sweep_depth": sweep_depth,
            "atr": cur_atr,
            "atr_pct_4h": cur_atr / last_close if last_close > 0 else 0.0,
            "rsi_4h_now": float(df4["rsi"].iloc[-1]) if "rsi" in df4.columns else None,
        },
    }


def evaluate_alt_detectors(adapter_name: str, symbol: str, df4: pd.DataFrame) -> list[dict]:
    """Run all enabled reversal detectors and return the list of fired-detector dicts."""
    if df4 is None or len(df4) < 65:
        return []
    fires = []
    detectors = [
        (SQUEEZE_DETECTOR_ENABLED,   detect_squeeze_breakout),
        (FAILED_BD_DETECTOR_ENABLED, detect_failed_breakdown),
        (SWEEP_DETECTOR_ENABLED,     detect_sweep_reclaim),
    ]
    for enabled, fn in detectors:
        try:
            fired, info = fn(df4)
        except Exception as e:
            logger.warning(f"[ALT_DET] {adapter_name} {symbol} {fn.__name__}: {type(e).__name__}: {e!r}")
            continue
        det_name = info.get("detector", fn.__name__)
        if fired:
            if enabled:
                fires.append(info)
        else:
            logger.debug(f"[ALT_DET] {adapter_name} {symbol} {det_name}: not fired ({info.get('reason')})")
    return fires


# ─── ENHANCED 4H SIGNAL DETECTION ────────────────────────────────────────────
async def detect_4h(adapter: ExchangeAdapter, symbol, now_ms):
    df4 = await fetch_4h(adapter, symbol, now_ms)

    df4_with_indicators = compute_all_indicators(df4)
    if df4_with_indicators is None:
        logger.debug(f"[{adapter.name}] {symbol}: insufficient data ({len(df4) if df4 is not None else 0} vs 80) -> skip")
        return ("REJ", "data", None)

    df4 = df4_with_indicators
    current_price = df4["Close"].iloc[-1]
    r2 = float(df4["rsi"].iloc[-1])
    if r2 > RSI_OVERSOLD_MAX:
        return ("REJ", "rsi_high", df4)

    # Adaptive downtrend check
    atr4h = df4["atr4h"].iloc[-1] if "atr4h" in df4.columns else df4["Close"].iloc[-1] * 0.02
    atr_pct = atr4h / current_price

    if atr_pct > 0.05:
        min_decline = 0.15
    elif atr_pct < 0.02:
        min_decline = 0.08
    else:
        min_decline = MIN_DECLINE_FOR_REVERSAL

    lookback_period = min(STRUCTURE_LOOKBACK_BARS, len(df4) - 1)
    recent_high_high = df4["High"].iloc[-lookback_period:].max()
    decline_from_high = (recent_high_high - current_price) / recent_high_high
    recent_swing_high = recent_high_high

    if decline_from_high < min_decline:
        logger.debug(
            f"[{adapter.name}] {symbol}: insufficient decline "
            f"(high={recent_swing_high:.6f} -> current={current_price:.6f}, decline={decline_from_high:.1%} < {min_decline:.0%})"
        )
        return ("REJ", "insufficient_decline", df4)

    # Pivot & divergence (price LL, RSI HL)
    lows = df4["Low"].to_numpy()
    rsi_arr = df4["rsi"].to_numpy()

    w = min(DIV_LB, len(df4) - 1)
    start = len(df4) - (w + 1)
    lows_w = lows[start:]

    pivot_idxs = find_pivot_lows(lows_w, PIVOT_L, PIVOT_R)
    pair = pick_two_pivots(pivot_idxs, lows_w, MIN_PIV_DIST)
    if not pair:
        return ("REJ", "pivots", df4)

    i1, i2 = pair

    p1 = float(lows_w[i1])
    p2 = float(lows_w[i2])

    r1 = float(rsi_arr[start + i1])
    r2 = float(rsi_arr[start + i2])

    price_decline_abs = abs(p1 - p2) / p1
    is_double_bottom = (abs(p2 - p1) / p1) < 0.002

    if not is_double_bottom:
        if p2 > p1:
            return ("REJ", "div_price_not_ll", df4)
        if price_decline_abs < MIN_PRICE_MOVE:
            return ("REJ", "move", df4)
    else:
        if price_decline_abs < (MIN_PRICE_MOVE * 0.5):
            return ("REJ", "move_db_noise", df4)

    rsi_delta = r2 - r1
    if rsi_delta < -0.5:
        return ("REJ", "div_rsi_not_hl", df4)

    # Divergence position: must form in the lower portion of the recent range
    lookback_low = df4["Low"].iloc[-lookback_period:].min()
    range_size = recent_swing_high - lookback_low

    if range_size > 0:
        div_position_in_range = (p2 - lookback_low) / range_size
    else:
        div_position_in_range = 0.0

    if MAX_DIV_POSITION_ENABLED and div_position_in_range > MAX_DIV_POSITION:
        logger.debug(
            f"[{adapter.name}] {symbol}: div not at bottom "
            f"(div={p2:.6f}, range={lookback_low:.6f}-{recent_swing_high:.6f}, "
            f"position={div_position_in_range:.1%} > {MAX_DIV_POSITION:.0%})"
        )
        return ("REJ", "div_not_at_bottom", df4)

    # Volume quality: compare current volume to 10-bar SMA
    try:
        vol_ma = df4["Volume"].iloc[-11:-1].mean()
        vol_cur = df4["Volume"].iloc[-1]
        if vol_cur > vol_ma * 1.5:
            vol_qual = "A"
        elif vol_cur < vol_ma * 0.7:
            vol_qual = "C"
        else:
            vol_qual = "B"
    except Exception:
        vol_qual = "B"

    abs_low = p2
    ts = int(df4["Close_time"].iloc[-1])

    div_score = compute_div_score(rsi_delta, div_position_in_range, decline_from_high, vol_qual)

    logger.info(
        f"[{adapter.name}] {symbol}: REVERSAL SETUP "
        f"Decline:{decline_from_high:.1%} "
        f"Range:[{lookback_low:.6f}-{recent_swing_high:.6f}] "
        f"DivPos:{div_position_in_range:.1%} "
        f"Price:{current_price:.6f}->{abs_low:.6f}(drop {(current_price-abs_low)/current_price:.1%}) "
        f"RSI:{r1:.1f}->{r2:.1f}(delta {rsi_delta:.1f}) Vol:{vol_qual} Score:{div_score}"
    )

    _atr4h_val = float(df4["atr4h"].iloc[-1]) if "atr4h" in df4.columns else 0.0
    _atr_pct_4h_val = (_atr4h_val / current_price) if current_price > 0 else 0.0
    return ("PASS", {
        "div_price": abs_low,
        "ts": ts,
        "vol_quality": vol_qual,
        "atr4h": _atr4h_val,
        "atr_pct_4h": _atr_pct_4h_val,
        "rsi_4h_now": float(r2),
        "rsi_delta": float(rsi_delta),
        "swing_high": float(recent_swing_high),
        "decline_pct": float(decline_from_high),
        "div_position": float(div_position_in_range),
        "div_score": int(div_score),
    }, df4)


# ─── 1H LOGIC HELPERS ────────────────────────────────────────────────────────
def check_confirm(df_closed, state, atr_val: float, adapter_name: str, symbol: str):
    close = float(df_closed["Close"].iloc[-1])
    prev_close = float(df_closed["Close"].iloc[-2]) if len(df_closed) >= 2 else close

    p_low = float(state["p_low"])
    p_high = float(state["p_high"])
    p_close = float(state["p_close"])

    atr_eff = atr_val if (atr_val and np.isfinite(atr_val)) else (p_close * 0.01)
    inv_hard_level = p_low - (1.00 * atr_eff)

    if close < inv_hard_level:
        return "INVALIDATE", f"hard_dump(c={close:.5f} < inv={inv_hard_level:.5f})"

    if len(df_closed) >= 3:
        c1 = float(df_closed["Close"].iloc[-1])
        c2 = float(df_closed["Close"].iloc[-2])
        c3 = float(df_closed["Close"].iloc[-3])

        strikes = sum(1 for c in (c1, c2, c3) if c < p_low)
        if strikes > 0 and DEBUG_CHANNEL:
            logger.debug(f"[{adapter_name}] {symbol}: Strike count={strikes}/3 (pL={p_low:.5f})")

        if c1 < p_low and c2 < p_low and c3 < p_low:
            return "INVALIDATE", "3_strike_low"

    if close > p_high:
        return "CONFIRM", "close>p_high"

    if close > p_close and prev_close > p_close:
        return "CONFIRM", "2x_close>p_close"

    if atr_val and not np.isnan(atr_val):
        if close > p_close + 0.45 * float(atr_val):
            return "CONFIRM", "close>p_close+0.45ATR"

    return "HOLD", ""


def detect_volatility_regime(
    state: dict,
    df1h: pd.DataFrame,
    atr1h: float,
    atr4h: float
) -> str:
    """
    Classify a setup into a volatility/momentum regime.

    Returns:
        "high":   strong momentum, aggressive TPs
        "medium": normal conditions, balanced TPs [DEFAULT]
        "low":    choppy/weak, conservative TPs

    Based on:
      1. ATR expansion ratio (ATR1H/ATR4H): >1.3 = choppy, <0.8 = consolidating
      2. Breakout quality: distance of close above p_high (in ATR units)
    """
    close = float(df1h["Close"].iloc[-1])
    p_high = float(state.get("p_high", close))

    atr_ratio = atr1h / atr4h if (atr4h and atr4h > 0) else 1.0
    breakout_atr = (close - p_high) / atr1h if (atr1h and atr1h > 0) else 0.5

    score = 0

    if atr_ratio > ATR_RATIO_CHOPPY_THRESHOLD:
        score -= 2
    elif atr_ratio > 1.0:
        score -= 1
    elif atr_ratio < ATR_RATIO_CONSOLIDATE_THRESHOLD:
        score += 2
    else:
        score += 1

    if breakout_atr > BREAKOUT_STRONG_ATR:
        score += 2
    elif breakout_atr > 0.8:
        score += 1
    elif breakout_atr < BREAKOUT_WEAK_ATR:
        score -= 1

    if score >= 2:
        return "high"
    elif score <= -1:
        return "low"
    else:
        return "medium"


def parse_weights() -> list[float]:
    ws = [float(x.strip()) for x in LADDER_WEIGHTS.split(",")]
    if len(ws) != LADDER_LEVELS:
        return [1.0 / LADDER_LEVELS] * LADDER_LEVELS
    s = sum(ws)
    return [w / s for w in ws] if s > 0 else [1.0 / LADDER_LEVELS] * LADDER_LEVELS


def parse_exit_weights() -> list[float]:
    ws = [float(x.strip()) for x in EXIT_WEIGHTS.split(",")]
    if len(ws) != EXIT_LEVELS:
        return [1.0 / EXIT_LEVELS] * EXIT_LEVELS
    s = sum(ws)
    return [w / s for w in ws] if s > 0 else [1.0 / EXIT_LEVELS] * EXIT_LEVELS


def _parse_floats_csv(s: str, n: int) -> list[float]:
    xs = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(xs) != n:
        raise ValueError(f"Expected {n} floats, got {len(xs)} from '{s}'")
    return xs


def floor_step(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x / step) * step


def ceil_step(x: float, step: float) -> float:
    if step <= 0: return x
    return math.ceil(x / step) * step


def build_wide_meanrev_ladder(
    ref_price: float,
    div_price: float,
    p_low: float,
    sl: float,
    atr1h: float,
    tick: float,
    levels: int = 5,
    ladder_buffer_atr: float = 0.10,
    ladder_min_gap_atr: float = 0.20,
    entry_span_atr1h: float = 1.2,
    entry_span_risk: float = 0.30,
) -> list[float]:
    """
    Mean-reversion (wide) entry ladder:
      - Returns up to `levels` strictly decreasing tick-rounded prices
      - Enforces minimum ladder depth based on ATR1H and risk-to-SL
      - Never places bids below SL + gap
    """
    if tick <= 0:
        tick = 1e-6
    if atr1h <= 0:
        atr1h = max(ref_price * 0.002, tick * 50)

    high = float(ref_price)

    base_low = max(float(div_price), float(p_low) if p_low is not None else 0.0)
    floor_candidate = base_low + ladder_buffer_atr * atr1h

    min_gap = max(ladder_min_gap_atr * atr1h, tick * 10)
    floor_candidate = max(floor_candidate, float(sl) + min_gap)

    risk = high - float(sl)
    if risk <= 0:
        min_span = entry_span_atr1h * atr1h
    else:
        min_span = max(entry_span_atr1h * atr1h, entry_span_risk * risk)

    floor_target = max(float(sl) + min_gap, high - min_span)

    floor = min(floor_candidate, floor_target)

    if floor >= high - tick * 5:
        floor = high - max(atr1h * 0.5, tick * 50)

    raw = []
    for k in range(levels):
        t = k / (levels - 1) if levels > 1 else 0.0
        t2 = t ** 1.7
        raw.append(high - (high - floor) * t2)

    hard_floor_limit = floor_step(float(sl) + min_gap, tick)

    out: list[float] = []
    prev = None
    for x in raw:
        x_rounded = floor_step(x, tick)
        if prev is None:
            out.append(max(x_rounded, hard_floor_limit))
            prev = out[-1]
            continue

        if x_rounded >= prev:
            x_rounded = prev - tick * 10
            x_rounded = floor_step(x_rounded, tick)

        x_rounded = max(x_rounded, hard_floor_limit)

        out.append(x_rounded)
        prev = x_rounded

    out = sorted(list(set(out)), reverse=True)

    if len(out) < 3:
        safe_floor = max(floor, hard_floor_limit)
        if high > safe_floor + tick * 2:
            mid = floor_step((high + safe_floor) / 2, tick)
            out = [high, mid, safe_floor]
            out = sorted(list(set(out)), reverse=True)
        else:
            return []

    if len(out) < levels:
        logger.warning(f"Ladder compressed: requested {levels}, got {len(out)} -> sending partial")

    return out


def build_tp_ladder_derisk(
    entry_ref: float,
    sl: float,
    atr1h: float,
    tick: float,
    r_mults: list = (0.20, 0.35, 0.60, 0.95, 1.35),
    atr_mults: list = (0.55, 0.85, 1.20, 1.70, 2.40),
    rr_floors: list = (0.15, 0.25, 0.40, 0.65, 0.90),
) -> list[float]:
    """
    5 TPs:
      - risk-based (R-multiples) but capped by ATR1H reachability
      - TP1/TP2 designed for de-risking in low-momentum regimes
    """
    risk = entry_ref - sl
    if tick <= 0: tick = 0.000001

    if atr1h <= 0 or not np.isfinite(atr1h):
        atr1h = max(entry_ref * 0.002, tick * 50)

    min_gap = max(0.12 * atr1h, tick * 5)

    tps: list[float] = []
    prev = entry_ref

    for r_m, a_m, rr_min in zip(r_mults, atr_mults, rr_floors):
        tp_r = entry_ref + r_m * risk
        tp_a = entry_ref + a_m * atr1h
        tp = min(tp_r, tp_a)
        tp = ceil_step(tp, tick)
        if tp <= prev + min_gap:
            tp = ceil_step(prev + min_gap, tick)
        tps.append(tp)
        prev = tp

    return tps


def build_tp_ladder_adaptive(
    entry_ref: float,
    sl: float,
    atr1h: float,
    tick: float,
    swing_high: float,
    regime: str = "medium",
    swing_mult: float = 0.85,
) -> list[float]:
    """
    Adaptive TP ladder based on the volatility/momentum regime.

    Regime profiles:
      - high:   aggressive TPs for strong momentum
      - medium: balanced TPs for normal conditions (DEFAULT)
      - low:    conservative TPs for choppy/weak setups
    """
    risk = entry_ref - sl
    if risk <= 0:
        risk = atr1h if (atr1h and np.isfinite(atr1h)) else (entry_ref * 0.02)

    if tick <= 0:
        tick = 0.000001

    if atr1h <= 0 or not np.isfinite(atr1h):
        atr1h = max(entry_ref * 0.002, tick * 50)

    swing_target = swing_high * swing_mult
    swing_gain = swing_target - entry_ref
    swing_r = swing_gain / risk if risk > 0 else 8.0
    swing_r = min(max(swing_r, 4.0), 15.0)

    # Regime-specific R-multiples (replay-selected: TP1/TP2 widened to avoid fee drag).
    if regime == "high":
        r_mults = [
            1.0,
            1.8,
            3.0,
            max(5.0, 0.75 * swing_r),
            max(7.0, 1.0 * swing_r),
        ]
    elif regime == "low":
        r_mults = [
            0.8,
            1.5,
            2.5,
            max(3.5, 0.50 * swing_r),
            max(5.0, 0.75 * swing_r),
        ]
    else:  # medium (DEFAULT)
        r_mults = [
            0.9,
            1.6,
            2.5,
            max(4.0, 0.65 * swing_r),
            max(6.0, 0.90 * swing_r),
        ]

    tps: list[float] = []
    prev = entry_ref
    min_gap = max(0.15 * atr1h, tick * 5)

    for r_m in r_mults:
        tp = entry_ref + r_m * risk
        tp = ceil_step(tp, tick)
        if tp <= prev + min_gap:
            tp = ceil_step(prev + min_gap, tick)
        tps.append(tp)
        prev = tp

    return tps


def _resolve_cell_profile(detector: str) -> tuple[str, str, str, str, str]:
    """Return (btc_regime, cell_key, r_name, w_name, tp_profile_id)."""
    btc_regime = _btc_regime_label(_btc_ret_7d)
    cell_key = f"{detector}|{btc_regime}"
    if REGIME_ADAPTIVE_TP_ENABLED and cell_key in _REGIME_TP_WINNERS:
        chosen = _REGIME_TP_WINNERS[cell_key]["chosen_profile"]
        r_name, w_name = chosen.split("|")
        return btc_regime, cell_key, r_name, w_name, chosen
    r_name, w_name = _ALT_DETECTOR_DEFAULT_PROFILES.get(detector, ("R_default", "W_runner"))
    return btc_regime, cell_key, r_name, w_name, f"{r_name}|{w_name}"


# ─── ALT DETECTOR SIGNAL SENDER ──────────────────────────────────────────────
async def send_alt_signal(
    adapter: ExchangeAdapter, symbol: str, det_info: dict, now_ms: int
) -> bool:
    """Fire a signal for a reversal-detector candidate (squeeze / failed_bd / sweep).

    Returns True if the signal was sent, False if rejected.
    """
    exchange   = adapter.name
    detector   = det_info["detector"]
    tag_internal = det_info.get("tag") or TAG_SQUEEZE
    entry_ref  = float(det_info["entry"])
    sl         = float(det_info["sl"])
    features    = det_info.get("features", {})
    atr_pct_4h  = float(features.get("atr_pct_4h", 0.0))
    vol_30d_usd = det_info.get("vol_30d_usd")
    cohort      = det_info.get("cohort") or _classify_cohort(vol_30d_usd)
    now_utc     = datetime.now(timezone.utc)
    _wd         = now_utc.weekday()

    # ── Reject filters ───────────────────────────────────────────────────────
    if symbol in SIGNAL_BLACKLIST:
        logger.info(f"[{exchange}] {symbol}: {detector} rejected (SIGNAL_BLACKLIST)")
        return False

    if vol_30d_usd is not None and float(vol_30d_usd) < MIN_VOL_30D_USD:
        logger.info(f"[{exchange}] {symbol}: {detector} rejected "
                    f"(vol_30d_usd ${float(vol_30d_usd)/1e6:.1f}M < ${MIN_VOL_30D_USD/1e6:.1f}M floor)")
        return False

    if (_wd == 2 and DAY_FILTER_WED_ENABLED) or (_wd == 5 and DAY_FILTER_SAT_ENABLED):
        logger.info(f"[{exchange}] {symbol}: {detector} suppressed (day filter {now_utc.strftime('%A')} UTC)")
        return False

    if DOW_HOUR_BLACKLIST_ENABLED and (_wd, now_utc.hour) in DOW_HOUR_BLACKLIST:
        logger.info(f"[{exchange}] {symbol}: {detector} suppressed (dow_hour_blacklist wd={_wd} h={now_utc.hour})")
        return False

    if sent_signals.in_cooldown(symbol, now_ms):
        logger.info(f"[{exchange}] {symbol}: {detector} suppressed (cooldown {SIGNAL_COOLDOWN_H}h)")
        return False

    if exchange.upper() == "BITGET":
        _last = sent_signals.get_last(symbol)
        if _last and _last.get("exchange") == "BINANCE":
            _age = (now_ms - int(_last.get("ts_ms", 0))) / 1000.0
            if _age <= BINANCE_PRIORITY_WINDOW_SEC:
                logger.info(f"[BITGET] {symbol}: {detector} suppressed (Binance priority {_age:.0f}s ago)")
                return False
        _binance_book = market_state.books.get(f"BINANCE:{symbol}")
        if _binance_book and not _binance_book.get("shadow_only"):
            logger.info(f"[BITGET] {symbol}: {detector} suppressed (Binance active setup)")
            return False

    # ── Cohort filter ────────────────────────────────────────────────────────
    if COHORT_FILTER_ENABLED and (detector, cohort) in DROP_COHORT_CELLS:
        logger.info(f"[{exchange}] {symbol}: {detector} rejected (cohort={cohort})")
        return False

    # ── BTC regime + per-cell TP profile ─────────────────────────────────────
    btc_regime, cell_key, r_name, w_name, tp_profile_id = _resolve_cell_profile(detector)

    tp_r      = TP_R_PROFILES[r_name]
    tp_w      = TP_W_PROFILES[w_name]
    risk      = max(entry_ref - sl, entry_ref * 0.001)
    tp_prices = [entry_ref + r * risk for r in tp_r]

    # Freshness guard: skip if the latest closed 1H bar already traded through TP1 or SL.
    try:
        _raw_1h = await adapter.get_klines(symbol, "1h", limit=4)
        _df_1h = pd.DataFrame(_raw_1h, columns=["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time"])
        for _c in ["Open", "High", "Low", "Close", "Volume"]:
            _df_1h[_c] = _df_1h[_c].astype(float)
        _df_1h["Open_time"] = _df_1h["Open_time"].astype("int64")
        _df_1h["Close_time"] = _df_1h["Close_time"].astype("int64")
        _df_1h = drop_live_candle(_df_1h, now_ms, SAFETY_1H_SEC)
        if not _df_1h.empty:
            _last_high = float(_df_1h["High"].iloc[-1])
            _last_low = float(_df_1h["Low"].iloc[-1])
            if _last_high >= float(tp_prices[0]) or _last_low <= float(sl):
                logger.info(
                    f"[{exchange}] {symbol}: {detector} stale-before-send "
                    f"(last_1h_high={_last_high:.6f} tp1={float(tp_prices[0]):.6f} "
                    f"last_1h_low={_last_low:.6f} sl={float(sl):.6f})"
                )
                return False
    except Exception as _fresh_err:
        logger.warning(f"[{exchange}] {symbol}: freshness guard failed ({type(_fresh_err).__name__}): {_fresh_err!r}")

    # ── Build signal text ─────────────────────────────────────────────────────
    def _fmt(x: float) -> str:
        if x < 0.00001: return f"{x:.9f}"
        if x < 0.001:   return f"{x:.8f}"
        return f"{x:.6f}"

    # Multi-entry mean-reversion ladder for squeeze only. The deepest rung is nudged
    # to -0.98R (just above SL) to keep all entries strictly above the stop.
    if tag_internal == TAG_SQUEEZE:
        _ladder = [(0.0, 0.20), (-0.5, 0.30), (-0.98, 0.50)]
        entry_line = ", ".join(
            f"{_fmt(entry_ref + off * risk)} ({int(round(w * 100))}%)"
            for off, w in _ladder
        )
    else:
        entry_line = f"{_fmt(entry_ref)} (100%)"
    tp_line    = ", ".join(f"{_fmt(tp)} ({int(round(w * 100))}%)" for tp, w in zip(tp_prices, tp_w))
    sl_line    = f"2H close below {_fmt(sl)}"

    text_internal = (
        f"{tag_internal}\n"
        f"Pair: {symbol}\n"
        f"Entry: {entry_line}\n"
        f"Targets: {tp_line}\n"
        f"Stop: {sl_line}"
    )
    tg_tag       = _telegram_display_tag(tag_internal)
    text_display = text_internal.replace(tag_internal, tg_tag, 1)

    logger.info(
        f"[{exchange}] {symbol}: ALT FIRED det={detector} "
        f"entry={_fmt(entry_ref)} sl={_fmt(sl)} profile={tp_profile_id} regime={btc_regime}"
    )

    try:
        await send_telegram(text_display, exchange=exchange, symbol=symbol, log_text=text_internal)
        sent_signals.mark_sent(symbol, exchange.upper(), now_ms)
        sent_signals.flush()
        return True
    except Exception as e:
        logger.error(f"[{exchange}] {symbol}: alt signal TG fail ({type(e).__name__}): {e!r}", exc_info=True)
        return False


# ─── MTF 1H CONFIRM GATE ─────────────────────────────────────────────────────
def _register_mtf_pending(adapter, symbol: str, det_info: dict, now_ms: int) -> None:
    """Register a reversal-detector candidate in the 1H MTF confirm window."""
    exchange = adapter.name
    key = f"{exchange}:{symbol}"

    if sent_signals.in_cooldown(symbol, now_ms):
        logger.info(f"[MTF] {symbol}: dropped (cooldown active)")
        return

    if key in _mtf_pending:
        logger.info(f"[MTF] {symbol}: dropped (already waiting)")
        return

    detector = det_info["detector"]
    _mtf_pending[key] = {
        "symbol": symbol,
        "exchange": exchange,
        "det_info": det_info,
        "registered_ts": time.time(),
    }
    _save_mtf_pending()
    logger.info(f"[MTF] {symbol}: PENDING — 1H confirm window open (det={detector})")


async def process_mtf_pending(adapters: list) -> None:
    """On each hourly tick: check 1H bars for each pending candidate; fire or expire."""
    if not _mtf_pending:
        return

    adapter_map = {a.name: a for a in adapters}
    now_ts  = time.time()
    now_ms  = int(now_ts * 1000)
    to_remove: list = []

    for key, entry in list(_mtf_pending.items()):
        symbol        = entry["symbol"]
        exchange      = entry["exchange"]
        det_info      = entry["det_info"]
        registered_ts = float(entry["registered_ts"])
        hours_elapsed = (now_ts - registered_ts) / 3600.0

        adapter = adapter_map.get(exchange)
        if not adapter:
            continue

        try:
            raw = await adapter.get_klines(symbol, "1h", limit=10)
            df = pd.DataFrame(raw, columns=["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time"])
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = df[c].astype(float)
            df["Close_time"] = df["Close_time"].astype("int64")
            reg_ms   = int(registered_ts * 1000)
            df_after = df[(df["Close_time"] > reg_ms) & (df["Close_time"] <= (now_ms - CLOSE_BUFFER_MS))]
            df_after = drop_live_candle(df_after.reset_index(drop=True), now_ms, SAFETY_1H_SEC)
        except Exception as e:
            logger.debug(f"[MTF] {symbol}: 1H fetch failed: {e!r}")
            continue

        bars_1h = [
            {"open": float(r["Open"]), "high": float(r["High"]),
             "low": float(r["Low"]), "close": float(r["Close"])}
            for _, r in df_after.iterrows()
        ]

        confirmed, pattern, bar_idx = _check_1h_confirmation(bars_1h, max_bars=MTF_MAX_WINDOW_H)

        if confirmed:
            elapsed_s = now_ts - registered_ts
            sent = await send_alt_signal(adapter, symbol, det_info, now_ms)
            if sent:
                to_remove.append(key)
                logger.info(
                    f"[MTF] {symbol}: CONFIRMED pattern={pattern} bar={bar_idx} "
                    f"elapsed={elapsed_s/3600:.1f}h det={det_info['detector']}"
                )
            else:
                logger.info(
                    f"[MTF] {symbol}: confirmed but not sent (det={det_info['detector']})"
                )
        elif hours_elapsed >= float(MTF_MAX_WINDOW_H):
            to_remove.append(key)
            logger.info(
                f"[MTF] {symbol}: EXPIRED {hours_elapsed:.1f}h no confirm det={det_info['detector']}"
            )

    for key in to_remove:
        _mtf_pending.pop(key, None)
    if to_remove:
        _save_mtf_pending()


# ─── CLASSIC DIVERGENCE LADDER SIGNAL ─────────────────────────────────────────
async def send_ladder_signal(adapter: ExchangeAdapter, symbol: str, state: dict, df1h: pd.DataFrame) -> bool:
    exchange = adapter.name
    close = float(df1h["Close"].iloc[-1])

    if symbol in SIGNAL_BLACKLIST:
        logger.info(f"[{exchange}] {symbol}: rejected (SIGNAL_BLACKLIST)")
        return False

    _now_utc = datetime.now(timezone.utc)
    _wd = _now_utc.weekday()
    if (_wd == 2 and DAY_FILTER_WED_ENABLED) or (_wd == 5 and DAY_FILTER_SAT_ENABLED):
        logger.info(
            f"[{exchange}] {symbol}: signal suppressed "
            f"(day filter: {_now_utc.strftime('%A')} UTC, weak-performance day)"
        )
        return False

    if DOW_HOUR_BLACKLIST_ENABLED and (_wd, _now_utc.hour) in DOW_HOUR_BLACKLIST:
        logger.info(
            f"[{exchange}] {symbol}: signal suppressed "
            f"(dow_hour_blacklist: weekday={_wd} hour={_now_utc.hour} UTC)"
        )
        return False

    if ATR_PCT_4H_MIN_ENABLED:
        _atr4h_pct = float(state.get("atr_pct_4h", 0.0) or 0.0)
        if _atr4h_pct < ATR_PCT_4H_MIN:
            logger.info(
                f"[{exchange}] {symbol}: rejected (atr_pct_4h={_atr4h_pct:.4f} < {ATR_PCT_4H_MIN})"
            )
            return False

    if BTC_GATE_ENABLED and not _btc_regime_ok:
        age_min = (time.time() - _btc_regime_ts) / 60.0
        logger.info(
            f"[{exchange}] {symbol}: suppressed (BTC bear regime active, updated {age_min:.0f}m ago)"
        )
        return False

    now_ms = int(time.time() * 1000)
    if sent_signals.in_cooldown(symbol, now_ms):
        last = sent_signals.get_last(symbol)
        logger.info(f"[{exchange}] {symbol}: suppressed by cooldown ({SIGNAL_COOLDOWN_H}h). Last={last}")
        return False

    if exchange.upper() == "BITGET":
        last = sent_signals.get_last(symbol)
        if last and last.get("exchange") == "BINANCE":
            age_sec = (now_ms - int(last.get("ts_ms", 0))) / 1000.0
            if age_sec <= BINANCE_PRIORITY_WINDOW_SEC:
                logger.info(f"[BITGET] {symbol}: suppressed (Binance priority, sent {age_sec:.0f}s ago)")
                return False

        binance_key = f"BINANCE:{symbol}"
        _bk = market_state.books.get(binance_key)
        if _bk and not _bk.get("shadow_only"):
            logger.info(f"[BITGET] {symbol}: suppressed (Binance has active setup)")
            return False

    div_score = int(state.get("div_score", 5))
    if MIN_DIV_SCORE_ENABLED and div_score < MIN_DIV_SCORE:
        logger.info(
            f"[{exchange}] {symbol}: rejected (div_score={div_score} < MIN_DIV_SCORE={MIN_DIV_SCORE})"
        )
        return False

    vol_qual = state.get("vol_quality", "B")
    if vol_qual == "C":
        logger.info(f"[{exchange}] {symbol}: rejected (weak volume, grade C)")
        return False

    atr1h = float(df1h["atr"].iloc[-1]) if "atr" in df1h else 0.0
    atr4h = float(state.get("atr4h", 0.0) or 0.0)

    div_price = float(state["div_price"])
    ref_price = close

    p_low = float(state.get("p_low", div_price))
    sl_anchor = p_low

    if atr1h and np.isfinite(atr1h) and atr1h > 0:
        sl = sl_anchor - SL_BUFFER_ATR * atr1h
    elif atr4h and np.isfinite(atr4h) and atr4h > 0:
        sl = sl_anchor - (SL_BUFFER_ATR * 0.5) * atr4h
    else:
        sl = sl_anchor * 0.98

    sl = min(sl, float(div_price) * 0.99)

    tick_sz = await adapter.get_tick_size(symbol)
    if tick_sz <= 0: tick_sz = 0.000001

    ladder_prices = build_wide_meanrev_ladder(
        ref_price=ref_price,
        div_price=div_price,
        p_low=p_low,
        sl=sl,
        atr1h=atr1h,
        tick=tick_sz,
        levels=LADDER_LEVELS,
        ladder_buffer_atr=LADDER_BUFFER_ATR,
        ladder_min_gap_atr=LADDER_MIN_GAP_ATR
    )

    if not ladder_prices:
        logger.warning(f"[{exchange}] {symbol}: LADDER ABORTED (compressed grid)")
        return False

    ladder_top = ladder_prices[0]
    ladder_bot = ladder_prices[-1]
    depth_pct = (ladder_top - ladder_bot) / ladder_top if ladder_top > 0 else 0
    risk_dist = ref_price - sl
    logger.info(
        f"[{exchange}] {symbol} LADDER: Atr1H={atr1h:.6f} Risk={risk_dist:.6f} "
        f"Top={ladder_top:.6f} Bot={ladder_bot:.6f} Depth={depth_pct:.2%} Tick={tick_sz:.8f}"
    )

    entry_weights = parse_weights()

    state["sl"] = float(sl)
    state["sl_anchor"] = float(sl_anchor)
    state["atr1h_at_signal"] = float(atr1h) if atr1h and not np.isnan(atr1h) else None
    state["atr4h_at_setup"] = float(atr4h) if atr4h and np.isfinite(atr4h) else None

    try:
        entry_ref = float(np.average(ladder_prices, weights=entry_weights))
    except Exception:
        entry_ref = ref_price
    state["entry_ref"] = float(entry_ref)

    def _fmt_price(x: float) -> str:
        if x < 0.00001: return f"{x:.9f}"
        if x < 0.001: return f"{x:.8f}"
        return f"{x:.6f}"

    def _fmt_pct(w: float) -> str:
        return f"{int(round(w*100))}%"

    # Hard-cap SL distance at MAX_SL_RISK_PCT of entry (backtest-validated).
    if entry_ref > 0 and (entry_ref - sl) / entry_ref > MAX_SL_RISK_PCT:
        sl_pre_cap = sl
        sl = entry_ref * (1.0 - MAX_SL_RISK_PCT)
        logger.info(
            f"[{exchange}] {symbol}: SL capped "
            f"{(entry_ref - sl_pre_cap)/entry_ref:.1%} -> {MAX_SL_RISK_PCT:.0%} "
            f"({_fmt_price(sl_pre_cap)} -> {_fmt_price(sl)})"
        )
        state["sl"] = float(sl)

    swing_high = float(state.get("swing_high", ref_price * 1.10))

    if USE_ADAPTIVE_TPS:
        regime = detect_volatility_regime(state, df1h, atr1h, atr4h)
    else:
        regime = "medium"

    exit_prices = build_tp_ladder_adaptive(
        entry_ref=entry_ref,
        sl=sl,
        atr1h=atr1h,
        tick=tick_sz,
        swing_high=swing_high,
        regime=regime,
        swing_mult=TP_SWING_MULT
    )

    atr_ratio = atr1h / atr4h if (atr4h and atr4h > 0) else 1.0
    adaptive_status = "ADAPTIVE" if USE_ADAPTIVE_TPS else "FIXED"
    logger.info(
        f"[{exchange}] {symbol}: {adaptive_status} Regime={regime.upper()} "
        f"ATR_Ratio={atr_ratio:.2f} Entry={entry_ref:.6f} Risk={(entry_ref-sl)/entry_ref:.1%}"
    )

    if not exit_prices or len(exit_prices) < 3:
        logger.warning(f"[{exchange}] {symbol}: TP ladder build failed, aborting signal")
        return False

    last_1h_high = float(df1h["High"].iloc[-1])
    last_1h_low = float(df1h["Low"].iloc[-1])
    if last_1h_high >= float(exit_prices[0]) or last_1h_low <= float(sl):
        logger.info(
            f"[{exchange}] {symbol}: stale signal suppressed "
            f"(last_1h_high={last_1h_high:.6f} tp1={float(exit_prices[0]):.6f} "
            f"last_1h_low={last_1h_low:.6f} sl={float(sl):.6f})"
        )
        return False

    entry_items = ", ".join(
        [f"{_fmt_price(p)} ({_fmt_pct(w)})" for p, w in zip(ladder_prices, entry_weights)]
    )

    exit_weights_list = parse_exit_weights()
    tp_items = ", ".join(
        [f"{_fmt_price(p)} ({_fmt_pct(w)})" for p, w in zip(exit_prices, exit_weights_list)]
    )

    text = (
        f"{TAG}\n"
        f"Pair: {symbol}\n"
        f"Entry: {entry_items}\n"
        f"Targets: {tp_items}\n"
        f"Stop: 2H close below {_fmt_price(sl)}"
    )

    if ALERT_ONLY_A_GRADE and state.get("grade") == "B":
        logger.info(f"[{exchange}] {symbol}: confirmed but suppressed (grade B)")
        return False

    text_internal = text.replace(TAG, TAG_CLASSIC_DIV, 1)

    if not CLASSIC_DIV_ENABLED:
        logger.info(f"[{exchange}] {symbol}: classic_bull_div skipped (CLASSIC_DIV_ENABLED=0)")
        state["ladder_sent_ts"] = now_ms
        return True

    try:
        await send_telegram(text, exchange=exchange, symbol=symbol, log_text=text_internal)
        state["ladder_sent_ts"] = now_ms
        sent_signals.mark_sent(symbol, exchange.upper(), now_ms)
        sent_signals.flush()
        return True
    except Exception as e:
        logger.error(
            f"[{exchange}] {symbol}: Telegram fail ({type(e).__name__}): {e!r}",
            exc_info=True,
        )
        return False


# ─── BTC REGIME CHECK ────────────────────────────────────────────────────────
async def update_btc_regime(adapter: ExchangeAdapter) -> None:
    """
    Fetch recent BTCUSDT 1H data and update the global _btc_regime_ok flag.

    BTC 7-day (168 bar) return <= BTC_7D_BEAR_THRESHOLD (-15%) -> suppress alt signals.
    On any data failure: defaults to True (non-blocking).
    """
    global _btc_regime_ok, _btc_regime_ts, _btc_ret_7d

    # Refresh _btc_ret_7d for regime classification even when BTC_GATE is OFF.
    try:
        raw = await adapter.get_klines("BTCUSDT", "1h", limit=172)
        if raw and len(raw) >= 168:
            _closes = [float(r[4]) for r in raw]
            if _closes[-168] > 0:
                _btc_ret_7d = (_closes[-1] - _closes[-168]) / _closes[-168]
    except Exception:
        pass

    if not BTC_GATE_ENABLED:
        return

    try:
        raw = await adapter.get_klines("BTCUSDT", "1h", limit=172)
        if not raw or len(raw) < 168:
            logger.debug(f"[BTC_GATE] Insufficient BTC data ({len(raw) if raw else 0} bars) — gate stays {_btc_regime_ok}")
            return

        closes = [float(r[4]) for r in raw]
        close_now  = closes[-1]
        close_7d   = closes[-168]
        ret_7d     = (close_now - close_7d) / close_7d if close_7d > 0 else 0.0
        _btc_ret_7d = ret_7d

        was_ok = _btc_regime_ok
        _btc_regime_ok = ret_7d > BTC_7D_BEAR_THRESHOLD
        _btc_regime_ts = time.time()

        if not _btc_regime_ok:
            logger.warning(
                f"[BTC_GATE] BEAR REGIME — BTC 7D return={ret_7d:.1%} <= {BTC_7D_BEAR_THRESHOLD:.0%}. "
                f"Alt signals SUPPRESSED."
            )
        elif not was_ok and _btc_regime_ok:
            logger.info(
                f"[BTC_GATE] Bear regime cleared — BTC 7D return={ret_7d:.1%}. Alt signals RESUMED."
            )
        else:
            logger.debug(f"[BTC_GATE] OK — BTC 7D return={ret_7d:.1%}")

    except Exception as e:
        logger.warning(
            f"[BTC_GATE] Failed to fetch BTC data ({type(e).__name__}): {e!r} — gate stays {_btc_regime_ok}",
            exc_info=True,
        )


# ─── 1H TICK PROCESSOR (ARMED BOOK) ──────────────────────────────────────────
async def process_hourly_tick(adapter: ExchangeAdapter):
    keys = [k for k in market_state.get_active_keys() if k.startswith(f"{adapter.name}:")]
    if not keys:
        return 0, 0, 0, 0

    s_armed = s_conf = s_exp = s_inv = 0
    now_ms = await safe_server_time_ms(adapter)

    async def process_single(key):
        nonlocal s_armed, s_conf, s_exp, s_inv

        symbol = key.split(":")[1]
        state = market_state.books[key]

        try:
            bars = await adapter.get_klines(symbol, "1h", limit=50)

            df = pd.DataFrame(bars, columns=["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time"])
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = df[c].astype(float)
            df["Open_time"]  = df["Open_time"].astype("int64")
            df["Close_time"] = df["Close_time"].astype("int64")
            df["ct"]         = df["Close_time"]

            df_closed = df[df["ct"] <= (now_ms - CLOSE_BUFFER_MS)].reset_index(drop=True)
            df_closed = drop_live_candle(df_closed, now_ms, SAFETY_1H_SEC)

            if len(df_closed) < 14:
                return

            df_closed["rsi"] = rsi(df_closed["Close"], 14)
            df_closed["atr"] = atr(df_closed, 14)

            last = df_closed.iloc[-1]
            last_close = float(last["Close"])

            hours_since_setup = (now_ms - state["setup_time"]) / (3600 * 1000)

            if state["status"] == "WAITING":
                if hours_since_setup > MAX_SETUP_TTL_H:
                    logger.info(f"[{adapter.name}] {symbol}: EXPIRED setup (TTL {MAX_SETUP_TTL_H}h)")
                    market_state.remove(key)
                    s_exp += 1
                    return

                if hours_since_setup > WAIT_PATTERN_TTL_H:
                    logger.info(f"[{adapter.name}] {symbol}: EXPIRED waiting-for-pattern ({WAIT_PATTERN_TTL_H}h)")
                    market_state.remove(key)
                    s_exp += 1
                    return

                pat, pat_idx = detect_pattern(df_closed, state["setup_time"])
                if pat:
                    curr = df_closed.iloc[pat_idx]

                    dist_from_div = (float(curr["Low"]) - float(state["div_price"])) / float(state["div_price"])

                    if dist_from_div > RETEST_ZONE_PCT:
                        logger.info(
                            f"[{adapter.name}] {symbol}: IGNORED {pat} "
                            f"(not retest: pLow={float(curr['Low']):.6f} vs div={float(state['div_price']):.6f}, "
                            f"dist=+{dist_from_div:.1%} > {RETEST_ZONE_PCT:.0%})"
                        )
                        return

                    state.update({
                        "status": "ARMED",
                        "armed_time": int(curr["Close_time"]),
                        "bars_waited": 0,
                        "pattern": pat,
                        "p_high": float(curr["High"]),
                        "p_low": float(curr["Low"]),
                        "p_close": float(curr["Close"]),
                        "grade": volume_grade(df_closed, pat_idx),
                        "ladder_sent": False,
                    })

                    if DEBUG_CHANNEL:
                        logger.info(f"[{adapter.name}] {symbol}: ARMED {pat} grade={state['grade']} pH={state['p_high']:.6g}")
                    else:
                        logger.info(f"[{adapter.name}] {symbol}: ARMED {pat} (grade {state['grade']})")

                    s_armed += 1

            elif state["status"] == "ARMED":
                state["bars_waited"] += 1

                if state["bars_waited"] > MAX_CONFIRM_BARS:
                    logger.info(f"[{adapter.name}] {symbol}: EXPIRED armed setup (waited {state['bars_waited']}h)")
                    market_state.remove(key)
                    s_exp += 1
                    return

                atr_val = float(df_closed["atr"].iloc[-1])
                decision, reason = check_confirm(df_closed, state, atr_val, adapter.name, symbol)

                if DEBUG_CHANNEL:
                    c_p = float(df_closed["Close"].iloc[-1])
                    p_L = float(state["p_low"])
                    p_H = float(state["p_high"])
                    p_C = float(state["p_close"])
                    waited = state["bars_waited"]

                    inv_lev = p_L - 0.15 * (atr_val if atr_val > 0 else c_p * 0.01)
                    target = p_C + 0.45 * (atr_val if atr_val > 0 else 0)

                    logger.debug(
                        f"[{adapter.name}] {symbol} CHECK: Close={c_p:.5f} pLow={p_L:.5f} pHigh={p_H:.5f} "
                        f"Inv={inv_lev:.5f} Target={target:.5f} Waited={waited} Dec={decision}"
                    )

                if decision == "INVALIDATE":
                    logger.info(f"[{adapter.name}] {symbol}: INVALIDATED {reason}")
                    market_state.remove(key)
                    s_inv += 1
                    return

                elif decision == "CONFIRM":
                    if not state.get("ladder_sent", False):
                        state["confirm_reason"] = reason
                        state["confirm_close"] = float(df_closed["Close"].iloc[-1])
                        state["confirm_time_ms"] = int(df_closed["Close_time"].iloc[-1])
                        sent = await send_ladder_signal(adapter, symbol, state, df_closed)
                        if sent:
                            state["ladder_sent"] = True
                            logger.info(f"[{adapter.name}] {symbol}: LADDER SENT ({reason})")
                            s_conf += 1
                            market_state.remove(key)
                            return
                        else:
                            logger.info(f"[{adapter.name}] {symbol}: Signal NOT sent (Wait/Retry)")

        except Exception as e:
            logger.error(
                f"[{adapter.name}] {symbol}: hourly process error ({type(e).__name__}): {e!r}",
                exc_info=True,
            )

    logger.info(f"[{adapter.name}] Processing {len(keys)} active setups...")
    for i in range(0, len(keys), 20):
        if i > 0:
            now_ms = await safe_server_time_ms(adapter)
        batch = keys[i:i + 20]
        tasks = [process_single(k) for k in batch]
        await asyncio.gather(*tasks)

    logger.info(f"[{adapter.name}] Hourly Tick Summary: Armed={s_armed}, Confirmed={s_conf}, Expired={s_exp}, Invalid={s_inv}, TotalWaited={len(keys)}")
    return s_armed, s_conf, s_exp, s_inv


# ─── MARKET SCAN ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10) + wait_random(RETRY_JITTER_MIN_SEC, RETRY_JITTER_MAX_SEC),
)
async def scan_markets(adapter: ExchangeAdapter):
    info_syms = await adapter.get_trading_symbols()
    tick = await adapter.get_tickers()

    vols = {}
    if adapter.name == "BINANCE":
        vols = {t["symbol"]: float(t["quoteVolume"]) for t in tick if "symbol" in t and "quoteVolume" in t}
    elif adapter.name == "BITGET":
        if tick and not getattr(scan_markets, "_bitget_keys_logged", False):
            logger.warning(f"[BITGET] ticker keys sample: {sorted(list(tick[0].keys()))}")
            scan_markets._bitget_keys_logged = True

        for t in tick:
            v = t.get("usdtVolume") or t.get("quoteVolume") or t.get("baseVolume") or 0
            try:
                vols[t["symbol"]] = float(v)
            except Exception:
                vols[t["symbol"]] = 0.0

        vols_norm = {}
        for k, v in vols.items():
            vols_norm[adapter._norm_symbol(k)] = max(vols_norm.get(adapter._norm_symbol(k), 0.0), float(v))
        vols = vols_norm

    min_vol = MIN_VOL_BITGET if adapter.name.upper().startswith("BITGET") else MIN_VOL_BINANCE
    filt = [s for s in info_syms if vols.get(s, 0.0) >= min_vol]
    filt = await _filter_universe_intersection(filt, adapter.name.upper())

    logger.info(f"[{adapter.name}] Universe volume filter: min={min_vol:.0f}")
    logger.info(f"[{adapter.name}] Universe: {len(info_syms)} total -> {len(filt)} after filter")

    if filt:
        filt_vols = [vols.get(s, 0.0) for s in filt]
        if filt_vols:
            p0 = min(filt_vols)
            p50 = float(np.median(filt_vols))
            p90 = float(np.percentile(filt_vols, 90))
            logger.info(f"[{adapter.name}] Volume Stats: Min={p0:.0f} Median={p50:.0f} P90={p90:.0f}")

    if len(filt) < 50:
        logger.warning(f"[{adapter.name}] Universe small after vol filter: {len(filt)} symbols. Check ticker symbol mapping/volume keys.")
        if len(vols) > 0:
            logger.warning(f" Sample vol keys: {list(vols.keys())[:5]}")

    return sorted(filt, key=lambda s: vols.get(s, 0.0), reverse=True)[:TOP_N]


# ─── TIME HELPER ──────────────────────────────────────────────────────────────
async def safe_server_time_ms(adapter: ExchangeAdapter) -> int:
    try:
        st = await adapter.get_server_time()
        if isinstance(st, (str, int, float)):
            return int(st)

        if isinstance(st, dict):
            for k in ("serverTime", "server_time", "ts", "timestamp", "time"):
                if k in st:
                    return int(st[k])
            if "data" in st:
                d = st["data"]
                if isinstance(d, (str, int, float)):
                    return int(d)
                if isinstance(d, dict):
                    for k in ("serverTime", "server_time", "ts", "timestamp", "time"):
                        if k in d:
                            return int(d[k])

        return int(time.time() * 1000)

    except Exception as e:
        logger.warning(
            f"[{adapter.name}] get_server_time failed ({type(e).__name__}): {e!r}; falling back to local time",
            exc_info=True,
        )
        return int(time.time() * 1000)


# ─── 4H CYCLE ─────────────────────────────────────────────────────────────────
async def cycle(adapter: ExchangeAdapter):
    """
    4H cycle:
      1. Scan markets
      2. Run reversal detectors + 4H divergence detection
      3. Register/refresh setups in MarketState
    """
    t0 = time.time()

    rejects = {
        "rsi_high": 0,
        "pivots": 0,
        "div_price_not_ll": 0,
        "move": 0,
        "div_rsi_not_hl": 0,
        "trend": 0,
        "data": 0,
        "insufficient_decline": 0,
        "div_not_at_bottom": 0,
    }

    samples = {k: [] for k in rejects.keys()}
    samples["other"] = []

    def sample_push(key, msg):
        if len(samples.get(key, [])) < 5:
            if key not in samples: samples[key] = []
            samples[key].append(msg)

    syms = await scan_markets(adapter)
    now_ms = await safe_server_time_ms(adapter)

    logger.info(f"[{adapter.name}] Starting 4H scan of {len(syms)} symbols")

    async def detect_symbol(symbol: str):
        try:
            res_tuple = await detect_4h(adapter, symbol, now_ms)
            # Evaluate reversal detectors; route fires to send path or MTF gate.
            # Reuse df4 from detect_4h (3rd tuple element) — no second network fetch.
            if SQUEEZE_DETECTOR_ENABLED or FAILED_BD_DETECTOR_ENABLED or SWEEP_DETECTOR_ENABLED:
                try:
                    _df4_alt_with = res_tuple[2] if (res_tuple and len(res_tuple) >= 3) else None
                    if _df4_alt_with is not None:
                        _alt_fires = evaluate_alt_detectors(adapter.name, symbol, _df4_alt_with)
                        _vol_30d = _compute_vol_30d_usd(_df4_alt_with)
                        for _fire in _alt_fires:
                            if symbol in SIGNAL_BLACKLIST:
                                logger.info(f"[{adapter.name}] {symbol}: {_fire['detector']} rejected (SIGNAL_BLACKLIST)")
                                continue
                            _fire["vol_30d_usd"] = _vol_30d
                            _fire["cohort"] = _classify_cohort(_vol_30d)
                            if _vol_30d is None:
                                logger.debug(f"[{adapter.name}] {symbol}: COHORT_UNKNOWN (insufficient 4H history)")
                            if MTF_1H_CONFIRM_ENABLED:
                                _register_mtf_pending(adapter, symbol, _fire, now_ms)
                            else:
                                asyncio.ensure_future(
                                    send_alt_signal(adapter, symbol, _fire, now_ms)
                                )
                except Exception as _alt_e:
                    logger.debug(f"[{adapter.name}] {symbol}: alt-detector eval skip: {_alt_e!r}")
            if res_tuple is None: return symbol, None
            return symbol, res_tuple
        except Exception as e:
            logger.error(
                f"[{adapter.name}] {symbol}: 4H detect error ({type(e).__name__}): {e!r}",
                exc_info=True,
            )
            return symbol, None

    new_setups = 0
    passes = 0

    for i in range(0, len(syms), BATCH_SIZE):
        if i > 0:
            now_ms = await safe_server_time_ms(adapter)

        batch = syms[i:i + BATCH_SIZE]
        tasks = [asyncio.create_task(detect_symbol(s)) for s in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for symbol, res_obj in results:
            if not res_obj: continue

            status, payload, *_ = res_obj
            if status == "REJ":
                reason = payload
                if reason in rejects:
                    rejects[reason] += 1
                    sample_push(reason, f"{symbol}")
                else:
                    if "other" not in rejects: rejects["other"] = 0
                    rejects["other"] += 1
                    sample_push("other", f"{symbol}:{reason}")
            elif status == "PASS":
                passes += 1
                price_at_div = payload["div_price"]
                ts = payload["ts"]
                atr4h_val = float(payload.get("atr4h", 0.0) or 0.0)

                swing_high = float(payload.get("swing_high", 0.0) or 0.0)
                decline_pct = float(payload.get("decline_pct", 0.0) or 0.0)
                div_position = float(payload.get("div_position", 0.0) or 0.0)

                market_state.update_4h_setup(adapter.name, symbol, price_at_div, ts, atr4h=atr4h_val)

                key = f"{adapter.name}:{symbol}"
                if key in market_state.books:
                    market_state.books[key]["swing_high"]   = swing_high
                    market_state.books[key]["decline_pct"]  = decline_pct
                    market_state.books[key]["div_position"] = div_position
                    market_state.books[key]["div_score"]    = int(payload.get("div_score", 5))
                    market_state.books[key]["vol_quality"]  = payload.get("vol_quality", "B")
                    market_state.books[key]["atr_pct_4h"]   = float(payload.get("atr_pct_4h", 0.0) or 0.0)
                    market_state.books[key]["rsi_4h_now"]   = float(payload.get("rsi_4h_now", 0.0) or 0.0)
                    market_state.books[key]["rsi_delta"]    = float(payload.get("rsi_delta", 0.0) or 0.0)
                    if not CLASSIC_DIV_ENABLED:
                        market_state.books[key]["shadow_only"] = True

                new_setups += 1

        await asyncio.sleep(0.5)

    market_state.flush()
    elapsed = time.time() - t0
    avg_lat = (elapsed / len(syms)) * 1000 if syms else 0

    logger.info(f"[{adapter.name}] 4H Scan Summary: Scanned={len(syms)} Found={new_setups} Latency={avg_lat:.1f}ms/sym")

    rej_str = " ".join([f"{k}={v}" for k, v in rejects.items() if v > 0])
    logger.info(f"[{adapter.name}] Rejections: {rej_str}")

    for k, v in samples.items():
        if v:
            logger.info(f"[{adapter.name}] Sample {k}: {v}")

    if len(_CACHE_STORE) > 1000:
        _CACHE_STORE.clear()


async def run_hourly_ticks(adapters):
    def _prio(ad):
        return 0 if ad.name.upper() == "BINANCE" else 1
    adapters_sorted = sorted(adapters, key=_prio)

    _btc_adapter = next((a for a in adapters_sorted if a.name.upper() == "BINANCE"), adapters_sorted[0])
    await update_btc_regime(_btc_adapter)

    results = await asyncio.gather(*[process_hourly_tick(ad) for ad in adapters_sorted])

    armed = sum(r[0] for r in results)
    conf  = sum(r[1] for r in results)
    exp   = sum(r[2] for r in results)
    inv   = sum(r[3] for r in results)

    logger.info(f"[ALL] Hourly totals: armed={armed} confirmed={conf} expired={exp} invalid={inv}")

    if MTF_1H_CONFIRM_ENABLED:
        await process_mtf_pending(adapters_sorted)

    market_state.flush()
    sent_signals.flush()

    for ad, res in zip(adapters_sorted, results):
        logger.info(f"[{ad.name}] Hourly Stats: Armed={res[0]} Confirmed={res[1]} Expired={res[2]} Invalid={res[3]}")


async def run_4h_scans(adapters):
    tasks = [cycle(ad) for ad in adapters]
    await asyncio.gather(*tasks)


# ─── MAIN ────────────────────────────────────────────────────────────────────
async def main():
    adapters = []

    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        bc = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    else:
        bc = await AsyncClient.create()
    adapters.append(BinanceAdapter(bc))

    if os.getenv("ENABLE_BITGET", "1") == "1":
        BG_KEY = os.getenv("BITGET_API_KEY")
        BG_SEC = os.getenv("BITGET_API_SECRET")
        BG_PASS = os.getenv("BITGET_PASSPHRASE")
        adapters.append(BitgetSpotAdapter(BG_KEY, BG_SEC, BG_PASS))

    if not adapters:
        logger.error("No exchanges configured! Exiting.")
        return

    acquire_pid_lock()
    fp = startup_fingerprint()
    logger.info(f"STARTUP fingerprint={fp} pid={os.getpid()}")
    logger.info(f"Adapters: {[a.name for a in adapters]}")
    logger.info("Strategy Config:")
    logger.info(f"  Reversal Filters: Decline>={MIN_DECLINE_FOR_REVERSAL:.0%}, DivPos<={MAX_DIV_POSITION:.0%}, Retest<={RETEST_ZONE_PCT:.0%}")
    logger.info(f"  Risk/Reward: SL={SL_BUFFER_ATR}ATR, DeRisk={TP_DERISK_R}R, SwingTarget={TP_SWING_MULT:.0%}")
    logger.info(f"  Scan: Top{TOP_N}, MinVol(Binance)={MIN_VOL_BINANCE:.0f}, MinVol(Bitget)={MIN_VOL_BITGET:.0f}, MinVol30d=${MIN_VOL_30D_USD/1e6:.0f}M")
    logger.info(f"  Day Filter: Wed={'ON' if DAY_FILTER_WED_ENABLED else 'OFF'}, Sat={'ON' if DAY_FILTER_SAT_ENABLED else 'OFF'}")
    logger.info(f"  DOW/Hour Blacklist: {'ENABLED' if DOW_HOUR_BLACKLIST_ENABLED else 'DISABLED'}, cells={DOW_HOUR_BLACKLIST}")
    logger.info(f"  Signal Blacklist ({len(SIGNAL_BLACKLIST)} symbols): {sorted(SIGNAL_BLACKLIST)}")
    logger.info(f"  SL Rule: 2H candle CLOSE below SL (signal text 'Stop: 2H close below X')")
    logger.info(f"  Divergence Score Gate: {'ENABLED' if MIN_DIV_SCORE_ENABLED else 'DISABLED'}, threshold={MIN_DIV_SCORE}/11")
    logger.info(f"  Div Position Gate: {'ENABLED' if MAX_DIV_POSITION_ENABLED else 'DISABLED'}, max={MAX_DIV_POSITION:.0%}")
    logger.info(f"  ATR_PCT_4H_MIN Gate: {'ENABLED' if ATR_PCT_4H_MIN_ENABLED else 'DISABLED'}, threshold={ATR_PCT_4H_MIN}")
    logger.info(f"  SL Risk Cap: MAX_SL_RISK_PCT={MAX_SL_RISK_PCT:.0%}")
    logger.info(f"  Exit Weights (back-heavy): {EXIT_WEIGHTS}")
    logger.info(f"  BTC Regime Gate: {'ENABLED' if BTC_GATE_ENABLED else 'DISABLED'}")
    logger.info("  Reversal Detectors:")
    logger.info(f"    squeeze_breakout: {'ON' if SQUEEZE_DETECTOR_ENABLED else 'OFF'}")
    logger.info(f"    failed_breakdown: {'ON' if FAILED_BD_DETECTOR_ENABLED else 'OFF'}")
    logger.info(f"    sweep_reclaim:    {'ON' if SWEEP_DETECTOR_ENABLED else 'OFF'}")
    logger.info(f"    classic_bull_div: {'ON' if CLASSIC_DIV_ENABLED else 'OFF'}")
    logger.info(f"  Regime-Adaptive TP: {'ON' if REGIME_ADAPTIVE_TP_ENABLED else 'OFF'} ({len(_REGIME_TP_WINNERS)} cells loaded)")
    logger.info(f"  MTF 1H Confirm: {'ON' if MTF_1H_CONFIRM_ENABLED else 'OFF'}")
    logger.info(f"  Cohort Filter:  {'ON' if COHORT_FILTER_ENABLED else 'OFF'}")
    logger.info(f"  Scanner Version: {SCANNER_VERSION}")

    await run_4h_scans(adapters)

    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        run_4h_scans,
        'cron', hour='*/4', minute=0, second=45,
        args=[adapters]
    )

    scheduler.add_job(
        run_hourly_ticks,
        'cron', minute=0, second=35,
        args=[adapters]
    )

    scheduler.start()
    logger.info("Scheduler active: 4H scans (:00:45), 1H ticks (:00:35)")

    stop_evt = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)
    await stop_evt.wait()

    logger.info("Shutdown requested...")
    scheduler.shutdown()
    for ad in adapters:
        await ad.close()

    global _tg_session
    if _tg_session and not _tg_session.closed:
        await _tg_session.close()

    release_pid_lock()
    logger.info("Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        release_pid_lock()
