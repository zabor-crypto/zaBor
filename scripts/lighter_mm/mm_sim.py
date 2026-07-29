#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lighter-mm-sim: Market-Making Strategy Simulator for Lighter.xyz DEX
================================================================================

Bar-by-bar simulation of a spread-based market-making strategy on SOL/USD,
calibrated to Lighter.xyz DEX mechanics (zero protocol fees, on-chain execution).

Key mechanics simulated
-----------------------
- Symmetric bid/ask quoting with ATR-scaled half-spread
- Inventory skew: widen the side you want to avoid, narrow the side you want
- Momentum toxicity filter: suppress quoting into strong directional flow
- Maker exit ladder: layered passive exits at 3 price levels
- Market exit fallback: taker exit after configurable timeout
- Edge gate: skip quoting if expected profit < threshold
- Drawdown watchdog: cooldown on intraday equity breach

Walk-forward optimization
-------------------------
Rolling train/test windows (default: 30k / 10k bars) with out-of-sample
constraint validation (PnL ≥ 0, MaxDD ≤ cap, AvgTradeSize ≥ minimum).
Parallel grid search via ProcessPoolExecutor.

Exchange profile: Lighter.xyz
------------------------------
- Maker fee: 0 bps
- Taker fee: 0 bps
- Gas fee: ~$0 (batched on-chain)
- Min order size: ~$400 notional

Usage
-----
    python mm_sim.py --data data/sol_1m_sample.csv --full-run
    python mm_sim.py --data data/sol_1m_sample.csv --walk-forward
    python mm_sim.py --data data/sol_1m_sample.csv --opt

See README.md for full parameter reference.

Disclaimer
----------
This is a research/simulation tool. Results do not guarantee live performance.
Do not deploy without your own validation on live data.
================================================================================
"""
from __future__ import annotations

import argparse
import copy
import glob
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Optional, Sequence, Tuple, List
import multiprocessing as mp

from collections import deque, defaultdict
import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError:
    stats = None


# === PATCH A: imports & globals ===
import math  # added by patch
# module-level worker globals for chunked parallel optimizer
_DF_GLOBAL = None
_BASE_GLOBAL = None
# --------------------- Configuration ---------------------
HARDCODED_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sol_1m_sample.csv")
DEBUG_FEE_CALCULATION = False  # Set to True for detailed fee debugging

# --------------------- Utilities ---------------------

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def fmt_td(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def validate_fee_model():
    """Validate fee model against actual Ostium data"""
    print("\n=== FEE MODEL VALIDATION (Ostium/Lighter) ===")
    
    # Test against user's actual data
    test_trades = [
        {"size": 1500, "expected_long": 1.50, "expected_short": 0.50},
        {"size": 2000, "expected_long": 2.00, "expected_short": 0.67},
        {"size": 3000, "expected_long": 3.00, "expected_short": 1.00},
    ]
    
    # Calibrated model from live data
    long_bps = 6.56
    short_bps = 6.78
    gas_fee = 0.02
    
    print(f"Model: Long {long_bps} bps + ${gas_fee} gas, Short {short_bps} bps + ${gas_fee} gas")
    print(f"NEW: Passive exits capture half-spread vs paying market slippage")
    print(f"NEW: Expected edge gating - only quote profitable sides")
    print(f"{'Size':<8} {'Long Actual':<12} {'Long Model':<12} {'Error':<8} {'Short Actual':<13} {'Short Model':<13} {'Error':<8}")
    print("-" * 80)
    
    total_error = 0
    for trade in test_trades:
        size = trade["size"]
        
        # Model calculation (OPENS ONLY)
        long_model = size * (long_bps / 10000) + gas_fee
        short_model = size * (short_bps / 10000) + gas_fee
        
        # Expected from actual data
        long_expected = trade["expected_long"]
        short_expected = trade["expected_short"]
        
        # Errors
        long_error = abs(long_model - long_expected) / long_expected * 100
        short_error = abs(short_model - short_expected) / short_expected * 100
        total_error += long_error + short_error
        
        print(f"${size:<7} ${long_expected:<11.2f} ${long_model:<11.2f} {long_error:<7.1f}% ${short_expected:<12.2f} ${short_model:<12.2f} {short_error:<7.1f}%")
    
    avg_error = total_error / (len(test_trades) * 2)
    print(f"\nAverage error: {avg_error:.1f}% ({'GOOD' if avg_error < 15 else 'NEEDS ADJUSTMENT'})")
    
    # Show passive exit benefits
    print(f"\nPASSIVE EXIT BENEFITS:")
    print(f"Market close: Pay ~2.2 bps slippage")
    print(f"Maker close: Capture ~half-spread (5-6 bps) when filled")
    print(f"Expected improvement: ~2-3 bps per trade with 70% maker success")
    print("=====================================\n")

# --------------------- Data Validation ---------------------

def validate_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Validate and clean input data"""
    original_len = len(df)
    issues = []
    
    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        issues.append(f"Missing values found: {dict(null_counts[null_counts > 0])}")
        df = df.fillna(method='ffill')
    
    # Check for price inconsistencies
    invalid_ohlc = (
        (df['High'] < df['Low']) | 
        (df['High'] < df['Open']) | 
        (df['High'] < df['Close']) | 
        (df['Low'] > df['Open']) | 
        (df['Low'] > df['Close'])
    )
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        issues.append(f"Invalid OHLC bars: {invalid_count}")
        df = df[~invalid_ohlc]
    
    # Check for extreme price gaps
    price_changes = df['Close'].pct_change().abs()
    extreme_moves = price_changes > 0.20
    if extreme_moves.any():
        extreme_count = extreme_moves.sum()
        issues.append(f"Extreme price moves (>20%): {extreme_count}")
    
    # Check for zero/negative prices
    zero_prices = (
        (df['Open'] <= 0) | 
        (df['High'] <= 0) | 
        (df['Low'] <= 0) | 
        (df['Close'] <= 0)
    )
    if zero_prices.any():
        zero_count = zero_prices.sum()
        issues.append(f"Zero/negative prices: {zero_count}")
        df = df[~zero_prices]
    
    # Check for duplicate timestamps
    if df['Timestamp'].duplicated().any():
        dup_count = df['Timestamp'].duplicated().sum()
        issues.append(f"Duplicate timestamps: {dup_count}")
        df = df.drop_duplicates(subset=['Timestamp'], keep='first')
    
    # Check for chronological order
    if not df['Timestamp'].is_monotonic_increasing:
        issues.append("Timestamps not in chronological order - sorting")
        df = df.sort_values('Timestamp')
    
    cleaned_len = len(df)
    data_loss_pct = (original_len - cleaned_len) / original_len * 100 if original_len > 0 else 0
    
    if verbose and issues:
        print(f"\n=== DATA VALIDATION REPORT ===")
        print(f"Original records: {original_len:,}")
        print(f"Cleaned records: {cleaned_len:,}")
        print(f"Data loss: {data_loss_pct:.2f}%")
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("===============================\n")
    
    return df.reset_index(drop=True)

def calculate_confidence_intervals(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence intervals for returns"""
    if len(returns) < 2:
        return 0.0, 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    n = len(returns)
    
    if stats is None:
        # Fallback to 1.96 for 95% CI if scipy missing
        t_stat = 1.96
    else:
        t_stat = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_stat * (std_return / np.sqrt(n))
    
    return mean_return - margin_error, mean_return + margin_error

def detect_market_regime(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Simple regime detection based on volatility clusters"""
    if len(returns) < window:
        return np.zeros(len(returns), dtype=int)
    
    vol = np.abs(returns)
    vol_ma = np.convolve(vol, np.ones(window)/window, mode='same')
    vol_std = np.std(vol)
    
    regimes = np.where(vol_ma < vol_std * 0.7, 0,
                      np.where(vol_ma < vol_std * 1.3, 1, 2))
    
    return regimes
    



# --------------------- Data Loading ---------------------

REQ_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

def load_csv(path: str, validate: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    rename = {}
    for c in cols:
        lc = c.lower()
        if lc in REQ_COLS or lc == "spread_bps":
            rename[c] = lc.capitalize() if lc != "spread_bps" else "spread_bps"
    df = df.rename(columns=rename)
    missing = [c.capitalize() for c in REQ_COLS if c.capitalize() not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp","Open","High","Low","Close"]).sort_values("Timestamp").reset_index(drop=True)
    
    if validate:
        df = validate_data(df, verbose=True)
    
    return df

# --------------------- Parameters ---------------------

@dataclass
class SimParams:
    # Ostium DEX Configuration
    equity_init_usd: float = 301.0      # 301 USDC collateral
    leverage: float = 11.0              # Fixed 11x leverage (301*11=3311 USD exposure)
    
    # Minimum order size enforcement
    min_order_usd: float = 1000.0       # Lowered from $1,500 (Lighter minimums ~$1k)
    auto_scale_to_min: bool = True      # Auto-scale to meet minimum when possible
    
    # CALIBRATED OSTIUM FEES (from live trading data)
    fee_model: str = "calibrated"
    # Exchange profile: "ostium" or "lighter"
    exchange: str = "lighter"           # Default to Lighter for this build

    # Lighter.xyz fee profile (Standard account = 0 fees)
    lighter_maker_fee_bps: float = 0.0
    lighter_taker_fee_bps: float = 0.0
    lighter_gas_fee_per_trade: float = 0.0

    # Slippage model for Lighter
    # Estimated slippage 0.00% (0 bps); cap worst-case at 0.1% (10 bps)
    max_slippage_bps: float = 10.0
       # "calibrated", "bimodal", or "conservative"
    
    # Calibrated fee levels from live data
    long_fee_bps: float = 6.56          # Long trade opening fees
    short_fee_bps: float = 6.78         # Short trade opening fees
    
    # Bimodal fee model (matches 3/10 bps split observed)
    use_bimodal_open_fee: bool = False
    open_fee_low_bps: float = 3.0       # Low fee tier
    open_fee_high_bps: float = 10.0     # High fee tier  
    open_fee_high_prob: float = 0.5     # Probability of high fee (~50/50 split)
    
    # Gas/network costs
    gas_fee_per_trade: float = 0.02     # Gas cost per trade (Arbitrum)
    
    # SLIPPAGE
    slippage_open_bps: float = 0.5      # Minimal for limit orders
    slippage_close_bps: float = 2.2     # Market close slippage

    # HIGH-IMPACT FIX 1: Passive exit parameters
    maker_exit_prob: float = 0.7        # Fraction of exits that try maker first
    exit_time_cap_bars: int = 5         # Fallback to market after timeout
    
    # Partial fills on exits (like entries)
    exit_max_volume_fraction: float = 0.10    # Max 10% of bar volume (vs 20% for entries)
    exit_partial_fill_slope: float = 0.6      # More conservative than entries (0.5)
    exit_queue_failure_prob: float = 0.15     # 15% chance limit order doesn't fill even when touched
    
    # Reduce forced taker exits
    # Note: baseline showed 35% maker (too low) with base_prob=0.25
    # Target: 60-70% maker → reduce to 10% base
    forced_taker_exit_base_prob: float = 0.10  # 10% base (was 25% - too aggressive)
    forced_taker_exit_vol_mult: float = 0.008  # +0.8% per bps volatility
    forced_taker_exit_size_mult: float = 0.30  # +30% per 1.0 size ratio
    # Expected: 10% base + multipliers = 15-30% forced taker → 70-85% maker ✓
    
    # Target efficiency from live data
    target_efficiency_bps: float = 6.7  # Calibrated from live: long 6.56, short 6.78
    efficiency_overshoot_penalty: float = 100.0  # Penalize >target efficiency
    
    # HIGH-IMPACT FIX 2: Nudged spreads for more edge
    half_spread_bps: float = 10.0       # TARGET: 10-12 bps (was 8.0)
    use_csv_spread: bool = True         # Use CSV spread data if available

    # **CRITICAL FIX A2**: Cut order sizing by 50%
    # Analysis: order_notional_frac 0.4-0.6 → 1.3-2k notional per side too aggressive
    order_notional_frac: float = 0.25   # Was 0.50 (cut 50%)
    
    # **CRITICAL FIX A1**: RISK CONTROL - Cut inventory to prevent 30-38% DD
    # Analysis showed: strategy over-leveraged with 3000-3500 max_net_inventory
    # Result: DD 30-38% in ALL OOS periods vs 25% target
    max_position_usd: float = 1_000.0        # Was 2000 (cut 50%)
    max_net_inventory_usd: float = 500.0     # Tighter default (was 1000)
    skew_bps_max: float = 4.0                # Keep for now

    # HIGH-IMPACT FIX 3: Tighter adverse momentum gating
    ewma_vol_span: int = 35                  # EWMA volatility window
    fill_base_prob: float = 0.35             # Conservative base fill probability (UNUSED IN OHLC CROSS)
    fill_vol_boost: float = 0.25             # Max volatility boost (UNUSED IN OHLC CROSS)
    adverse_mom_bps: float = 8.0             # TIGHTENED: from 10.0 to 8.0

    # **CRITICAL FIX B1**: Edge gating - raise minimum edge threshold
    # Analysis: min_edge 0.5-1.0 insufficient, avg OOS PnL = -$7.17
    # Spread capture 5.2 bps barely covers slippage/adverse selection
    min_edge_bps: float = 2.0            # Was 1.2 (raise 67%) - CRITICAL
    # Adaptive min-edge gate (based on rolling maker exit ratio)
    adaptive_min_edge: bool = True
    maker_exit_ratio_window: int = 30          # bars
    maker_exit_ratio_low: float = 0.60         # <60% → raise gate
    maker_exit_ratio_high: float = 0.70        # >70% → relax gate
    adaptive_min_edge_low: float = 1.5         # Was 1.25 (raise 20%)
    adaptive_min_edge_high: float = 2.5        # Was 1.70 (raise 47%)
    
    # Maker-exit ladder
    use_exit_ladder: bool = True
    exit_ladder_multipliers: Tuple[float, ...] = (1.00, 0.75, 0.50)  # ×half-spread levels, toward mid
    exit_ladder_stagger_bars: int = 1
    exit_ladder_tick_inside_bps: float = 0.5   # final level: 1 tick inside
    
    # Time-of-day biasing
    tod_overlap_size_mult: float = 1.15        # EU–US overlap
    tod_dead_size_mult: float = 0.80           # dead hours
    tod_dead_edge_delta_bps: float = 0.20      # increase min_edge during dead hours
    
    # Vol-of-vol aware adverse momentum guard
    vol_of_vol_window: int = 30
    adverse_mom_bps_hi: float = 13.0
    adverse_mom_bps_hi2: float = 14.0
    vol_of_vol_z_hi: float = 1.0
    vol_of_vol_z_hi2: float = 2.0
    
    # **CRITICAL FIX A3**: Strengthen inventory unwind bias
    # Analysis: unwind_bias_bps 0.8 insufficient with half_spread 6-12 bps
    unwind_bias_bps: float = 5.0               # INCREASED to 5.0 for inventory churn
    
    # Objective guardrail for optimizer
    efficiency_bps_floor: float = 1.5
    min_pnl_guard_usd: float = 0.0

    # **CRITICAL FIX A4**: DD watchdog - trigger pause earlier, realistic cap
    # Analysis: DD 30-38% in all OOS periods → need earlier intervention
    dd_cap_pct: float = 12.0                 # Hard cap (User requirement)
    dd_watchdog_pct: float = 10.0            # Trigger pause earlier (was 12.0)
    dd_cooldown_bars: int = 200              # Bars to pause trading after DD breach
    early_abort_dd_mult: float = 1.5         # Abort at dd_cap * 1.5
    
    # New Parameters for OHLCV Standardization
    daily_loss_stop_pct: float = 1.5
    spread_floor_bps: float = 12.0           # GOLDEN DEFAULT
    k_spread: float = 0.5
    tox_thr_bps: float = 10.0
    tox_scale_bps: float = 10.0
    tox_haircut_bps: float = 10.0

    # ORACLE STRATEGY (Latency Arbitrage)
    oracle_mode: bool = False           # Enable Virtual Oracle (Next Bar Lookahead)
    oracle_threshold_bps: float = 5.0   # Min lag to trigger arbitrage
    oracle_scaling: float = 2.0         # Multiplier for skew strength
    
    # General
    seed: int = 42

    # TREND-AWARE MARKET MAKING (Refactor v6)
    trend_ema_fast: int = 20            # Fast EMA window (e.g. 20m)
    trend_ema_slow: int = 100           # Slow EMA window (e.g. 100m)
    trend_filter_mode: int = 0          # 0=Disabled (Golden), 1=Skew, 2=Gate
    trend_gate_min_str: float = 0.5     # Min trend strength (bps diff) to trigger gate


@dataclass
class SimResult:
    total_return_pct: float
    max_dd_pct: float
    sharpe: float
    trades: int
    pnl_usd: float
    traded_usd: float
    equity_curve: np.ndarray
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    regime_performance: Dict[str, float] = None
    # Stability metrics
    equity_volatility: float = 0.0
    max_consecutive_losses: int = 0
    stability_score: float = 0.0
    avg_trade_pnl: float = 0.0
    win_rate: float = 0.0
    # Enhanced fee tracking
    total_fees_paid: float = 0.0
    long_trade_count: int = 0
    short_trade_count: int = 0
    avg_trade_size: float = 0.0
    # Control flags
    early_aborted: bool = False
    cooldown_triggered: bool = False
    min_order_violations: int = 0
    # HIGH-IMPACT FIX 1: Passive exit telemetry
    maker_exit_attempts: int = 0
    maker_exit_successes: int = 0
    maker_exit_ratio: float = 0.0
    avg_time_to_exit: float = 0.0
    spread_capture_bps: float = 0.0
    adverse_selection_bps: float = 0.0
    edge_gate_suppressions: int = 0
    # New diagnostics
    fees_per_million: float = 0.0
    session_fees: Dict[str, float] = None
    session_traded: Dict[str, float] = None
    maker_exit_ladder_fills: Dict[int, int] = None


# --------------------- Enhanced Simulation ---------------------


# === PATCH B: utilities ===
def _precompute_hours(df, ts_candidates=("Timestamp","timestamp","time","datetime","Date","date")):
    import pandas as pd
    for c in ts_candidates:
        if c in df.columns:
            return pd.DatetimeIndex(df[c]).hour.to_numpy("int8")
    return pd.DatetimeIndex(df.index).hour.to_numpy("int8")


# --- Helper: build ladder dict state for maker exit ladder ---
def _build_ladder_state(p, started_bar, side: str):
    """
    Create a ladder_state dict for maker exit ladder.
    side: 'long' or 'short' (the position we need to unwind).
    """
    levels = getattr(p, "exit_ladder_multipliers", (1.00, 0.75, 0.50))
    stagger = getattr(p, "exit_ladder_stagger_bars", 1)
    tick_in = getattr(p, "exit_ladder_tick_inside_bps", 0.5)

    return {
        "levels": list(levels),          # e.g. [1.00, 0.75, 0.50] of half-spread
        "placed": 0,                     # how many levels already placed
        "stagger_bars": int(stagger),    # stagger spacing in bars
        "tick_inside_bps": float(tick_in),
        "started_bar": int(started_bar),
        "side": side,                    # 'long' or 'short'
        # put any extra metadata here (NEVER add tuple fields):
        # "meta": {"trade_id": trade_id}
    }

class MarketMakerSim:
    def __init__(self, df: pd.DataFrame, params: SimParams) -> None:
        self.df = df
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        # Exchange-specific defaults
        if self.p.exchange.lower() == "lighter":
            # Fees zero; gas zero
            self.p.lighter_maker_fee_bps = 0.0
            self.p.lighter_taker_fee_bps = 0.0
            self.p.gas_fee_per_trade = 0.0
            # Slippage: 0 est, 10 bps cap
            self.p.slippage_open_bps = 0.0
            self.p.slippage_close_bps = 0.0
            self.p.max_slippage_bps = 10.0
            # No enforced minimum order size in sim unless you set it
            self.p.min_order_usd = 0.0
            self.p.auto_scale_to_min = True

        # Precompute arrays for speed
        # V8 Realism Fix: Use OPEN price as reference to avoid lookahead bias
        # (High+Low)/2 is cheating (centers quote in future range).
        self.mid = df["Open"].to_numpy(float)
        self.close = df["Close"].to_numpy(float)
        self.high = df["High"].to_numpy(float)
        self.low = df["Low"].to_numpy(float)
        self.volume = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
        self.ts = df["Timestamp"].to_numpy()
        
        # === PATCH C1: precompute hours once ===
        self.hours = _precompute_hours(self.df)
        self.csv_spread_bps = df["spread_bps"].to_numpy(float) if ("spread_bps" in df.columns and self.p.use_csv_spread) else None

        # Enhanced momentum and volatility calculations
        px = self.close
        ret = np.diff(px, prepend=px[0])
        mom_bps = np.zeros_like(px)
        mom_bps[1:] = 10_000.0 * ret[1:] / np.maximum(px[:-1], 1e-12)
        self.mom_bps = mom_bps

        # EWMA volatility
        abs_bps = np.abs(mom_bps)
        span = max(2, int(self.p.ewma_vol_span))
        alpha = 2.0 / (span + 1.0)
        vol = np.empty_like(abs_bps)
        vol[0] = abs_bps[0]
        for i in range(1, len(abs_bps)):
            vol[i] = alpha * abs_bps[i] + (1.0 - alpha) * vol[i - 1]
        self.vol_bps = vol
        
        # Efficient volatility reference (O(n) instead of O(n²))
        self.vol_ref = pd.Series(self.vol_bps).ewm(span=200).mean().to_numpy()
        
        # Vol-of-vol z-score for dynamic adverse momentum gate
        try:
            vol_series = pd.Series(self.vol_bps)
            vol_mean = vol_series.rolling(self.p.vol_of_vol_window).mean()
            vol_std = vol_series.rolling(self.p.vol_of_vol_window).std()
            self.vol_z = ((vol_series - vol_mean) / (vol_std + 1e-9)).to_numpy()
        except:
            self.vol_z = np.zeros_like(self.vol_bps)
            
        # ORACLE SIGNAL PRE-COMPUTATION
        # In Virtual Mode, Oracle Price = Next Bar's Close (Perfect Foresight)
        # In Production, this would be loaded from external CEX data
        if self.p.oracle_mode:
            # Shift Close backwards by 1 to get "Next Close" at current time
            # Fill last value with current close (no lookahead at end)
            self.oracle_price = np.roll(self.close, -1)
            self.oracle_price[-1] = self.close[-1]
        else:
            self.oracle_price = None

        
        # Volume moving average
        vol_window = min(20, len(self.volume))
        self.volume_ma = np.convolve(self.volume, np.ones(vol_window)/vol_window, mode='same')
        
        # ATR Calculation for Endogenous Spread
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = np.roll(self.close, 1)
        prev_close[0] = self.close[0]
        tr1 = self.high - self.low
        tr2 = np.abs(self.high - prev_close)
        tr3 = np.abs(self.low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Use EWM with alpha=1/14 (Wilder's smoothing)
        self.atr = pd.Series(tr).ewm(alpha=1.0/14.0, adjust=False, min_periods=1).mean().to_numpy()
        
        # TREND LOGIC: Precompute Fast/Slow EMAs
        close_s = pd.Series(self.close)
        self.ema_fast = close_s.ewm(span=max(5, self.p.trend_ema_fast), adjust=False).mean().to_numpy()
        self.ema_slow = close_s.ewm(span=max(20, self.p.trend_ema_slow), adjust=False).mean().to_numpy()
        # Trend strength in basis points
        self.trend_bps = (self.ema_fast - self.ema_slow) / self.mid * 10_000.0
        
        # Market regime detection
        self.regimes = detect_market_regime(ret)
        
        # Fee debugging
        self.fee_debug_count = 0
        self.total_fees_debug = 0.0

    def calculate_fees(self, qty: float, price: float, is_long: bool, trade_type: str = "open") -> float:
        # Lighter.xyz: fee-less trading (Standard), no gas
        if self.p.exchange.lower() == "lighter":
            return 0.0
        """
        Base fees ONLY on opens, gas only on closes
        """
        notional = abs(qty * price)
        
        # Base fees only on opens, not closes
        if trade_type == "open":
            if self.p.use_bimodal_open_fee:
                # Bimodal model: randomly choose between low/high fee tiers
                bps = self.p.open_fee_high_bps if self.rng.random() < self.p.open_fee_high_prob else self.p.open_fee_low_bps
            else:
                # Calibrated or fixed model
                if self.p.fee_model == "calibrated":
                    bps = self.p.long_fee_bps if is_long else self.p.short_fee_bps
                else:  # conservative fallback
                    bps = 7.5 if is_long else 4.5
            base_fee = notional * (bps / 10_000.0)
        else:
            # Closes only pay gas, NO base fee bps
            base_fee = 0.0
        
        # Always add small per-trade gas cost
        gas_fee = self.p.gas_fee_per_trade
        total_fee = base_fee + gas_fee
        
        # Fee debugging and validation
        self.fee_debug_count += 1
        self.total_fees_debug += total_fee
        
        if DEBUG_FEE_CALCULATION and self.fee_debug_count <= 10:
            direction = "LONG" if is_long else "SHORT"
            fee_bps = bps if trade_type == "open" else 0
            print(f"DEBUG Trade {self.fee_debug_count} ({trade_type}): ${notional:.0f} notional, "
                  f"{direction}, Fee: {fee_bps:.2f} bps + ${gas_fee:.2f} gas = ${total_fee:.4f}")
        
        return total_fee

    def ohlc_cross_fill(self, low, high, price, is_buy: bool) -> bool:
        return (low <= price) if is_buy else (high >= price)

    def run(self, half_spread_bps: Optional[float] = None) -> SimResult:
        p = self.p
        # === V8 FIX: Enforce Min Order ===
        if p.min_order_usd < 10.0:
            # Fix corrupted/zero min_order
            p.min_order_usd = 100.0
        # =================================
        # half_spread_bps ignored in favor of endogenous logic
        if self.fee_debug_count == 0:
             print(f"DEBUG RUN: min_order=${self.p.min_order_usd}, max_inv=${self.p.max_net_inventory_usd}, notional_frac={self.p.order_notional_frac}")

        
        equity = p.equity_init_usd
        cash = equity
        pos_qty = 0.0
        equity_curve = np.empty(len(self.close), dtype=float)
        
        # Tracking
        exit_events = deque()
        rolling_attempts = 0
        rolling_successes = 0
        
        trades = 0
        traded_usd = 0.0
        total_fees_paid = 0.0
        min_order_violations = 0
        long_trade_count = 0
        short_trade_count = 0
        
        # Telemetry
        position_id_counter = 0
        position_tracker = {} 
        total_exit_time = 0.0
        spread_capture_total = 0.0
        adverse_selection_total = 0.0
        edge_gate_suppressions = 0
        
        pending_exits = [] 
        
        # Risk State
        peak_equity = equity
        early_aborted = False
        cooldown_triggered = False
        cooldown_remaining = 0
        
        # Precompute
        slip_open = p.slippage_open_bps / 10_000.0
        slip_close = (0.0 if p.exchange.lower() == "lighter" else min(p.slippage_close_bps, p.max_slippage_bps)) / 10_000.0
        
        # Daily accounting
        current_day = None
        daily_start_equity = equity
        
        # Metrics
        max_dd = 0.0

        num_bars = len(self.close)

        for i in range(num_bars):
            mid = self.mid[i]
            ts = self.ts[i]
            low = self.low[i]
            high = self.high[i]
            
            # Daily reset
            if hasattr(ts, 'day'):
                day = ts.day
            else:
                day = (i // 1440)
            
            if current_day != day:
                current_day = day
                daily_start_equity = equity
            
            # 1. RISK CHECKS
            day_pnl_pct = (equity - daily_start_equity) / daily_start_equity * 100.0 if daily_start_equity > 0 else 0.0
            if day_pnl_pct < -p.daily_loss_stop_pct:
                equity = cash + pos_qty * self.close[i]
                equity_curve[i] = equity
                continue
                
            current_dd = (peak_equity - equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0
            if current_dd > p.dd_cap_pct:
                early_aborted = True
                equity_curve[i:] = equity
                break
                
            peak_equity = max(peak_equity, equity)
            
            # Watchdog
            if current_dd > p.dd_watchdog_pct and cooldown_remaining <= 0:
                cooldown_triggered = True
                cooldown_remaining = p.dd_cooldown_bars
            
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                equity = cash + pos_qty * self.close[i]
                equity_curve[i] = equity
                continue

            # 2. PROCESS EXITS
            new_pending_exits = []
            
            for exit_req in pending_exits:
                e_qty, e_px, bars, is_long, ladder_state, pos_id = exit_req
                bars += 1
                
                # Check Fill
                filled = False
                if is_long:
                    if high >= e_px: filled = True
                else:
                    if low <= e_px: filled = True
                
                if filled:
                    # Execute
                    fee = self.calculate_fees(e_qty, e_px, is_long, "close")
                    if is_long:
                        cash += e_qty * e_px - fee
                    else:
                        # V3 FIX: Deduct cost AND fee when buying to close short
                        cash -= e_qty * e_px + fee
                    
                    pos_qty -= e_qty if is_long else -e_qty
                    total_fees_paid += fee
                    
                    if pos_id in position_tracker:
                        position_tracker[pos_id]['maker_exit_qty'] += e_qty
                    
                    exit_events.append((i, True))
                    rolling_attempts += 1
                    rolling_successes += 1
                    
                    if pos_id in position_tracker:
                         entry_px = position_tracker[pos_id].get('entry_px', e_px)
                         cap = abs(e_px - entry_px)/entry_px * 10000.0
                         spread_capture_total += cap
                         
                else:
                    # Not filled
                    if bars > p.exit_time_cap_bars:
                        # Market Exit
                        mkt_px = mid * (1.0 - slip_close if is_long else 1.0 + slip_close)
                        fee = self.calculate_fees(e_qty, mkt_px, is_long, "close")
                        if is_long:
                            cash += e_qty * mkt_px - fee
                        else:
                            # V3 FIX: Deduct cost AND fee
                            cash -= e_qty * mkt_px + fee
                        
                        pos_qty -= e_qty if is_long else -e_qty
                        total_fees_paid += fee
                        
                        if pos_id in position_tracker:
                             position_tracker[pos_id]['taker_exit_qty'] += e_qty
                        
                        exit_events.append((i, False))
                        rolling_attempts += 1
                    else:
                        new_pending_exits.append((e_qty, e_px, bars, is_long, ladder_state, pos_id))
            
            pending_exits = new_pending_exits
            
            # Prune rolling stats
            while exit_events and (i - exit_events[0][0]) > p.maker_exit_ratio_window:
                old = exit_events.popleft()
                rolling_attempts -= 1
                if old[1]: rolling_successes -= 1
            
            # 3. ENTRY LOGIC
            
            # ATR & Base Spread
            atr_val = self.atr[i] if i < len(self.atr) else 0.0
            atr_bps = (atr_val / mid * 10000.0) if mid > 1e-12 else 0.0
            base_half = max(p.spread_floor_bps, p.k_spread * atr_bps)
            
            # Toxicity
            mom_bps = self.mom_bps[i]
            tox_bid = np.clip(((-mom_bps) - p.tox_thr_bps) / p.tox_scale_bps, 0.0, 1.0)
            tox_ask = np.clip(((+mom_bps) - p.tox_thr_bps) / p.tox_scale_bps, 0.0, 1.0)
            
            # V3 FIX: Inventory Skew
            # signed inventory fraction in [-1, 1]
            max_inv = p.max_net_inventory_usd
            inv_usd = abs(pos_qty * mid)
            # Inventory Skew
            inv_frac = np.clip((pos_qty * mid) / max(max_inv, 1.0), -1.0, 1.0)
            skew = inv_frac * p.skew_bps_max
            
            # REF V7: ORACLE STRATEGY (Latency Arbitrage)
            # If oracle_mode is ON, we override skew based on future price direction.
            if p.oracle_mode and self.oracle_price is not None:
                oracle_px = self.oracle_price[i]
                alpha_bps = (oracle_px - mid) / mid * 10_000.0
                
                # If "Future" > Current + Threshold -> BUY NOW
                if alpha_bps > p.oracle_threshold_bps:
                    # Calculate skew to exactly cross the spread + 2bps buffer
                    # Target: bid_half = -(base_half + 2.0)
                    # Formula: base_half + skew = -(base_half + 2.0)
                    # skew = -2*base_half - 2.0
                    skew = -(2.0 * base_half + 2.0)
                     
                # If "Future" < Current - Threshold -> SELL NOW
                elif alpha_bps < -p.oracle_threshold_bps:
                    # Target: ask_half = -(base_half + 2.0)
                    skew = (2.0 * base_half + 2.0)
            
            # REF V6: TREND SKEW (Mode 1 - MEAN REVERSION FLIP)
            # Hypothesis: Trend following lost money. Market is mean reverting.
            # If Uptrend (trend > 0) -> Add to skew -> Make skew positive -> Widen Bid (Avoid Longs), Tighten Ask (Short into strength)
            # If Downtrend (trend < 0) -> Subtract -> Make skew negative -> Widen Ask (Avoid Shorts), Tighten Bid (Buy into weakness)
            if p.trend_filter_mode == 1:
                skew += self.trend_bps[i]
            
            # Apply skew (Add skew to side we want to avoid, subtract from side we want to fill?)
            # User formula: 
            # bid_px = mid * (1.0 - (base_half + max(0, skew))/10000.0)
            # ask_px = mid * (1.0 + (base_half + max(0, -skew))/10000.0)
            
            # If long (skew > 0): max(0, skew) is positive -> Bid spread widens (lower px). Good.
            # max(0, -skew) is 0 -> Ask spread stays base (or tighter? Formula says base_half).
            # Wait, user formula: `base_half + max(0, -skew)`. If skew>0, this is base_half.
            # This logic strictly *widens* the dangerous side, but doesn't tighten the helping side?
            # User said: "Inventory control... effectively removed... improvement without microstructure fantasy."
            # The user's formula effectively effectively *gates* the side adding to inventory by widening it.
            # It doesn't tighten the other side. This is safe/conservative.
            
            # REF V7: ORACLE FIX - Allow crossing mid (Negative Spread)
            if p.oracle_mode:
                # In Oracle mode, we WANT to cross if skew is huge.
                # Do not clamp to min_edge.
                bid_half_final = base_half + skew
                ask_half_final = base_half - skew
            else:
                # Normal mode: Clamp to min_edge (Defense)
                bid_half_final = max(p.min_edge_bps, base_half + skew)
                ask_half_final = max(p.min_edge_bps, base_half - skew)
            
            # Entry Prices
            bid_px = mid * (1.0 - bid_half_final/10000.0)
            ask_px = mid * (1.0 + ask_half_final/10000.0)
            
            # Effective Edge (incorporating skew widening into the edge check?)
            # Usually Edge = Spread - Fees...
            # If we widen spread, edge increases? No, edge to MID increases.
            # But "real" edge? 
            # User: "quote only if edge_side >= min_edge_bps"
            # If we widen due to skew, do we want to quote *more* likely?
            # No, if we widen, we are further from mid -> harder to fill -> less inventory accumulation.
            # But `edge` calc: 
            
            # Let's simple check edge on the *base* spread or *final* spread?
            # User snippet for edge: "edge_bid = bid_spread_bps - tox_bid * tox_haircut_bps"
            # It implies using the final spread.
            # So if we widen bid, edge increases, so we are *more* likely to pass gate?
            # Yes, but price is lower, so less likely to fill. That works.
            
            edge_bid = bid_half_final - tox_bid * p.tox_haircut_bps
            edge_ask = ask_half_final - tox_ask * p.tox_haircut_bps
            
            min_edge = p.min_edge_bps
            
            # Quote Gating
            # (inv_usd < max_inv) check is partly redundant with skew logic but safest to keep as hard check for capacity.
            # If inv_frac is 1.0, skew is max. We might still quote if edge is huge?
            # No, keeping (inv_usd < max_inv or pos_qty < 0) ensures we don't breach hard cap.
            
            quote_bid = (edge_bid >= min_edge) and (inv_usd < max_inv or pos_qty < 0)
            quote_bid = (edge_bid >= min_edge) and (inv_usd < max_inv or pos_qty < 0)
            quote_ask = (edge_ask >= min_edge) and (inv_usd < max_inv or pos_qty > 0)

            # REF V6: TREND GATING (The "Alpha")
            # If trend is strong, DISABLE counter-trend quotes entirely.
            if p.trend_filter_mode == 2:
                trend_val = self.trend_bps[i]
                if trend_val > p.trend_gate_min_str:     # Strong UPTREND
                    quote_ask = False                    # Don't sell (ride it)
                elif trend_val < -p.trend_gate_min_str:  # Strong DOWNTREND
                    quote_bid = False                    # Don't buy (limit loss)
            
            
            

            if not quote_bid and not quote_ask:
                edge_gate_suppressions += 1
            
            # Size
            size_usd = p.order_notional_frac * equity
            if size_usd < p.min_order_usd:
                if p.auto_scale_to_min:
                   size_usd = min(p.max_position_usd, p.min_order_usd)
                else:
                   quote_bid = False
                   quote_ask = False
            
            qty = size_usd / mid if mid > 1e-12 else 0.0
            
            # EXECUTION
            # Bid Fill
            if quote_bid:
                if pos_qty >= 0 and (inv_usd + size_usd) > max_inv:
                    qty_c = max(0, (max_inv - inv_usd) / mid)
                else:
                    qty_c = qty
                
                # V3 FIX: Use p.min_order_usd only
                if qty_c * mid >= p.min_order_usd * 0.999: 
                     

                    if self.ohlc_cross_fill(low, high, bid_px, True):
                         # Fill
                         fee = self.calculate_fees(qty_c, bid_px, True, "open")
                         cash -= qty_c * bid_px + fee
                         pos_qty += qty_c
                         total_fees_paid += fee
                         trades += 1
                         long_trade_count += 1
                         traded_usd += qty_c * bid_px
                         
                         pos_id = position_id_counter
                         position_id_counter += 1
                         position_tracker[pos_id] = {'entry_qty': qty_c, 'maker_exit_qty':0, 'taker_exit_qty':0, 'entry_px': bid_px, 'is_long': True}
                         
                         pending_exits.append((qty_c, bid_px, 0, True, {}, pos_id))
            
            # Ask Fill
            if quote_ask:
                if pos_qty <= 0 and (inv_usd + size_usd) > max_inv:
                    qty_c = max(0, (max_inv - inv_usd) / mid)
                else:
                    qty_c = qty
                
                # V3 FIX: Use p.min_order_usd only
                if qty_c * mid >= p.min_order_usd * 0.999:
                     if self.ohlc_cross_fill(low, high, ask_px, False):
                         fee = self.calculate_fees(qty_c, ask_px, False, "open")
                         cash += qty_c * ask_px - fee
                         pos_qty -= qty_c
                         total_fees_paid += fee
                         trades += 1
                         short_trade_count += 1
                         traded_usd += qty_c * ask_px
                         
                         pos_id = position_id_counter
                         position_id_counter += 1
                         position_tracker[pos_id] = {'entry_qty': qty_c, 'maker_exit_qty':0, 'taker_exit_qty':0, 'entry_px': ask_px, 'is_long': False}
                         
                         pending_exits.append((qty_c, ask_px, 0, False, {}, pos_id))
            
            # Mark to Market
            equity = cash + pos_qty * self.close[i]
            equity_curve[i] = equity
            
        # END LOOP
        
        # Cleanup
        final_mid = self.close[-1]
        if pos_qty != 0 or pending_exits:
             mkt_slip = slip_close
             for e in pending_exits:
                 qty, px, b, is_long, l, pid = e
                 mkt_px = final_mid * (1.0 - mkt_slip if is_long else 1.0 + mkt_slip)
                 fee = self.calculate_fees(qty, mkt_px, is_long, "close")
                 if is_long: cash += qty * mkt_px - fee
                 else: 
                     # V3 FIX cleanup
                     cash -= qty * mkt_px + fee
                 pos_qty -= qty if is_long else -qty
             
             if abs(pos_qty) > 1e-9:
                  mkt_px = final_mid * (1.0 - mkt_slip if pos_qty > 0 else 1.0 + mkt_slip)
                  fee = self.calculate_fees(abs(pos_qty), mkt_px, pos_qty > 0, "close")
                  cash += pos_qty * mkt_px - fee
                  pos_qty = 0
             
             equity = cash
             if not early_aborted:
                 equity_curve[-1] = equity
        
        # Stats
        pnl = equity - p.equity_init_usd
        if len(equity_curve) > 1:
            # === V5 FIX: Array Broadcasting ===
            diffs = np.diff(equity_curve, prepend=equity_curve[0])
            denoms = np.concatenate(([equity_curve[0]], equity_curve[:-1]))
            rets = diffs / np.maximum(denoms, 1e-12)
            # ==================================
            mu = np.mean(rets)
            sd = np.std(rets)
            sharpe = (mu / sd * math.sqrt(24*60)) if sd > 0 else 0
        else:
            sharpe = 0
        
        peak = -np.inf
        curr_dd = 0
        for v in equity_curve:
            peak = max(peak, v)
            if peak > 0: curr_dd = max(curr_dd, (peak - v)/peak)
        max_dd = curr_dd
        
        vol_pen = min( (np.std(rets)*math.sqrt(1440)*100) / 50.0, 2.0 ) if len(rets)>1 else 0
        stability_score = vol_pen
        
        n_maker = 0
        inv_pos = 0
        for pid, data in position_tracker.items():
            inv_pos += 1
            if data['maker_exit_qty'] > (data['entry_qty'] * 0.9):
                n_maker += 1
        maker_exit_ratio = n_maker / max(1, inv_pos)
        avg_trade_size = traded_usd / trades if trades > 0 else 0.0
        avg_spread_capture = spread_capture_total / trades if trades > 0 else 0.0
        avg_time_to_exit = 0.0 # Placeholder or implementation needed if tracked
        
        # Calculate fees per million
        fees_per_million = (total_fees_paid / traded_usd * 1_000_000) if traded_usd > 0 else 0.0


        return SimResult(
            total_return_pct=(equity - p.equity_init_usd)/p.equity_init_usd*100,
            max_dd_pct=max_dd*100,
            sharpe=sharpe,
            trades=trades,
            pnl_usd=pnl,
            traded_usd=traded_usd,
            equity_curve=equity_curve,
            maker_exit_ratio=maker_exit_ratio,
            stability_score=stability_score,
            min_order_violations=min_order_violations,
            early_aborted=early_aborted,
            
            # V6 FIX: Telemetry
            total_fees_paid=total_fees_paid,
            long_trade_count=long_trade_count,
            short_trade_count=short_trade_count,
            avg_trade_size=avg_trade_size,
            spread_capture_bps=avg_spread_capture,
            edge_gate_suppressions=edge_gate_suppressions,
            fees_per_million=fees_per_million,
            maker_exit_attempts=rolling_attempts,
            maker_exit_successes=rolling_successes,
            # avg_time_to_exit passed as default 0.0 for now
        )

OPTIMIZATION_GRIDS = {
        "conservative": {
        "spread_floor_bps": [8.0, 12.0, 16.0],
        "k_spread": [0.25, 0.50],
        "tox_thr_bps": [10.0, 15.0],
        "tox_scale_bps": [10.0],
        "tox_haircut_bps": [5.0, 10.0],
        "min_edge_bps": [4.0, 6.0, 8.0],
        
        "order_notional_frac": [0.15, 0.25],
        "ewma_vol_span": [25, 50],
        "skew_bps_max": [5.0, 10.0],
        "max_net_inventory_usd": [1000],
        "maker_exit_prob": [0.7],
        "adverse_mom_bps": [10.0],
        "dd_watchdog_pct": [10.0],
    },
    
    "moderate": {
        # GOLDEN CONFIGURATION (Defensive)
        "spread_floor_bps": [12.0],
        "min_edge_bps": [6.0], 
        "order_notional_frac": [0.15],
        
        # Maximize Inventory Unwind for Stability
        "unwind_bias_bps": [5.0],
        
        # Disable Trend Filters (Noise Reduction)
        "trend_filter_mode": [0],
        
        # Oracle OFF (Requires Taker Bot architecture)
        "oracle_mode": [False],
        "oracle_threshold_bps": [5.0],
        
        # Standard Risk Controls
        "max_net_inventory_usd": [1000.0],
        "k_spread": [0.5],             
        "tox_thr_bps": [10.0],
        "tox_scale_bps": [10.0],
        "tox_haircut_bps": [10.0],
        "ewma_vol_span": [25],
        "skew_bps_max": [5.0],
        "max_net_inventory_usd": [1000],
        "maker_exit_prob": [0.7],
        "adverse_mom_bps": [10.0],
        "dd_watchdog_pct": [10.0],
    },

    "aggressive": {
        "spread_floor_bps": [3.0, 5.0],
        "k_spread": [0.10, 0.20, 0.30],
        "tox_thr_bps": [8.0, 12.0],
        "tox_scale_bps": [8.0, 12.0],
        "tox_haircut_bps": [2.0, 4.0, 6.0],
        "min_edge_bps": [0.5, 1.0, 1.5],
        
        "order_notional_frac": [0.35, 0.45, 0.55],
        "ewma_vol_span": [15, 20, 25, 30],
        "skew_bps_max": [1.0, 2.0, 3.0],
        "max_net_inventory_usd": [1500, 2000, 2500],
        "maker_exit_prob": [0.5, 0.7, 0.9],
        "adverse_mom_bps": [4.0, 6.0, 8.0],
        "dd_watchdog_pct": [10.0, 12.0],
    }
}

# --------------------- Parallel Grid Search ---------------------

# === PATCH H: objective for 'max volume with slight profit & stability' ===
# === PATCH H2: User Mandated Stable Profit Objective ===
def compute_score_for_goal(row, base_params):
    pnl = float(row.get("pnl_usd", 0.0))
    traded = float(row.get("traded_usd", 0.0))
    maxdd = float(row.get("max_dd_pct", 1e9))
    sharpe = float(row.get("sharpe", 0.0))
    stab = float(row.get("stability_score", 0.0))
    trades = int(row.get("trades", 0))
    early = bool(row.get("early_aborted", False))
    maker_exit_ratio = float(row.get("maker_exit_ratio", 0.0))

    # Hard reject: not enough trades / unstable
    if early or trades < 200:
        return -1e12

    dd_cap = float(getattr(base_params, "dd_cap_pct", 12.0))
    if pnl <= 0.0 or maxdd > dd_cap:
        return -1e11

    # Efficiency sanity floor (avoid “profit via luck + low turnover”)
    eff_bps = (pnl / traded) * 10_000 if traded > 0 else -1e9
    eff_floor = float(getattr(base_params, "efficiency_bps_floor", 1.5))
    if eff_bps < eff_floor:
        return -1e11

    # Maker realism band (Standard latency makes 90%+ suspicious)
    if maker_exit_ratio < 0.45 or maker_exit_ratio > 0.80:
        return -1e11

    dd_pen = (maxdd / dd_cap) ** 2

    # Profit-first, stability enforced; traded is only a tiny tie-breaker
    return (
        1.0 * pnl
        + 250.0 * sharpe
        + 80.0 * stab
        - 800.0 * dd_pen
        + 0.00001 * traded
    )


def _edge_prescreen(base_params, override_dict) -> bool:
    from dataclasses import asdict as _asdict
    p_dict = _asdict(base_params).copy()
    p_dict.update(override_dict)

    half = float(p_dict.get("half_spread_bps", 10.0))
    maker_prob = float(p_dict.get("maker_exit_prob", 0.7))
    min_edge = float(p_dict.get("min_edge_bps", 1.2))
    slippage_close = float(p_dict.get("slippage_close_bps", 2.0))
    fee_model = p_dict.get("fee_model", "calibrated")
    use_bimodal = bool(p_dict.get("use_bimodal_open_fee", False))

    def _open_fee_bps(side):
        if fee_model == "calibrated":
            return float(p_dict.get("long_fee_bps", 6.5) if side > 0 else p_dict.get("short_fee_bps", 6.8))
        if use_bimodal:
            hi_p = float(p_dict.get("open_fee_high_prob", 0.5))
            hi_b = float(p_dict.get("open_fee_high_bps", 8.0))
            lo_b = float(p_dict.get("open_fee_low_bps", 5.0))
            return hi_p * hi_b + (1.0 - hi_p) * lo_b
        return float(p_dict.get("open_fee_avg_bps", 6.0))

    exp_close = maker_prob * (-0.5 * half) + (1.0 - maker_prob) * slippage_close
    margin = 0.2
    edge_long  = half - _open_fee_bps(+1) - exp_close
    edge_short = half - _open_fee_bps(-1) - exp_close
    return (edge_long >= (min_edge - margin)) or (edge_short >= (min_edge - margin))

# --- Robust exit item normalizer (supports 4/5/6+ tuple forms, dicts, objects) ---
from collections.abc import Mapping, Sequence

def _normalize_exit_item(exit_item):
    """
    Normalize a pending-exit record to a canonical 6-tuple:
        (exit_qty, entry_price, bars_waiting, is_long, ladder_state, pos_id)

    Accepts:
      - dict-like: keys 'qty','entry_price','bars_waiting','is_long','ladder_state','pos_id'
      - object with attributes above
      - sequence (tuple/list/numpy array) with len >= 4:
            [qty, entry_price, bars_waiting, is_long, (ladder_state), (pos_id), ...ignored extras...]

    Added pos_id (6th element) for position tracking
    """

    # 1) Mapping (dict-like)
    if isinstance(exit_item, Mapping):
        exit_qty      = exit_item.get("qty")
        entry_price   = exit_item.get("entry_price")
        bars_waiting  = exit_item.get("bars_waiting", 0)
        is_long       = exit_item.get("is_long")
        ladder_state  = exit_item.get("ladder_state", None)
        pos_id        = exit_item.get("pos_id", None)
        return float(exit_qty), float(entry_price), int(bars_waiting), bool(is_long), ladder_state, pos_id

    # 2) Object with attributes
    attr_sets = [
        ("qty","entry_price","bars_waiting","is_long","ladder_state","pos_id"),
        ("qty","entry_price","bars_waiting","is_long","ladder_state"),
        ("qty","entry_price","bars_waiting","is_long"),
    ]
    for attrs in attr_sets:
        if all(hasattr(exit_item, a) for a in attrs):
            exit_qty     = getattr(exit_item, "qty")
            entry_price  = getattr(exit_item, "entry_price")
            bars_waiting = getattr(exit_item, "bars_waiting", 0)
            is_long      = getattr(exit_item, "is_long")
            ladder_state = getattr(exit_item, "ladder_state", None)
            pos_id       = getattr(exit_item, "pos_id", None)
            return float(exit_qty), float(entry_price), int(bars_waiting), bool(is_long), ladder_state, pos_id

    # 3) Generic sequence (tuple/list/np.array) — but not strings
    if isinstance(exit_item, Sequence) and not isinstance(exit_item, (str, bytes, bytearray)):
        seq = tuple(exit_item)
        n = len(seq)
        if n >= 6:
            exit_qty, entry_price, bars_waiting, is_long, ladder_state, pos_id = seq[:6]
        elif n == 5:
            exit_qty, entry_price, bars_waiting, is_long, ladder_state = seq
            pos_id = None  # Old format without pos_id
        elif n == 4:
            exit_qty, entry_price, bars_waiting, is_long = seq
            ladder_state = None
            pos_id = None
        elif n == 3:
            # Very old form? assume (qty, entry_price, is_long)
            exit_qty, entry_price, is_long = seq
            bars_waiting = 0
            ladder_state = None
            pos_id = None
        else:
            raise ValueError(f"Bad pending_exits item (len={n}): {exit_item!r}")

        return float(exit_qty), float(entry_price), int(bars_waiting), bool(is_long), ladder_state, pos_id

    # 4) Fallback
    raise TypeError(f"Unsupported pending_exits item type: {type(exit_item).__name__} -> {exit_item!r}")



def _init_worker_parallel(df, base_params):
    global _DF_GLOBAL, _BASE_GLOBAL
    _DF_GLOBAL = df
    _BASE_GLOBAL = base_params


def _run_combo_dict(combo_dict):
    global _DF_GLOBAL, _BASE_GLOBAL
    params = type(_BASE_GLOBAL)(**{**asdict(_BASE_GLOBAL), **combo_dict})
    sim = MarketMakerSim(_DF_GLOBAL, params)
    res = sim.run(half_spread_bps=params.half_spread_bps)
    return {
        "params": combo_dict,
        "pnl_usd": res.pnl_usd,
        "max_dd_pct": res.max_dd_pct,
        "traded_usd": res.traded_usd,
        "avg_trade_usd": res.avg_trade_size,
        "stability_score": getattr(res, "stability_score", 0.0),
        "early_aborted": getattr(res, "early_aborted", False),
        "maker_exit_success_ratio": getattr(res, "maker_exit_ratio", 0.0),
    }


def optimize_grid_chunked(df, base_params, grid_dict, max_workers=None, chunksize=128, topk=200):
    """
    Pre-screens combos, runs chunked parallel pool with global DF broadcast,
    returns (top_rows, diagnostics). Uses a tiebreaker in heap to avoid dict comparisons.
    Adds elapsed_sec to diag.
    """
    import heapq
    from itertools import product, count
    import time

    t0 = time.perf_counter()
    keys = list(grid_dict.keys())
    values = [list(v) for v in grid_dict.values()]

    # diagnostics
    diag = {"tested": 0, "prescreen_kept": 0, "elapsed_sec": 0.0}

    # lazy generator over combo dicts with prescreen
    def _gen():
        for vals in product(*values):
            d = dict(zip(keys, vals))
            if _edge_prescreen(base_params, d):
                yield d

    # small top-K heap: (score, tie_id, row)
    tie = count()
    heap = []

    # Single-threaded path (no Pool) for --single-thread runs
    use_workers = (max_workers or max(1, mp.cpu_count() - 1))
    if use_workers <= 1:
        _init_worker_parallel(df, base_params)  # set globals for _run_combo_dict
        for combo_dict in _gen():
            diag["prescreen_kept"] += 1
            row = _run_combo_dict(combo_dict)
            diag["tested"] += 1
            score = compute_score_for_goal(row, base_params)
            row["score"] = score
            item = (score, next(tie), row)
            if len(heap) < topk:
                heapq.heappush(heap, item)
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, item)
    else:
        # Parallel path
        with mp.Pool(processes=use_workers,
                     initializer=_init_worker_parallel,
                     initargs=(df, base_params)) as pool:
            for row in pool.imap_unordered(_run_combo_dict, _gen(), chunksize=chunksize):
                diag["tested"] += 1
                diag["prescreen_kept"] += 1  # generator has already filtered
                score = compute_score_for_goal(row, base_params)
                row["score"] = score
                item = (score, next(tie), row)
                if len(heap) < topk:
                    heapq.heappush(heap, item)
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, item)

    # unpack heap, best first
    top_rows = [r for (_s, _i, r) in sorted(heap, key=lambda x: x[0], reverse=True)]
    diag["elapsed_sec"] = time.perf_counter() - t0
    return top_rows, diag

def run_single_combo(args):
    """Run single parameter combination - used for parallel processing"""
    df, base_params, combo_dict, keys = args
    
    p_dict = asdict(base_params).copy()
    for k, v in combo_dict.items():
        p_dict[k] = v
    params = SimParams(**p_dict)

    sim = MarketMakerSim(df, params)
    res = sim.run(half_spread_bps=params.half_spread_bps)

    # UPDATED CONSTRAINTS with all fixes
    # Relax trade size constraint
    # Hard constraint: ≥$400 (real Lighter minimum)
    # Soft penalty: <$1,000 (prefer larger trades but don't reject)
    pnl_constraint = res.pnl_usd >= 0.0
    dd_constraint = res.max_dd_pct <= base_params.dd_cap_pct
    min_trade_size_constraint = res.avg_trade_size >= 400.0  # $400 (was $1,500)
    constraint_met = pnl_constraint and dd_constraint and min_trade_size_constraint
    
    if constraint_met and not res.early_aborted:
        # Updated scoring with soft trade size penalty
        base_score = res.traded_usd
        stability_bonus = max(0, (2.0 - res.stability_score) * 0.05 * base_score)
        pnl_bonus = min(res.pnl_usd * 60, base_score * 0.05)
        
        # Soft penalty for small trades (no hard rejection)
        # Prefer $1,000+ trades but don't reject $400-1,000
        if res.avg_trade_size < 600:
            trade_size_penalty = -50000  # Strong penalty for very small trades
        elif res.avg_trade_size < 1000:
            trade_size_penalty = -20000 * (1000 - res.avg_trade_size) / 400  # Linear from -20k to 0
        else:
            trade_size_penalty = 0  # No penalty for ≥$1,000
        
        # Light penalty for extremely low turnover
        # Encourage strategies that stay engaged when edge exists
        # Don't penalize intentional low activity during poor edge periods
        if res.traded_usd < 300_000:  # <$300k total volume
            turnover_penalty = -10000  # Light penalty (doesn't dominate scoring)
        else:
            turnover_penalty = 0
        
        dd_penalty = -1000000 * max(0, res.max_dd_pct - base_params.dd_cap_pct)
        
        score = base_score + stability_bonus + pnl_bonus + trade_size_penalty + turnover_penalty + dd_penalty
    else:
        score = -1000.0

    return {
        **combo_dict,
        "traded_usd": res.traded_usd,
        "pnl_usd": res.pnl_usd,
        "total_return_pct": res.total_return_pct,
        "max_dd_pct": res.max_dd_pct,
        "sharpe": res.sharpe,
        "trades": res.trades,
        "total_fees": res.total_fees_paid,
        "fee_pct_of_volume": res.total_fees_paid / res.traded_usd * 100 if res.traded_usd > 0 else 0,
        "long_trades": res.long_trade_count,
        "short_trades": res.short_trade_count,
        "avg_trade_size": res.avg_trade_size,
        "stability_score": res.stability_score,
        "win_rate": res.win_rate,
        "pnl_constraint": pnl_constraint,
        "dd_constraint": dd_constraint,
        "min_trade_size_constraint": min_trade_size_constraint,
        "constraint_met": constraint_met,
        "early_aborted": res.early_aborted,
        "cooldown_triggered": res.cooldown_triggered,
        "min_order_violations": res.min_order_violations,
        # HIGH-IMPACT FIX 1: Passive exit telemetry
        "maker_exit_ratio": res.maker_exit_ratio,
        "avg_time_to_exit": res.avg_time_to_exit,
        "spread_capture_bps": res.spread_capture_bps,
        "edge_gate_suppressions": res.edge_gate_suppressions,
        "fees_per_million": getattr(res, "fees_per_million", 0.0),
        "session_fees": getattr(res, "session_fees", {}),
        "session_traded": getattr(res, "session_traded", {}),
        "maker_exit_ladder_fills": getattr(res, "maker_exit_ladder_fills", {}),
        "score": score,
    }

def parallel_grid_search(df: pd.DataFrame, base: SimParams, grid: Dict[str, Sequence], 
                        max_workers: Optional[int] = None, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parallelized grid search for efficiency"""
    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    combos = list(product(*values))
    total = len(combos)
    start = time.time()

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Reasonable default

    if verbose:
        print(f"\nRunning PARALLEL grid search: {total:,} combinations on {max_workers} workers")
        print(f"HIGH-IMPACT FIXES: Passive exits + 10-12 bps spreads + edge gating")
        print(f"CONSTRAINTS: PnL≥0 & MaxDD≤{base.dd_cap_pct}% & AvgTradeSize≥$1,500")

    # Prepare arguments for parallel processing
    combo_dicts = [{keys[j]: combo[j] for j in range(len(keys))} for combo in combos]
    args_list = [(df, base, combo_dict, keys) for combo_dict in combo_dicts]
    
    rows = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_combo = {executor.submit(run_single_combo, args): args for args in args_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_combo):
            try:
                result = future.result()
                rows.append(result)
                completed += 1
                
                if verbose and (completed == 1 or completed % max(1, total // 50) == 0 or completed == total):
                    print(f"{now_ts()} {progress_eta(completed, total, start)}", flush=True)
                    
            except Exception as e:
                print(f"Error processing combination: {e}")
                completed += 1

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["constraint_met", "score", "stability_score"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # Find best result
    best_row = results.iloc[0].to_dict() if len(results) > 0 else {}

    if verbose:
        elapsed = time.time() - start
        print(f"\n=== HIGH-IMPACT OPTIMIZATION COMPLETE ===")
        print(f"Total time: {fmt_td(elapsed)} on {max_workers} workers")
        print(f"OBJECTIVE: MAXIMIZE Traded USD")
        print(f"CONSTRAINTS: PnL≥0 & MaxDD≤{base.dd_cap_pct}% & AvgTradeSize≥$1,500")
        print("============================================")
        
        display_cols = [
            *[k for k in keys if k in results.columns], "traded_usd", "pnl_usd", "max_dd_pct", "avg_trade_size", 
            "maker_exit_ratio", "spread_capture_bps", "edge_gate_suppressions", "constraint_met"
        ]
        if len(results) > 0:
            top10 = results[display_cols].head(10).copy()
            
            # Format key columns
            for col in ["traded_usd", "pnl_usd", "avg_trade_size"]:
                if col in top10.columns:
                    if col == "pnl_usd":
                        top10[col] = top10[col].apply(lambda x: f"${x:.2f}")
                    else:
                        top10[col] = top10[col].apply(lambda x: f"${x:,.0f}")
            
            if "maker_exit_ratio" in top10.columns:
                top10["maker_exit_ratio"] = top10["maker_exit_ratio"].apply(lambda x: f"{x:.1%}")
            
            print(f"\nTop 10 Results (Passive Exits + Higher Spreads):")
            print(top10.to_string(index=False))
        
        # Enhanced constraint analysis
        total_runs = len(results)
        constraint_success = (results['constraint_met'] == True).sum()
        pnl_violations = (results['pnl_constraint'] == False).sum()
        dd_violations = (results['dd_constraint'] == False).sum()
        min_size_violations = (results['min_trade_size_constraint'] == False).sum()
        
        print(f"\n=== CONSTRAINT ANALYSIS ===")
        print(f"Total parameter combinations: {total_runs:,}")
        print(f"All constraints met: {constraint_success}/{total_runs} ({constraint_success/total_runs*100:.1f}%)")
        print(f"PnL violations (not profitable): {pnl_violations} ({pnl_violations/total_runs*100:.1f}%)")
        print(f"MaxDD violations (unstable): {dd_violations} ({dd_violations/total_runs*100:.1f}%)")
        print(f"Min trade size violations (<$1,500): {min_size_violations} ({min_size_violations/total_runs*100:.1f}%)")
        
        if constraint_success > 0:
            valid_results = results[results['constraint_met'] == True]
            avg_traded = valid_results['traded_usd'].mean()
            max_traded = valid_results['traded_usd'].max()
            avg_pnl = valid_results['pnl_usd'].mean()
            avg_maker_ratio = valid_results['maker_exit_ratio'].mean()
            avg_spread_capture = valid_results['spread_capture_bps'].mean()
            print(f"Valid strategies:")
            print(f"  Avg traded: ${avg_traded:,.0f}, Max traded: ${max_traded:,.0f}")
            print(f"  Avg PnL: ${avg_pnl:.2f} (TARGET: flip from -$2 to positive)")
            print(f"  Avg maker exit ratio: {avg_maker_ratio:.1%}")
            print(f"  Avg spread capture: {avg_spread_capture:.2f} bps")
        print("==============================")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = f"optimization_results_{timestamp}.csv"
    results.to_csv(out_csv, index=False)
    results.to_csv("optimization_results.csv", index=False)
    
    if verbose:
        print(f"\nSaved all results to:")
        print(f"  📊 {os.path.abspath(out_csv)} (timestamped)")
        print(f"  📊 {os.path.abspath('optimization_results.csv')} (latest)")

    return results, best_row

def progress_eta(done: int, total: int, start_ts: float) -> str:
    elapsed = time.time() - start_ts
    rate = safe_div(done, elapsed, 0.0)
    eta = (total - done) / rate if rate > 0 else float("inf")
    bar_len = 24
    filled = int((done / total) * bar_len) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    return f"[{bar}] {done}/{total} ({done/total:.1%}) | elapsed {fmt_td(elapsed)} | ETA {fmt_td(eta)}"

# Keep the original single-threaded version as fallback
def grid_search(df: pd.DataFrame, base: SimParams, grid: Dict[str, Sequence], 
                verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Original single-threaded grid search - fallback for debugging"""
    return parallel_grid_search(df, base, grid, max_workers=1, verbose=verbose)

# --------------------- Walk-Forward Optimization ---------------------

def walk_forward_optimization(df: pd.DataFrame, base: SimParams, grid: Dict[str, Sequence],
                             train_size: int = 30_000, test_size: int = 10_000,
                             step_size: int = 10_000, use_parallel: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    **v5 PATCH 1**: Proper rolling walk-forward with fixed-size windows
    
    Creates 5 rolling steps with 30k train, 10k test windows:
    Step 1: Train [0:30k], Test [30k:40k]
    Step 2: Train [10k:40k], Test [40k:50k]
    Step 3: Train [20k:50k], Test [50k:60k]
    Step 4: Train [30k:60k], Test [60k:70k]
    Step 5: Train [40k:70k], Test [70k:80k]
    """
    total_bars = len(df)
    steps = (total_bars - train_size) // test_size
    
    if steps < 3:
        raise ValueError(f"Need ≥{train_size + 3*test_size} bars for 3-step WF. Got {total_bars}")
    
    print(f"\n=== WALK-FORWARD OPTIMIZATION v5 (Reality-Checked) ===")
    print(f"Total bars: {total_bars:,}")
    print(f"Train window: {train_size:,} bars (fixed)")
    print(f"Test window: {test_size:,} bars (fixed)")
    print(f"Step size: {step_size:,} bars (rolling)")
    print(f"Number of steps: {steps}")
    print(f"Parallel processing: {'ON' if use_parallel else 'OFF'}")
    print(f"v5 FIXES: Partial fills + Efficiency-first + Regime control + Stability tracking")
    print("=" * 70 + "\n")
    
    all_results = []
    param_history = []  # Track parameter drift
    walk_forward_start = time.time()
    
    for step in range(steps):
        step_start_time = time.time()
        
        train_start = step * step_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > total_bars:
            test_end = total_bars
        if test_end <= test_start:
            break
            
        print(f"Step {step + 1}/{steps}: Training on bars {train_start:,}-{train_end:,}, Testing on {test_start:,}-{test_end:,}")
        
        # Training phase
        train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        if use_parallel:
            train_results, best_params = parallel_grid_search(train_df, base, grid, verbose=True)
        else:
            train_results, best_params = grid_search(train_df, base, grid, verbose=True)
        
        # Out-of-sample testing
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)
        if best_params and len(test_df) > 100:
            test_param_dict = asdict(base).copy()
            for k in grid.keys():
                if k in best_params:
                    test_param_dict[k] = best_params[k]
            test_params = SimParams(**test_param_dict)
            
            sim = MarketMakerSim(test_df, test_params)
            test_result = sim.run()
            
            oos_pnl_constraint = test_result.pnl_usd >= 0.0
            oos_dd_constraint = test_result.max_dd_pct <= base.dd_cap_pct
            oos_min_trade_size_constraint = test_result.avg_trade_size >= 400.0  # $400 (was $1,500)
            oos_constraint_met = oos_pnl_constraint and oos_dd_constraint and oos_min_trade_size_constraint
            
            # Regime detection and per-day metrics
            test_days = len(test_df) / (60 * 24)  # Assuming 1-min bars
            trades_per_day = test_result.trades / max(0.1, test_days)
            notional_per_day = test_result.traded_usd / max(0.1, test_days)
            
            # Classify activity regime
            # High activity: >$2M per window, Low activity: <$1M per window
            if test_result.traded_usd > 2_000_000:
                activity_regime = "HIGH"
            elif test_result.traded_usd < 1_000_000:
                activity_regime = "LOW"
            else:
                activity_regime = "MEDIUM"
            
            step_result = {
                'step': step + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_bars': len(train_df),
                'test_bars': len(test_df),
                'test_days': test_days,
                'oos_return_pct': test_result.total_return_pct,
                'oos_sharpe': test_result.sharpe,
                'oos_max_dd_pct': test_result.max_dd_pct,
                'oos_trades': test_result.trades,
                'oos_trades_per_day': trades_per_day,
                'oos_traded_usd': test_result.traded_usd,
                'oos_notional_per_day': notional_per_day,
                'activity_regime': activity_regime,
                'oos_pnl_usd': test_result.pnl_usd,
                'oos_total_fees': test_result.total_fees_paid,
                'oos_long_trades': test_result.long_trade_count,
                'oos_short_trades': test_result.short_trade_count,
                'oos_avg_trade_size': test_result.avg_trade_size,
                'oos_fee_pct_of_volume': test_result.total_fees_paid / test_result.traded_usd * 100 if test_result.traded_usd > 0 else 0,
                'oos_stability_score': test_result.stability_score,
                'oos_win_rate': test_result.win_rate,
                'oos_early_aborted': test_result.early_aborted,
                'oos_cooldown_triggered': test_result.cooldown_triggered,
                'oos_min_order_violations': test_result.min_order_violations,
                'oos_maker_exit_ratio': test_result.maker_exit_ratio,
                'oos_avg_time_to_exit': test_result.avg_time_to_exit,
                'oos_spread_capture_bps': test_result.spread_capture_bps,
                'oos_edge_gate_suppressions': test_result.edge_gate_suppressions,
                'oos_pnl_constraint': oos_pnl_constraint,
                'oos_dd_constraint': oos_dd_constraint,
                'oos_min_trade_size_constraint': oos_min_trade_size_constraint,
                'constraint_met': oos_constraint_met,
                **{f'best_{k}': best_params.get(k) for k in grid.keys()}
            }
            all_results.append(step_result)
            # === V4 PATCH: INTERMEDIATE SAVE ===
            try:
                 pd.DataFrame(all_results).to_csv(f"walk_forward_results_intermediate.csv", index=False)
            except Exception as e:
                 print(f"Warning: Could not save intermediate results: {e}")
            # ===================================

            
            step_time = time.time() - step_start_time
            constraint_status = "✓" if oos_constraint_met else "✗"
            pnl_status = f"${test_result.pnl_usd:.2f}"
            maker_status = f"({test_result.maker_exit_ratio:.0%}mk)"
            spread_status = f"({test_result.spread_capture_bps:.1f}bps)"
            
            print(f"  OOS Results {constraint_status}: Traded ${test_result.traded_usd:,.0f} | "
                  f"PnL {pnl_status} | DD {test_result.max_dd_pct:.1f}% | "
                  f"AvgTrade ${test_result.avg_trade_size:.0f} | "
                  f"MakerExit {maker_status} | Spread {spread_status} | Time {fmt_td(step_time)}")
        else:
            print(f"  Skipping step {step + 1} - insufficient data or no valid parameters")
    
    if not all_results:
        print("No valid walk-forward results generated")
        return pd.DataFrame(), {}
    
    wf_results = pd.DataFrame(all_results)
    valid_results = wf_results[wf_results['constraint_met'] == True]
    
    if len(valid_results) == 0:
        print("⚠ No periods met ALL constraints!")
        print("Expected: Passive exits + higher spreads should flip from -$2 to positive")
        print("\nDEBUG INFO:")
        if len(all_results) > 0:
            avg_pnl = wf_results['oos_pnl_usd'].mean()
            avg_maker_ratio = wf_results['oos_maker_exit_ratio'].mean()
            avg_spread_capture = wf_results['oos_spread_capture_bps'].mean()
            print(f"Average OOS PnL: ${avg_pnl:.2f} (target: >$0)")
            print(f"Average maker exit ratio: {avg_maker_ratio:.1%}")
            print(f"Average spread capture: {avg_spread_capture:.2f} bps")
        return wf_results, {}
    
    # Parameter consistency analysis
    param_counts = {}
    for k in grid.keys():
        param_col = f'best_{k}'
        if param_col in valid_results.columns:
            param_counts[k] = valid_results[param_col].mode().iloc[0] if len(valid_results[param_col].mode()) > 0 else list(grid[k])[0]
    
    # **v5 PATCH 5**: Parameter stability tracking
    param_stability = {}
    param_history = []
    for idx, row in wf_results.iterrows():
        step_params = {k: row[f'best_{k}'] for k in grid.keys() if f'best_{k}' in row}
        param_history.append(step_params)
    
    if len(param_history) >= 2:
        for param in grid.keys():
            values = [float(p[param]) for p in param_history if param in p]
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 1e-9 else 0.0
                param_stability[param] = cv
    
    # Enhanced performance summary
    avg_traded = valid_results['oos_traded_usd'].mean()
    max_traded = valid_results['oos_traded_usd'].max()
    avg_pnl = valid_results['oos_pnl_usd'].mean()
    avg_dd = valid_results['oos_max_dd_pct'].mean()
    
    # **v5 PATCH 3**: Calculate efficiency (bps) - CRITICAL METRIC
    avg_efficiency_bps = (valid_results['oos_pnl_usd'] / valid_results['oos_traded_usd'] * 10_000).mean() if len(valid_results) > 0 else 0.0
    avg_fees = valid_results['oos_fee_pct_of_volume'].mean()
    avg_trade_size = valid_results['oos_avg_trade_size'].mean()
    avg_maker_ratio = valid_results['oos_maker_exit_ratio'].mean()
    avg_spread_capture = valid_results['oos_spread_capture_bps'].mean()
    win_rate = (valid_results['oos_pnl_usd'] > 0).mean() * 100
    
    total_time = time.time() - walk_forward_start
    
    print(f"\n=== WALK-FORWARD SUMMARY ===")
    print(f"Maker ratio calibrated; trade size soft-penalty; activity regime detection")
    print(f"✓ CALIBRATED: Target {base.target_efficiency_bps:.1f} bps (from live data)")
    print(f"✓ CONSTRAINTS: PnL≥0 & MaxDD≤{base.dd_cap_pct}% & AvgTradeSize≥$400")
    print(f"Valid periods: {len(valid_results)}/{len(all_results)} ({len(valid_results)/len(all_results)*100:.1f}%)")
    print(f"Average OOS traded: ${avg_traded:,.0f} (max: ${max_traded:,.0f})")
    print(f"Average OOS PnL: ${avg_pnl:.2f}")
    
    # Activity regime breakdown
    if 'activity_regime' in valid_results.columns:
        print(f"\n📊 ACTIVITY REGIME BREAKDOWN:")
        regime_counts = valid_results['activity_regime'].value_counts()
        for regime, count in regime_counts.items():
            regime_data = valid_results[valid_results['activity_regime'] == regime]
            avg_notional = regime_data['oos_notional_per_day'].mean()
            avg_trades = regime_data['oos_trades_per_day'].mean()
            print(f"  {regime}: {count} periods | ${avg_notional/1e6:.2f}M/day | {avg_trades:.0f} trades/day")
    
    print(f"\n🎯 EFFICIENCY VALIDATION:")
    print(f"  Achieved: {avg_efficiency_bps:.2f} bps")
    print(f"  Target: {base.target_efficiency_bps:.1f} bps (from live: long 6.56, short 6.78)")
    
    # Tighter efficiency bounds (2-12 bps, was 2-15)
    if avg_efficiency_bps > 12.0:
        print(f"  ❌ REJECT: Efficiency >12 bps is overfitted (was {avg_efficiency_bps:.2f} bps)")
    elif avg_efficiency_bps > base.target_efficiency_bps + 2.0:
        print(f"  ⚠ WARNING: {avg_efficiency_bps - base.target_efficiency_bps:.1f} bps above target (suspicious)")
    elif avg_efficiency_bps < 2.0:
        print(f"  ⚠ WARNING: Efficiency <2 bps too low for viable MM")
    elif avg_efficiency_bps >= base.target_efficiency_bps - 1.0 and avg_efficiency_bps <= base.target_efficiency_bps + 2.0:
        print(f"  ✓ EXCELLENT: Within target range ({base.target_efficiency_bps-1:.1f}-{base.target_efficiency_bps+2:.1f} bps)")
    else:
        print(f"  ✓ Efficiency in acceptable range (2-12 bps)")
    
    print(f"\n🎯 MAKER EXIT RATIO VALIDATION:")
    print(f"  Achieved: {avg_maker_ratio:.1%}")
    print(f"  Target: 60-80% maker (DEX market making typical range)")
    print(f"  Note: Adjust target if live Lighter data shows different behavior")
    if avg_maker_ratio > 0.95:
        print(f"  ❌ REJECT: {avg_maker_ratio:.0%} maker exits is impossible (100% unrealistic)")
    elif avg_maker_ratio > 0.85:
        print(f"  ⚠ WARNING: {avg_maker_ratio:.0%} is too high (exit model still optimistic)")
    elif avg_maker_ratio >= 0.60 and avg_maker_ratio <= 0.80:
        print(f"  ✓ EXCELLENT: In target range (60-80%)")
    elif avg_maker_ratio >= 0.50 and avg_maker_ratio < 0.60:
        print(f"  ✓ ACCEPTABLE: {avg_maker_ratio:.0%} (slightly below target but may reflect real behavior)")
    else:
        print(f"  ⚠ Outside range: {avg_maker_ratio:.0%} (calibrate to live Lighter data)")
    
    print(f"\nOther Metrics:")
    print(f"  Average OOS max DD: {avg_dd:.2f}%")
    print(f"  Average OOS trade size: ${avg_trade_size:.0f}")
    print(f"  Average spread capture: {avg_spread_capture:.2f} bps")
    print(f"  Period win rate: {win_rate:.1f}%")
    print(f"Total time: {fmt_td(total_time)}")
    
    # Parameter stability report
    if param_stability:
        print(f"\n=== PARAMETER STABILITY (CV < 0.30 = stable) ===")
        unstable_params = []
        for param, cv in sorted(param_stability.items(), key=lambda x: x[1], reverse=True):
            status = "✓" if cv < 0.30 else "✗ UNSTABLE"
            print(f"  {param:25s}: CV={cv:.3f} {status}")
            if cv >= 0.30:
                unstable_params.append(param)
        
        if unstable_params:
            print(f"\n⚠ OVERFITTING WARNING: {len(unstable_params)} parameters unstable")
            print(f"  Unstable: {', '.join(unstable_params)}")
            print(f"  → High parameter drift suggests overfitting to specific periods")
        else:
            print(f"\n✓ All parameters stable (CV < 0.30)")
    
    print("\nMost consistent parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S") 
    wf_csv = f"walk_forward_results_{timestamp}.csv"
    wf_results.to_csv(wf_csv, index=False)
    wf_results.to_csv("walk_forward_results.csv", index=False)
    
    print(f"📊 Walk-forward results saved to:")
    print(f"  📈 {os.path.abspath(wf_csv)} (timestamped)")  
    print(f"  📈 {os.path.abspath('walk_forward_results.csv')} (latest)")
    
    return wf_results, param_counts

# --------------------- Plotting ---------------------

def plot_equity(equity: np.ndarray, title: str = "Equity Curve") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available ({e}); skipping plot.")
        return
    plt.figure(figsize=(12,6))
    plt.plot(equity, linewidth=1.5)
    plt.title(f"{title} (High-Impact Fixes Applied)")
    plt.xlabel("Time (1-minute bars)")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=140)
    print(f"Saved equity curve to {os.path.abspath('equity_curve.png')}")

# --------------------- CLI / Main ---------------------

def show_fee_model_summary():
    """Display the high-impact fixes"""
    print("🎯 HIGH-IMPACT FIXES TO FLIP FROM -$2 TO PROFITABLE:")
    print(f"   1. PASSIVE EXITS: Capture spread vs paying slippage (~70% maker)")
    print(f"   2. HIGHER SPREADS: Target 10-12 bps for more edge")
    print(f"   3. TIGHTER GATING: Avoid adverse momentum (6-10 bps threshold)")
    print(f"   4. STRONGER SKEW: Faster inventory mean-reversion (3-5 bps)")
    print(f"   5. EDGE GATING: Only quote sides with >1.2 bps expected profit")

def menu() -> str:
    print(f"\n=== Lighter MM Simulator ===")
    print(f"Configuration: 301 USDC × 11x leverage = 3,311 USD exposure")
    print(f"")
    show_fee_model_summary()
    print(f"")
    print(f"🎯 OPTIMIZATION OBJECTIVE:")
    print(f"   MAXIMIZE: Traded USD Volume")
    print(f"   SUBJECT TO: PnL ≥ $0 & MaxDD ≤ X% & AvgTradeSize ≥ $1,500")
    print(f"   EXPECTED: Flip from -$2.14 loss to positive PnL")
    print(f"")
    print("Options:")
    print("1) Run with high-impact fixes (recommended)")
    print("2) Run conservative optimization (proven ranges)")
    print("3) Run walk-forward optimization (recommended, parallel)")
    print("4) Run moderate optimization (wider ranges, parallel)")
    print("5) Run aggressive optimization (full ranges, parallel)")
    print("6) Validate fee model against actual data")
    print("7) Enable fee calculation debugging")
    print("8) Run single-threaded (debugging mode)")
    choice = input("Select [1/2/3/4/5/6/7/8]: ").strip()
    return choice or "1"

def main(argv: Optional[Sequence[str]] = None) -> None:
    global DEBUG_FEE_CALCULATION
    
    parser = argparse.ArgumentParser(description="Market-making simulator for Lighter.xyz DEX")
    parser.add_argument("--data", type=str, default=HARDCODED_DATA_PATH, help="Path to CSV")
    parser.add_argument("--full-run", action="store_true", help="Run single simulation on FULL history with calibrated params")
    parser.add_argument("--opt", action="store_true", help="Run conservative optimization")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--dd-cap", type=float, default=None, help="Max drawdown cap (%%)")
    parser.add_argument("--debug-fees", action="store_true", help="Enable fee calculation debugging")
    parser.add_argument("--single-thread", action="store_true", help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (defaults to CPU-1)")
    parser.add_argument("--bimodal", action="store_true", help="Use bimodal fee model")
    parser.add_argument("--exchange", type=str, default="lighter", choices=["ostium", "lighter"], help="Exchange profile (affects fees & slippage)")
    parser.add_argument("--use-csv-spread", action="store_true", help="Use per-bar spread_bps from CSV")
    args = parser.parse_args(argv)

    # Apply exchange profile
    exch = (args.exchange or "lighter").lower()

    if args.debug_fees:
        DEBUG_FEE_CALCULATION = True
        print("🔍 Fee calculation debugging ENABLED")

    # Load and validate data
    df = load_csv(args.data, validate=True)
    print(f"Loaded {len(df):,} rows from {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")

    base = SimParams()
    
    base.exchange = exch
    base.use_csv_spread = args.use_csv_spread

    if args.dd_cap is not None:
        base.dd_cap_pct = float(args.dd_cap)
    if args.bimodal:
        base.use_bimodal_open_fee = True
        base.fee_model = "bimodal"

    # Determine mode
    if args.full_run:
        choice = "FULL"
    elif args.opt:
        choice = "2"
    elif args.walk_forward:
        choice = "3"
    else:
        choice = menu()

    use_parallel = not args.single_thread

    if choice == "1":
        # Single run with all high-impact fixes
        print(f"\n🎯 Running simulation with HIGH-IMPACT FIXES...")
        print(f"Target: Flip from -$2.14 loss to positive PnL")
        print(f"Half-spread: {base.half_spread_bps} bps (targeting 10-12)")
        print(f"Maker exit prob: {base.maker_exit_prob:.0%} (time cap: {base.exit_time_cap_bars} bars)")
        print(f"Min edge: {base.min_edge_bps} bps (gate unprofitable sides)")
        print(f"Adverse momentum: {base.adverse_mom_bps} bps (tightened gating)")
        
        sim = MarketMakerSim(df, base)
        res = sim.run()
        
        print(f"\nResults:")
        print(f"Traded Volume: ${res.traded_usd:,.0f}")
        print(f"PnL: ${res.pnl_usd:.2f} ({'✓ SUCCESS!' if res.pnl_usd > 0 else '✗ Still negative'})")
        print(f"Max DD: {res.max_dd_pct:.2f}% (constraint: ≤{base.dd_cap_pct}%) {'✓' if res.max_dd_pct <= base.dd_cap_pct else '✗'}")
        print(f"Avg Trade Size: ${res.avg_trade_size:.0f} ({'✓' if res.avg_trade_size >= 1500 else '✗'} ≥$1,500)")
        print(f"Total Fees: ${res.total_fees_paid:.2f} ({res.total_fees_paid/res.traded_usd*100:.3f}% of volume; {res.fees_per_million:.1f} $/M)")
        print(f"Trade Count: {res.trades} ({res.long_trade_count} long, {res.short_trade_count} short)")
        print(f"")
        print(f"HIGH-IMPACT TELEMETRY:")
        print(f"  Maker Exit Ratio: {res.maker_exit_ratio:.1%} (target: >50%)")
        print(f"  Avg Time to Exit: {res.avg_time_to_exit:.1f} bars")
        print(f"  Spread Capture: {res.spread_capture_bps:.2f} bps (positive = good)")
        print(f"  Edge Gate Suppressions: {res.edge_gate_suppressions} ({res.edge_gate_suppressions/res.trades*100:.1f}% of potential trades)")
        print(f"  Win Rate: {res.win_rate:.1f}%")
        # Session slices
        if res.session_traded:
            print("Session efficiency (bps) and traded USD:")
            for sess, traded_val in res.session_traded.items():
                if traded_val > 0:
                    eff = 0.0
                    # We only tracked fees per session; efficiency per session approximated via fees isn't precise, so show traded and fees
                    fees = res.session_fees.get(sess, 0.0) if res.session_fees else 0.0
                    print(f"  {sess:>7}: traded ${traded_val:,.0f} | fees ${fees:,.2f}")
        if res.maker_exit_ladder_fills:
            print("Maker-exit ladder fills by level index:", dict(res.maker_exit_ladder_fills))

        plot_equity(res.equity_curve)
        return
    
    elif choice == "FULL":
        # === FULL-HISTORY RUN with fixed calibrated params ===
        # Parameters from the most recent walk-forward validation:
        base.half_spread_bps      = 10.0
        base.order_notional_frac  = 0.35
        base.ewma_vol_span        = 25
        base.fill_base_prob       = 0.35
        base.skew_bps_max         = 5.0
        base.max_net_inventory_usd = 750.0
        base.maker_exit_prob      = 0.7
        base.adverse_mom_bps      = 10.0
        base.min_edge_bps         = 2.0

        print("\n🎯 FULL-HISTORY RUN (all bars) with calibrated params")
        print(f"Rows: {len(df):,}")
        print(f"Half-spread: {base.half_spread_bps} bps")
        print(f"Order notional frac: {base.order_notional_frac}")
        print(f"Min edge: {base.min_edge_bps} bps")
        print(f"Adverse momentum: {base.adverse_mom_bps} bps")

        sim = MarketMakerSim(df, base)
        res = sim.run()

        # === local edge computation, in bps ===
        if res.traded_usd > 0:
            edge_bps = res.pnl_usd / res.traded_usd * 1e4
        else:
            edge_bps = 0.0

        print("\n=== FULL-HISTORY RESULTS ===")
        print(f"Traded Volume: ${res.traded_usd:,.0f}")
        print(f"PnL: ${res.pnl_usd:,.2f}")
        print(f"Max DD: {res.max_dd_pct:.2f}%")
        print(f"Total return: {res.total_return_pct:.2f}%")
        print(f"Sharpe: {res.sharpe:.2f}")
        print(f"Trades: {res.trades}")
        print(f"Avg Trade Size: ${res.avg_trade_size:,.0f}")
        print(f"Maker Exit Ratio: {res.maker_exit_ratio*100:.1f}%")
        print(f"Spread Capture: {res.spread_capture_bps:.2f} bps")
        print(f"Edge (PnL / notional): {edge_bps:.2f} bps")

        plot_equity(res.equity_curve)
        return


    if choice in ["2", "4", "5"]:
        # Grid-based optimization with high-impact fixes
        grid_type = {"2": "conservative", "4": "moderate", "5": "aggressive"}[choice]
        grid = OPTIMIZATION_GRIDS[grid_type]
        
        mode_str = "PARALLEL" if use_parallel else "SINGLE-THREADED"
        print(f"\nRunning {grid_type} optimization ({mode_str}) with {len(list(product(*grid.values()))):,} combinations...")
        print(f"HIGH-IMPACT FIXES: Passive exits + 10-12 bps spreads + edge gating")
        print(f"ALL CONSTRAINTS: PnL≥0 & MaxDD≤{base.dd_cap_pct}% & AvgTradeSize≥$1,500")
        
        top_rows, diag = optimize_grid_chunked(df, base, grid, max_workers=(args.workers if use_parallel else 1), chunksize=128, topk=200)
        print(f"[Optimizer] Tested {diag['tested']:,} combos after pre-screen; elapsed {diag['elapsed_sec']:.1f}s")
        results = pd.DataFrame(top_rows)
        best = top_rows[0]["params"] if len(top_rows) else {}
            
        if best:
            best_params = SimParams(**{**asdict(base), **{k: best.get(k, getattr(base, k)) for k in grid.keys()}})
            sim = MarketMakerSim(df, best_params)
            res = sim.run()
            
            pnl_met = res.pnl_usd >= 0
            dd_met = res.max_dd_pct <= base.dd_cap_pct  
            size_met = res.avg_trade_size >= 1500
            constraints_met = pnl_met and dd_met and size_met
            
            status = "✓ ALL CONSTRAINTS MET" if constraints_met else "✗ CONSTRAINTS VIOLATED"
            if res.pnl_usd > 0:
                status += f" - SUCCESS: ${res.pnl_usd:.2f} profit!"
            
            print(f"\n{status} - BEST RESULT:")
            print(f"🎯 Traded Volume: ${res.traded_usd:,.0f} (PRIMARY OBJECTIVE)")
            print(f"💰 PnL: ${res.pnl_usd:.2f} ({'✓' if pnl_met else '✗'} target: flip from -$2)")
            print(f"📉 Max DD: {res.max_dd_pct:.2f}% ({'✓' if dd_met else '✗'} ≤{base.dd_cap_pct}%)")
            print(f"📏 Avg Trade Size: ${res.avg_trade_size:.0f} ({'✓' if size_met else '✗'} ≥$1,500)")
            print(f"🔄 Maker Exit Ratio: {res.maker_exit_ratio:.1%}")
            print(f"📊 Spread Capture: {res.spread_capture_bps:.2f} bps")
            plot_equity(res.equity_curve, f"Optimized Strategy ({grid_type})")

    if choice == "3":
        # Walk-forward optimization with high-impact fixes
        grid = OPTIMIZATION_GRIDS["moderate"]  # Trend-Hunter Grid
        
        try:
            wf_results, best_params = walk_forward_optimization(df, base, grid, use_parallel=use_parallel)
            
            if best_params:
                recent_df = df.tail(10_000).copy()
                final_params = SimParams(**{**asdict(base), **best_params})
                sim = MarketMakerSim(recent_df, final_params)
                res = sim.run()
                
                pnl_met = res.pnl_usd >= 0
                dd_met = res.max_dd_pct <= base.dd_cap_pct
                size_met = res.avg_trade_size >= 1500
                
                print(f"\nFinal validation on recent data:")
                print(f"Traded Volume: ${res.traded_usd:,.0f}")  
                print(f"PnL: ${res.pnl_usd:.2f} ({'✓ SUCCESS!' if pnl_met else '✗'})")
                print(f"Max DD: {res.max_dd_pct:.2f}% ({'✓' if dd_met else '✗'})")
                print(f"Avg Trade Size: ${res.avg_trade_size:.0f} ({'✓' if size_met else '✗'})")
                print(f"Maker Exit Ratio: {res.maker_exit_ratio:.1%}")
                print(f"Spread Capture: {res.spread_capture_bps:.2f} bps")
                print(f"Edge Gate Suppressions: {res.edge_gate_suppressions}")
                print(f"Optimized parameters:")
                for k, v in best_params.items():
                    print(f"  {k}: {v}")
                plot_equity(res.equity_curve, "Walk-Forward Optimized Strategy")
                
        except ValueError as e:
            print(f"Walk-forward error: {e}")

    if choice == "6":
        validate_fee_model()
        return

    if choice == "7":
        DEBUG_FEE_CALCULATION = True
        print("🔍 Fee calculation debugging ENABLED for next run")
        print("Re-run any option to see detailed fee calculations")
        return

    if choice == "8":
        print("🐛 Single-threaded mode enabled for debugging")
        print("Re-run optimization options for single-threaded execution")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise

"""
Final validation on recent data:
Traded Volume: $858,045
PnL: $604.20 (✓ SUCCESS!)
Max DD: 1.58% (✓)
Avg Trade Size: $789 (✗)
Maker Exit Ratio: 43.0%
Spread Capture: 17.64 bps
Edge Gate Suppressions: 3162
Optimized parameters:
  half_spread_bps: 10.0
  order_notional_frac: 0.35
  ewma_vol_span: 25
  fill_base_prob: 0.35
  skew_bps_max: 5.0
  max_net_inventory_usd: 750
  maker_exit_prob: 0.7
  adverse_mom_bps: 10
  min_edge_bps: 2.0
"""