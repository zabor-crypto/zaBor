# lighter-mm-sim

> Bar-by-bar market-making strategy simulator for [Lighter.xyz](https://lighter.xyz) DEX, written in Python.

Simulates a spread-based MM strategy on SOL/USD using 1-minute OHLCV data.
Includes walk-forward parameter optimization with out-of-sample constraint validation.

---

## Why this exists

Lighter.xyz is an on-chain order-book DEX with **zero protocol fees** — the only cost is adverse selection and slippage. This simulator explores whether a simple quoting strategy can generate net-positive PnL under those conditions, and at what parameter settings.

**Honest result:** the strategy is marginally profitable in some regimes but not robustly so across all walk-forward windows. The simulator is the artifact — the research process, not a finished signal.

---

## Quickstart

```bash
git clone https://github.com/zabor-crypto/lighter-mm-sim.git
cd lighter-mm-sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run full simulation on sample data
python mm_sim.py --data data/sol_1m_sample.csv --full-run

# Walk-forward optimization (parallel)
python mm_sim.py --data data/sol_1m_sample.csv --walk-forward

# Conservative grid search
python mm_sim.py --data data/sol_1m_sample.csv --opt
```

---

## Strategy mechanics

### Quoting

Each bar, the bot places a bid and ask around the mid-price:

```
bid = mid × (1 − (half_spread + max(0,  skew)) / 10000)
ask = mid × (1 + (half_spread + max(0, −skew)) / 10000)
```

`half_spread` is ATR-scaled with a floor: `max(spread_floor_bps, k_spread × ATR_bps)`

### Inventory skew

Signed inventory fraction drives `skew` — widening the side that would increase
exposure, narrowing the side that reduces it. Max skew is configurable.

### Toxicity filter

Momentum over a short lookback window is computed in bps. If it exceeds
`tox_thr_bps`, the corresponding side is suppressed (no quote placed).
This reduces adverse selection from strongly directional flow.

### Maker exit ladder

Fills open a position. Exit is attempted passively at 3 price levels
(configurable multipliers of half-spread). If not filled within `exit_time_cap_bars`,
the position is closed at market (taker).

### Edge gate

Before placing a quote, the expected profit is estimated. If it falls below
`edge_gate_min_bps`, the side is skipped. This prevents quoting in regimes
where the spread doesn't cover adverse selection.

### Risk controls

| Control | Parameter | Default |
|---------|-----------|---------|
| Daily loss stop | `daily_loss_stop_pct` | 5% |
| Max drawdown cap | `dd_cap_pct` | 20% |
| Drawdown watchdog | `dd_watchdog_pct` | 10% |
| Cooldown on breach | `dd_cooldown_bars` | 60 bars |

---

## Lighter.xyz fee model

```python
maker_fee_bps    = 0.0   # zero protocol fees
taker_fee_bps    = 0.0   # zero protocol fees
gas_fee_per_trade = 0.0  # batched on-chain, negligible
min_order_usd    = ~400  # minimum notional
```

Only costs modeled: slippage on taker exits and adverse selection on entries.

---

## Walk-forward optimization

```
Step N:  Train [n×step : n×step + train_size]
         Test  [n×step + train_size : n×step + train_size + test_size]
```

Default: 30,000 train / 10,000 test bars, 5 rolling steps.

Each step selects the best parameter set on training data and evaluates it
out-of-sample with constraints:

- PnL ≥ $0
- MaxDD ≤ `dd_cap_pct`
- AvgTradeSize ≥ $400

Activity regimes (HIGH / MEDIUM / LOW) are tracked per step to understand
turnover degradation in later windows.

---

## Data format

CSV with columns: `timestamp, open, high, low, close, volume`

```csv
timestamp,open,high,low,close,volume
2025-10-14 21:00:00,200.61,200.76,200.53,200.72,2628.03
```

`data/sol_1m_sample.csv` — 10,000 bars of SOL/USD 1-minute data (included).

To use your own data:
```bash
python mm_sim.py --data /path/to/your/data.csv --full-run
```

---

## CLI reference

| Flag | Description |
|------|-------------|
| `--data PATH` | Path to OHLCV CSV (default: sample data) |
| `--full-run` | Run single simulation on full dataset with calibrated params |
| `--opt` | Conservative grid search |
| `--walk-forward` | Walk-forward optimization (recommended) |
| `--dd-cap FLOAT` | Override max drawdown cap (%) |
| `--exchange {lighter,ostium}` | Exchange fee profile |
| `--workers N` | Parallel workers for grid search (default: CPU−1) |
| `--single-thread` | Disable parallelism (debug mode) |
| `--debug-fees` | Print detailed fee breakdown per trade |

---

## Key parameters (`SimParams`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `equity_init_usd` | 301.0 | Starting collateral (USDC) |
| `leverage` | 11.0 | Fixed leverage |
| `spread_floor_bps` | 12.0 | Minimum half-spread |
| `k_spread` | 0.5 | ATR multiplier for spread |
| `skew_bps_max` | 4.0 | Max inventory skew |
| `tox_thr_bps` | 10.0 | Momentum threshold for toxicity suppression |
| `exit_time_cap_bars` | 15 | Bars before forced market exit |
| `edge_gate_min_bps` | 1.2 | Minimum expected edge to quote |
| `daily_loss_stop_pct` | 5.0 | Intraday loss limit |

---

## Disclaimer

This is a **research simulator**. Simulated results do not reflect live trading performance.
The strategy has not been validated as consistently profitable across market regimes.
Do not deploy capital without independent validation on live data.

---

## License

MIT — see the [repository LICENSE](../../LICENSE).
