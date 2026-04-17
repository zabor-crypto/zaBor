# zaBor

> Production-grade systematic crypto trading toolkit: execution, risk controls, research, and TradingView indicators.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Safety (READ FIRST)

This repository contains **live-trading capable** components.  
Default mode is **`dry_run: true`**. You are responsible for:

- API key security and IP whitelisting
- Position sizing and risk limits
- Exchange-specific rules (min size, lot size, tick, leverage caps)
- Compliance with local regulations

**Never run in live mode without a dry-run validation period (24–48 hours minimum).**

---

## What's inside

| Path | Description |
|------|-------------|
| [`scripts/killswitch/`](scripts/killswitch/) | Automated emergency risk shutdown — monitors equity drawdown across Binance/Bybit/Bitget and flattens positions on threshold breach |
| [`scripts/lighter_mm/`](scripts/lighter_mm/) | Market-making strategy simulator for Lighter.xyz DEX — spread quoting, inventory skew, toxicity filter, walk-forward optimization |
| [`exchanges/bitget/`](exchanges/bitget/) | Typed Bitget REST client adapter (order lifecycle, credentials, retries) |
| [`indicators/tradingview/`](indicators/tradingview/) | Pine Script indicators for TradingView signal prototyping |
| [`configs/`](configs/) | YAML config templates |

---

## Quickstart

```bash
git clone git@github.com:zabor-crypto/zaBor.git
cd zaBor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

---

## Scripts

### Kill-Switch (`scripts/killswitch/`)

Monitors equity drawdown across multiple exchanges (Futures + Spot). On threshold breach executes configurable protective actions:

- **Tier A** — conservative: close longs only, cooldown 60 min
- **Tier B** — aggressive: flatten all positions, sell non-stable spot

Exchanges: Binance, Bybit, Bitget · SQLite state DB · Structured JSON logging

```bash
cd scripts/killswitch
cp config.example.yaml config.yaml
# set API keys via environment variables
python3 src/killswitch.py --config config.yaml   # dry_run: true by default
```

→ See [`scripts/killswitch/README.md`](scripts/killswitch/README.md)

---

### Lighter MM Simulator (`scripts/lighter_mm/`)

Bar-by-bar simulation of a spread-based market-making strategy calibrated to [Lighter.xyz](https://lighter.xyz) DEX mechanics (zero protocol fees, on-chain order book).

Key mechanics: ATR-scaled spread · inventory skew · momentum toxicity filter · maker exit ladder · edge gate · daily drawdown stop

```bash
cd scripts/lighter_mm
pip install -r ../../requirements-sim.txt

# Full simulation on included sample data
python mm_sim.py --data data/sol_1m_sample.csv --full-run

# Walk-forward parameter optimization
python mm_sim.py --data data/sol_1m_sample.csv --walk-forward
```

→ See [`scripts/lighter_mm/README.md`](scripts/lighter_mm/README.md)

---

## TradingView Indicators (`indicators/tradingview/`)

Pine Script indicators for signal research and visual chart validation.

**zaBor RSI + AO + Stochastic Entry System** — structured BUY/SELL signal overlay:
- RSI regular & hidden divergences
- Awesome Oscillator momentum filter
- Stochastic reversal trigger with cooldown anti-spam

→ See [`indicators/tradingview/zaBor_RSI_AO_Stoch_Entry_System/README.md`](indicators/tradingview/zaBor_RSI_AO_Stoch_Entry_System/README.md)

---

## Configuration

All sensitive values are passed via environment variables — never hardcoded.

```bash
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."
export BITGET_API_KEY="..."
export BITGET_API_SECRET="..."
export BITGET_PASSWORD="..."
```

See [`.env.example`](.env.example) for the full list.

---

## What is intentionally NOT in this repository

- Live strategy logic and signal parameters (alpha)
- Backtesting results and performance metrics
- Position history, trade logs, or runtime state
- Private configuration files (`.env`, `config.yaml` with real keys)

---

## Requirements

- Python 3.10+
- Per-script dependencies listed in each `scripts/*/requirements.txt`
- Exchange API keys with **trade-only** permissions (no withdrawal)

---

## License

MIT — see [LICENSE](LICENSE).  
TradingView Pine Script components carry their own license — see indicator subdirectory.
