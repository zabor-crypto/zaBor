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
| `scripts/killswitch/` | Automated emergency risk shutdown — closes positions across Binance/Bybit/Bitget when drawdown thresholds are breached |
| `exchanges/bitget/` | Typed Bitget client adapter (REST, order lifecycle) |
| `indicators/tradingview/` | Pine Script indicators for TradingView signal prototyping |
| `configs/` | Config templates (YAML) |
| `docs/` | Per-component documentation |

> `strategies/`, `backtests/`, `analytics/` — in progress; added as components are battle-tested.

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

To run the kill-switch in dry-run mode:

```bash
cd scripts/killswitch
cp config.example.yaml config.yaml
# Fill in API keys via environment variables (see docs/ENV_EXAMPLE.md)
python3 src/killswitch.py --config config.yaml   # dry_run: true by default
```

---

## Components

### Kill-Switch (`scripts/killswitch/`)

Monitors equity drawdown across multiple exchanges and account types (Futures + Spot). On threshold breach, executes configurable protective actions:

- **Tier A** — conservative: close longs only, cool down 60 min
- **Tier B** — aggressive: flatten all positions, sell non-stable spot

Supports: Binance, Bybit, Bitget · REST/WebSocket · SQLite state DB · Structured JSON logging

→ See [`scripts/killswitch/README.md`](scripts/killswitch/README.md)

---

### TradingView Indicators (`indicators/tradingview/`)

Pine Script indicators for signal research and visual chart validation.

**zaBor RSI + AO + Stochastic Entry System** — structured BUY/SELL signal overlay:
- RSI regular & hidden divergences
- Awesome Oscillator momentum filter
- Stochastic reversal trigger
- Cooldown anti-spam

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
- Backtesting results and performance data
- Position history, logs, or runtime state
- Private configuration files (`.env`, `config.yaml` with real keys)

---

## Requirements

- Python 3.10+
- ccxt, pydantic, tenacity, PyYAML, pandas, numpy, rich
- Exchange API keys with **trade-only** permissions (no withdrawal)

---

## License

MIT — see [LICENSE](LICENSE).  
TradingView Pine Script components carry their own license — see indicator subdirectory.
