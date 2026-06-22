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
| [`scripts/alt_4h_scanner/`](scripts/alt_4h_scanner/) | Binance spot scanner — 4H accumulation breakout detection with 9-filter stack, computes structured limit-order levels for pullback entries |
| [`scripts/killswitch/`](scripts/killswitch/) | Emergency risk management system v7.0 — PnL attribution + staged closure plus a portfolio-level Regime Guard for slow-bleed drawdowns, Binance/Bybit/Bitget, 139 tests |
| [`scripts/funding_arb_research/`](scripts/funding_arb_research/) | Funding rate arbitrage research stack — async collectors for 6 venues, strategy-agnostic backtest engine, 8 strategies evaluated |
| [`scripts/long_gate_orchestrator/`](scripts/long_gate_orchestrator/) | Causal regime-gating layer for swing-long strategies — enable in favorable regimes, suppress the downtrend tail; shared regime panel + per-strategy thresholds, full WFO/placebo validation battery |
| [`scripts/liquidation_signal_research/`](scripts/liquidation_signal_research/) | Binance liquidation cascade signal — event-sourced paper-trading pipeline, microstructure features, 22K paper trade validation. Research in progress. |
| [`scripts/lighter_mm/`](scripts/lighter_mm/) | Market-making strategy simulator for Lighter.xyz DEX — spread quoting, inventory skew, toxicity filter, walk-forward optimization |
| [`exchanges/bitget/`](exchanges/bitget/) | Typed Bitget REST client adapter (order lifecycle, credentials, retries) |
| [`indicators/tradingview/`](indicators/tradingview/) | Pine Script indicators — entry-signal overlay + visual market-context dashboard |
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

### Kill-Switch v7.0 (`scripts/killswitch/`)

Emergency risk management system with two complementary contours. The **fast-crash stages** identify which side caused a sharp loss (longs vs shorts), rank positions by a composite risk score, and close surgically — escalating only if it worsens. The **Regime Guard** (v7.0) adds a portfolio-level contour for the *slow bleed* the fast stages never see: positions held underwater for days, or a single coin squeezed against you. Close-only, with drawdown-velocity selection and a side-aware macro gate.

**Stages:** 1 → close top-N risk contributors · 2 → close dominant losing direction · 3 → full stop (manual reset required)  
**Regime Guard:** L0 catastrophe cap · L2 peak-drawdown · L3 correlated cluster · L4 daily loss · log-only rollout mode  
**Exchanges:** Binance, Bybit, Bitget · Futures + Spot · SQLite state · 139 tests

```bash
cd scripts/killswitch
cp .env.example .env  # fill in API keys
python3 killswitch.py --test-mock
python3 -m pytest tests/ -v
python3 killswitch.py --config config.yaml   # dry_run: true by default
```

→ See [`scripts/killswitch/README.md`](scripts/killswitch/README.md)

---

### Funding Rate Arbitrage Research (`scripts/funding_arb_research/`)

Multi-venue funding rate arbitrage research stack. Async data collectors for 6 venues (Binance, Bybit, OKX, Hyperliquid, Bitget, GMX v2), strategy-agnostic backtest engine, 8 strategies tested across multiple hypotheses.

**Honest result:** all strategies REJECT at current fee levels. Research infra is reusable for further hypotheses.

```bash
cd scripts/funding_arb_research
pip install -r requirements.txt
cp .env.example .env
python3 main_collect.py      # start data collection
python3 main_backtest.py     # run backtest
```

→ See [`scripts/funding_arb_research/README.md`](scripts/funding_arb_research/README.md)

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

**Crypto Context Dashboard** — visual present-state context panel (not a strategy):
- Trend / MTF agreement / volatility / extension / candle-noise context in a compact table
- Causal, trailing-only formulas; non-repainting higher-timeframe context
- Descriptive only — no trade calls, no forecasts; neutral state-change alerts

→ See [`indicators/tradingview/crypto_context_dashboard/README.md`](indicators/tradingview/crypto_context_dashboard/README.md)

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
