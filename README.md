# zaBor
Production-grade crypto trading scripts: research, backtests, execution utilities, and analytics.

## Safety (READ FIRST)
This repository contains **live-trading capable** components.  
Default mode is **DRY_RUN**. You are responsible for:
- API key security
- position sizing and risk limits
- exchange-specific rules (min size, lot, tick, leverage)
- compliance with local regulations

## What’s inside
- `scripts/` runnable entry points (CLI)
- `exchanges/` exchange adapters (Bitget first)
- `strategies/` signal logic only (no exchange calls)
- `indicators/` built-in indicators for TradigView mostly
- `backtests/` deterministic backtest runners
- `analytics/` portfolio/correlation/performance tools
- `docs/` per-script documentation
- `configs/` config templates

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
