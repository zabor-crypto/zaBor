# scripts/bitget_order_plan.py

## Purpose
Validates a trade plan with strict safety defaults.
This is a reference CLI layout for future live trading scripts.

## Configuration
- `app.dry_run`: boolean (default true)
- `exchange.symbol`: e.g., ETHUSDT
- `risk.max_notional_usdt`: hard cap for plan size
- `plan.*`: trade intent

## Run
```bash
cp configs/zabor_example.yaml configs/local.yaml
cp .env.example .env
python scripts/bitget_order_plan.py --config configs/local.yaml
