# Quick Start Guide

## Installation

```bash
pip install ccxt pyyaml tenacity
```

## Configuration

Create `.env` file:
```bash
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."
export BITGET_API_KEY="..."
export BITGET_API_SECRET="..."
export BITGET_PASSWORD="..."
```

## Step 1 — Run Tests

```bash
python3 killswitch.py --test-mock
python3 -m pytest tests/ -v
```

Both must pass before proceeding. If tests fail, do not deploy.

## Step 2 — Pre-flight Dry-Run Smoke Test

```bash
source .env
python3 tools/dryrun_smoke.py --config config.yaml --minutes 5
```

Expected: Success rate ≥95% for all enabled scopes.

## Step 3 — Start in Dry-Run Mode

```bash
source .env
python3 killswitch.py --config config.yaml
```

Ensure `dry_run: true` in config.yaml (this is the default).

Run for 24–48 hours and monitor:
- SQLite snapshots growing (`snapshots`, `position_snapshots`)
- Stage state table: `SELECT * FROM scope_stage_state`
- No unexpected exceptions
- Trading lock file is written if a stage fires: `cat killswitch_trading_lock.json`

## Step 4 — Go Live

After successful dry-run:

1. Edit config.yaml: `dry_run: false`
2. Restart: `python3 killswitch.py --config config.yaml`
3. Monitor intensively first week

## Monitoring

```bash
# Database checks
sqlite3 killswitch.sqlite "SELECT COUNT(*) FROM snapshots"
sqlite3 killswitch.sqlite "SELECT COUNT(*) FROM position_snapshots"
sqlite3 killswitch.sqlite "SELECT * FROM scope_stage_state"
sqlite3 killswitch.sqlite "SELECT * FROM actions ORDER BY ts DESC LIMIT 5"

# Trading lock status
cat killswitch_trading_lock.json
```

## Stage Architecture (Futures)

```
Stage 1 — CLOSE_TOP_RISK_CONTRIBUTORS
  Closes top-N positions on the losing side (partial close, 50% default)
  Trigger: 4.5% in 15m or 7% in 1h (3 consecutive confirmations)
  Cooldown: 30 min

Stage 2 — CLOSE_DOMINANT_LOSS_DIRECTION
  Closes all positions on the dominant losing side
  Trigger: 7% in 15m or 10% in 1h (2 consecutive confirmations)
  Cooldown: 90 min

Stage 3 — CLOSE_ALL_POSITIONS
  Cancels all entry orders, closes all positions, cancels orphan orders
  Trigger: 10% in 15m or 14% in 1h (1 confirmation)
  Cooldown: 360 min
  Lock: requires manual removal
```

Stages escalate upward only. A higher stage always fires even during a lower stage's cooldown.

## External Bot Integration

When any stage fires, `killswitch_trading_lock.json` is written. Your trading bots must check this file before placing orders:

```python
import json, os
lock_file = "./killswitch_trading_lock.json"
if os.path.exists(lock_file):
    with open(lock_file) as f:
        lock = json.load(f)
    if lock.get("locked"):
        print(f"Trading locked: {lock['reason']}")
        return  # do not trade
```

## Backward Compatibility

Existing configs using `tier_a`/`tier_b` continue to work. They are auto-mapped:
- `tier_a` → `stage_1` (if no explicit `stage_1`)
- `tier_b` → `stage_3` (if no explicit `stage_3`)

`CLOSE_LONGS_ONLY` mode triggers a deprecation warning; use `CLOSE_TOP_RISK_CONTRIBUTORS` in new configs.
