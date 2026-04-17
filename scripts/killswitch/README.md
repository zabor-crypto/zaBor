# Kill-Switch — Automated Emergency Risk Shutdown

**Status:** Production Ready · **Version:** 2.0 · **Exchanges:** Binance / Bybit / Bitget

Monitors equity drawdown across multiple exchanges (Futures + Spot) and executes protective actions automatically when configured thresholds are breached.

---

## How it works

```
Monitor Equity (every N seconds)
  ↓
Calculate Drawdown (multiple time windows: 1m, 5m, 15m, 1h, ...)
  ↓
Tier Selection (A: conservative / B: aggressive)
  ↓
Confirmation (N consecutive snapshots must breach threshold)
  ↓
Cooldown Check (prevent rapid re-triggering)
  ↓
De-Risked Check (prevent re-liquidation loops)
  ↓
Execute Action
  ├─ Futures: CLOSE_LONGS_ONLY | CLOSE_ALL_POSITIONS
  └─ Spot:    SELL_BLACKLIST_ONLY | SELL_ALL_NON_USDT
  ↓
Record to SQLite + Set Cooldown + Set De-Risked State
```

---

## Quickstart

```bash
cd scripts/killswitch
pip install -r requirements.txt

# Configure credentials (environment variables only — never in config.yaml)
export BINANCE_API_KEY="..."
export BYBIT_API_KEY="..."
export BITGET_API_KEY="..."
# ... see .env.example in repo root

# Copy and edit config
cp config.example.yaml config.yaml

# Step 1 — validate connection (dry-run, 5 minutes)
python3 tools/dryrun_smoke.py --config config.yaml --minutes 5

# Step 2 — run in dry-run mode for 24–48 hours
python3 src/killswitch.py --config config.yaml  # dry_run: true by default

# Step 3 — switch to live (edit config.yaml: dry_run: false)
python3 src/killswitch.py --config config.yaml
```

---

## Configuration

```yaml
poll_seconds: 60
dry_run: true          # ⚠️ Always start with true
state_db: "./killswitch.sqlite"
stables_keep: ["USDT", "USDC", "DAI"]

exchanges:
  binance:
    enabled: true
    accounts:
      futures:
        enabled: true
        windows: [5, 15, "1h"]
        tier_a:
          thresholds: {"5": 0.02}      # 2% drop in 5 min
          confirm_consecutive: 2
          mode: "CLOSE_LONGS_ONLY"
          cooldown_min: 60
        tier_b:
          thresholds: {"1h": 0.05}     # 5% drop in 1 hour
          mode: "CLOSE_ALL_POSITIONS"
          cooldown_min: 120
```

Full reference: [`config.example.yaml`](config.example.yaml)

---

## Monitoring

```bash
# Health check
python3 tools/dryrun_smoke.py --config config.yaml --minutes 3

# Inspect state DB
sqlite3 killswitch.sqlite "SELECT * FROM actions ORDER BY ts DESC LIMIT 10"
sqlite3 killswitch.sqlite "SELECT * FROM scope_state"
```

---

## Test coverage

| Suite | Tests | Pass |
|-------|-------|------|
| Unit (logic) | 37 | 37 |
| Integration | 14 | 14 |
| Extended scenarios | 13 | 10 |

Scenario files in `tests/scenarios/` — deterministic JSON fixtures, no live API required.

---

## Production features

- **Atomic cooldown** — check-and-set prevents race conditions
- **Exponential backoff** — network resilience on order placement
- **Partial fill handling** — re-fetches position after each order
- **Precision enforcement** — `amount_to_precision` on all orders
- **Min-notional filter** — prevents dust rejections
- **Multi-hop spot routing** — USDT → BTC → ETH fallback
- **Circuit breaker** — alerts on N consecutive failures
- **Independent tier cooldowns** — Tier A and B do not share state
- **De-risked state tracking** — persisted across restarts

---

## Documentation

- [`docs/USER_GUIDE_RU.md`](docs/USER_GUIDE_RU.md) — full user guide (RU)
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — systemd / screen deployment
- [`docs/SECURITY.md`](docs/SECURITY.md) — API key security guidelines
- [`docs/ENV_EXAMPLE.md`](docs/ENV_EXAMPLE.md) — environment variable reference

---

## License

MIT — see repo root [LICENSE](../../LICENSE).
