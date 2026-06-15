# killswitch-crypto

**Automated emergency risk management for multi-exchange crypto trading.**

Stop blindly closing everything when drawdown hits. Killswitch identifies *which side caused the loss*, ranks positions by risk, and closes surgically — escalating to more aggressive stages only if the situation worsens.

**Version:** 7.0 — fast-crash stages + Regime Guard for slow-bleed protection  
**Status:** Requires dry-run / log-only validation before live use (see [QUICKSTART.md](QUICKSTART.md))

See [CHANGELOG.md](CHANGELOG.md) for the full v7.0 changes.

---

## Why this exists

Standard kill-switches close all positions on drawdown. The problem: if you're running a hedged book (longs + shorts), closing everything converts a temporary drawdown into a realized loss on both sides. Worse, if only shorts are bleeding, closing longs removes your hedge.

Killswitch has two complementary contours:

### Contour 1 — fast-crash stages (v6.0)

For a *sharp* move, identify which side caused the loss and close it surgically, escalating only if it worsens:

| Stage | Mode | What it does |
|-------|------|-------------|
| 1 | `CLOSE_TOP_RISK_CONTRIBUTORS` | Closes top-N positions on the *losing side only*, 50% partial by default |
| 2 | `CLOSE_DOMINANT_LOSS_DIRECTION` | Closes all positions on the dominant losing direction |
| 3 | `CLOSE_ALL_POSITIONS` | Full stop: cancels entries, closes everything, cancels orphan orders |

Stages escalate upward only. Stage 3 requires manual reset.

### Contour 2 — Regime Guard (new in v7.0)

The fast stages require a sharp move (≥4.5% / 15m). But a drawdown can also accumulate as a **slow grind** — positions held underwater for days, a single coin squeezed against you — where no 15-minute threshold ever trips. The Regime Guard is a second, **additive**, portfolio-level contour that runs every cycle alongside the fast stages. It is **close-only** (never flips, never imposes a blanket per-trade stop) and acts only when the *market regime* turns against the book.

| Layer | Trigger | Action |
|-------|---------|--------|
| **L0** catastrophe cap | Any single position loses > 8% of account equity | Close that position immediately, any regime |
| **L2** portfolio peak-drawdown | Equity ≤ −5% from rolling 48h peak, sustained | Close the dominant losing side |
| **L3** correlated cluster | ≥4 same-side positions all ≤ −6% on margin | Close that side |
| **L4** daily loss | Equity ≤ −6% from UTC day-open | Close the dominant losing side |

Three design choices distinguish it:

- **Drawdown-velocity selection** — closes the top-3 *fastest-bleeding* positions (blended 15m+1h+4h bleed-rate), not the biggest current loss.
- **Side-aware macro gate** — L2/L3 fire only when BTC macro is confirmed against the side (longs when BTC 7d < 0, shorts when BTC 7d > 0), so the guard doesn't flush recoverable dips inside a slow trend. L4 is always-on.
- **Symmetric** — longs and shorts handled identically.

Validated by counterfactual replay of a real account's own 60-second history: max drawdown roughly halved, positive/neutral across uptrend / chop / downtrend.

**Rollout guardrails:** `log_only` (compute + log/Telegram "WOULD close X", execute nothing), `reentry_cooldown_min` (stop guard↔bot churn), `max_closes_per_day` (runaway backstop). Fully additive — if `regime_guard` is absent from the config, behaviour is identical to v6.0.

---

## Quickstart (≤5 commands)

```bash
git clone https://github.com/zabor-crypto/killswitch-crypto
cd killswitch-crypto
pip install -r requirements.txt
cp .env.example .env && nano .env        # fill in your API keys
python3 killswitch.py --test-mock        # must pass before anything else
```

See [QUICKSTART.md](QUICKSTART.md) for the full dry-run → live deployment flow.

---

## Architecture

```
Equity Drawdown Detected
        ↓
Position Snapshots Fetched + Stored in SQLite
        ↓
PnL Delta Attribution
  (delta = current_pnl − reference_pnl; a declining profitable position counts)
  → source: SHORT | LONG | MIXED | UNKNOWN
        ↓
Risk Score per Position
  = 0.40 × |pnl_delta| + 0.25 × |current_loss| + 0.20 × liq_proximity
    + 0.10 × notional_share + 0.05 × margin_loss_pct
        ↓
Stage Selection — highest triggered stage wins (3 > 2 > 1)
        ↓
Stage Machine Check
  - Escalation (new > current): always allowed
  - Same stage: allowed only after cooldown expires
  - De-escalation (new < current): always blocked
        ↓
Execute Stage Action → Write trading lock file → Record state in SQLite
```

---

## Configuration

Default trigger thresholds (futures):

| Stage | 15m trigger | 1h trigger | Confirmations | Cooldown |
|-------|-------------|------------|---------------|---------|
| 1 | −4.5% | −7.0% | 3 consecutive | 30 min |
| 2 | −7.0% | −10.0% | 2 consecutive | 90 min |
| 3 | −10.0% | −14.0% | 1 | 360 min |

All thresholds, cooldowns, exchange settings, and the `regime_guard` block are in `config.yaml`. The file is heavily commented. API keys are loaded from environment variables — never hardcoded.

See `config_bitget_futures_only.yaml` for a single-exchange example with the Regime Guard and Telegram alerts enabled, and `config_stage_based_example.yaml` for a full multi-exchange template.

**Telegram alerts** are optional: set `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env` and enable the `notifications.telegram` block. The notifier uses only the Python standard library (no extra dependencies).

---

## External bot integration

When any stage fires, `killswitch_trading_lock.json` is written. Your bots must check it before placing orders:

```python
import json, os, time
lock_path = "./killswitch_trading_lock.json"
if os.path.exists(lock_path):
    lock = json.load(open(lock_path))
    if lock.get("locked") and lock.get("expires_ts", 0) > time.time():
        raise RuntimeError(f"Trading locked: {lock['reason']}")
```

---

## Monitoring

```bash
# Stage state
sqlite3 killswitch.sqlite "SELECT * FROM scope_stage_state"

# Recent actions
sqlite3 killswitch.sqlite "SELECT * FROM actions ORDER BY ts DESC LIMIT 10"

# Trading lock status
cat killswitch_trading_lock.json
```

---

## Test suite

```bash
python3 -m pytest tests/ -v
```

130+ tests across 7 suites. The system is **not ready for live use** until all tests pass in your environment.

| Suite | Coverage |
|-------|---------|
| `test_logic.py` | Drawdown calculation, window parsing, config loading |
| `test_actions.py` | Action execution paths |
| `test_attribution.py` | PnL attribution (SHORT/LONG/MIXED), risk ranking, liq proximity |
| `test_stage_machine.py` | Stage transitions, cooldown, escalation, de-escalation block |
| `test_integration.py` | End-to-end: losing shorts, losing longs, mixed, escalation, stale orders |
| `test_regime_guard.py` | L0/L2/L3/L4 layers, velocity selection, macro gates, symmetry |
| `test_guardrails.py` | log_only, re-entry cooldown, daily close cap |

---

## File structure

```
killswitch.py              # Main script
risk_attribution.py        # PnL attribution + risk ranking engine
stage_machine.py           # Stage state machine (SQLite-backed)
regime_guard.py            # Regime Guard — slow-bleed / portfolio-level protection (v7.0)
telegram_notifier.py       # Optional Telegram alerts (stdlib only, no extra deps)
order_safety.py            # CloseInstruction dataclass
trading_lock.py            # File-based trading lock
position_store.py          # Position snapshot persistence
logger.py                  # Structured logging
config.yaml                # Full multi-exchange config (dry_run: true by default)
config_bitget_futures_only.yaml   # Single-exchange example with regime_guard + Telegram
config_stage_based_example.yaml   # Annotated multi-exchange template
requirements.txt
tests/                     # 130+ tests
tools/                     # dryrun_smoke.py, e2e_demo_runner.py
```

---

## What is NOT in this public version

This is the complete production codebase. Nothing has been removed from the risk management logic. The only differences from a private deployment are:

- No actual API keys (you supply your own via `.env`)
- Example spot blacklists use generic tokens; replace with your own

---

## Known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

---

## Supported exchanges

Binance, Bybit, Bitget — futures and spot. Uses [ccxt](https://github.com/ccxt/ccxt) for exchange connectivity.

---

## Disclaimer

This software is provided for educational and research purposes. Automated trading carries substantial risk of financial loss. Test thoroughly in dry-run mode before deploying with real funds. The authors are not responsible for any trading losses.

---

## License

MIT — see [LICENSE](LICENSE).
