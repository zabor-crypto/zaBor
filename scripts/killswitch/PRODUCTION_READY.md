# Killswitch - Deployment Checklist

## Status: Requires Validation Before Live Use

**Version:** 7.0 (v6.0 fast-crash stages + Regime Guard for slow-bleed protection)
**Last updated:** 2026-06-11

---

## v7.0 — Regime Guard rollout checklist (in addition to the stage checklist below)

1. `python3 -m pytest tests/ -v` — incl. `test_regime_guard.py` + `test_guardrails.py` (130+ tests).
2. Add the `regime_guard:` block to your futures account config with **`log_only: true`**.
3. Run the kill-switch; confirm `[REGIME_GUARD] [LOG-ONLY] …` lines appear each cycle with no errors,
   and that the macro fetch (`btc7d=…`) is populated.
4. Observe 2–5 days. Verify it stays quiet in a fine macro and that its "WOULD close" picks look right.
5. Flip **`log_only: false`** and restart; monitor closely the first week.

The Regime Guard is **additive and fail-safe**: any error inside it is swallowed so the fast stages are
never affected, and if the `regime_guard` block is absent behaviour is identical to v6.0.

---

## Pre-Deployment Checklist

### 1. Run mock test

```bash
python3 killswitch.py --test-mock
```

Expected: all assertions pass with no errors.

### 2. Run unit and integration tests

```bash
python3 -m pytest tests/ -v
```

Expected: 94+ tests pass. The script is **not ready for live use** until all tests pass in your environment.

### 3. Run dry-run smoke test

```bash
source .env
python3 tools/dryrun_smoke.py --config config.yaml --minutes 5
```

Expected: ≥95% success rate for all enabled scopes, no exceptions.

### 4. Run in dry-run mode for 24–48 hours

```bash
source .env
python3 killswitch.py --config config.yaml
# Ensure dry_run: true in config.yaml
```

Monitor during this period:
- SQLite tables growing: `snapshots`, `position_snapshots`, `scope_stage_state`, `actions`
- No unexpected exceptions in logs
- Stage machine transitions logged correctly

### 5. Verify trading lock file behavior

When a stage executes, `killswitch_trading_lock.json` is written. External bots must read and respect this file before placing orders.

### 6. Go live

Only after completing steps 1–5:

```bash
# Edit config.yaml
dry_run: false

python3 killswitch.py --config config.yaml
```

---

## Test Coverage (v7.0 — 130+ tests)

| Suite | Notes |
|-------|-------|
| `tests/test_logic.py` | Drawdown + config logic |
| `tests/test_actions.py` | Action execution |
| `tests/test_attribution.py` | PnL attribution, risk ranking, liq proximity |
| `tests/test_stage_machine.py` | Stage state machine transitions |
| `tests/test_integration.py` | End-to-end scenarios A–E |
| `tests/test_regime_guard.py` | ★ Regime Guard: L0 cap, L2/L3/L4, velocity selection, side-aware gates |
| `tests/test_guardrails.py` | ★ log-only, re-entry cooldown, daily-cap (integration) |

Run `python3 -m pytest tests/ -v` to verify all pass in your environment before trusting results.

---

## Architecture Overview

```
Equity Drawdown Detected
        ↓
Position Snapshots Fetched (stored in SQLite)
        ↓
PnL Attribution (delta-based: which side caused the loss?)
        ↓
Risk Ranking (composite score: PnL delta + current PnL + liq proximity + notional)
        ↓
Stage Selection (highest triggered stage wins: 3 > 2 > 1)
        ↓
Stage Machine Check (escalation always allowed; de-escalation blocked)
        ↓
Execute Stage Action:
  Stage 1: CLOSE_TOP_RISK_CONTRIBUTORS (top-N on losing side, partial close)
  Stage 2: CLOSE_DOMINANT_LOSS_DIRECTION (all positions on dominant side)
  Stage 3: CLOSE_ALL_POSITIONS (cancel entries + close all + cancel orphans)
        ↓
Write Trading Lock File + Record Stage State
```

---

## Key Safety Properties

- **Stage escalation is one-way**: a higher stage always fires even during a lower stage's cooldown. You cannot de-escalate.
- **Stage 3 lock requires manual reset**: `reset()` prints a warning and refuses if stage ≥ 3. Remove the lock file manually.
- **Spot routing**: `sell_intermediate_delta_only: true` (default) means only the BTC/ETH received from hop-1 is sold; pre-existing holdings are preserved.
- **External bot coordination**: bots must check `killswitch_trading_lock.json` before placing orders.
- **Attribution uses PnL delta**: a position still profitable but declining counts as a drawdown contributor.

---

## Files

```
killswitch.py          # Main script (v7.0)
regime_guard.py        # ★ Regime Guard (v7.0): L0/L2/L3/L4 + velocity + gates + guardrails
risk_attribution.py    # PnL attribution + risk ranking
stage_machine.py       # Stage state machine (SQLite-backed)
order_safety.py        # CloseInstruction dataclass
trading_lock.py        # File-based trading lock for external bots
position_store.py      # Position snapshot persistence
config.yaml            # Configuration (dry_run: true by default)
logger.py              # Structured logging
requirements.txt       # Dependencies

tests/
  test_logic.py        # Drawdown + config unit tests
  test_actions.py      # Action execution unit tests
  test_attribution.py  # Attribution and risk ranking tests
  test_stage_machine.py # Stage machine tests
  test_integration.py  # End-to-end scenario tests
```
