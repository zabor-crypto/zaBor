# Known Limitations

## Regime Guard (v7.0) — honest caveats

The Regime Guard is **risk insurance, not alpha**. Read these before trusting it with real execution.

- **Validated on limited regimes.** The backtest replays one real account's history (a downtrend) plus
  a *reconstructed* window containing a BTC uptrend. It is not validated across many years / many
  regimes. The first 20–30 live closes are the real out-of-sample test — roll out via `log_only: true`
  first.
- **It pays an insurance premium in a pure bull.** Even with the adverse-macro gate, in a strong uptrend
  it can occasionally cut a position that then recovers (a "false cut"). The trade-off
  protection ⇄ false-cuts is irreducible; the gate minimises it but cannot remove it.
- **Close-only + no lock can fight your bots.** The guard closes a position; your bot may re-open it on
  its next signal → churn. The `reentry_cooldown_min` guardrail mitigates this (it re-closes a
  re-appeared symbol), but the cleanest fix is for your bots to also honour the trading-lock file.
- **Execution on thin alts.** The guard sends reduce-only market orders; on illiquid names during
  stress, fills can slip well past the last mark. Size and venue matter.
- **Cross-margin makes liquidation-distance useless** as a signal — that is why L0 keys on
  *loss-as-%-of-equity*, not proximity to the (account-backed) liquidation price.
- **`log_only` is observe-only by design.** While `log_only: true`, the guard never executes — do not
  expect protection until you flip it to `false`.

---

## Spot Multi-Hop Routing — Pre-existing BTC/ETH

### Status: RESOLVED in v6.0

The previous behavior sold ALL BTC/ETH balance after hop-1 (COIN → BTC → USDT), which could liquidate pre-existing holdings.

**Current behavior (v6.0):** `sell_intermediate_delta_only: true` (default) captures the BTC/ETH balance before hop-1 and sells only `max(0, after - before)`. Pre-existing BTC/ETH is never touched.

**Config option:**
```yaml
spot_routing:
  sell_intermediate_delta_only: true           # default: safe
  allow_liquidate_preexisting_intermediate: false  # set true only intentionally
```

---

## Stage 3 Lock Requires Manual Removal

After Stage 3 fires, the trading lock file (`killswitch_trading_lock.json`) must be removed manually. `reset()` will warn and refuse if current stage is ≥ 3.

**Rationale:** Stage 3 is a full-account closure — automatic re-arming after a meltdown would be dangerous.

**To reset after manual review:**
```bash
rm killswitch_trading_lock.json
# then call reset() in code or restart with cleared SQLite scope_stage_state row
```

---

## Attribution Requires Historical Snapshots

Stage 1 and Stage 2 use PnL delta attribution — comparing current positions against a reference snapshot from the start of the trigger window.

If the process just started (no historical snapshots in SQLite), the reference is empty and attribution treats all current losses as starting from zero. This is conservative (overcounts the losing side) but may produce a less precise risk ranking for the first few polling cycles.

**Mitigation:** After running for one full polling cycle (default 60s), the reference baseline is populated.

---

## Test Suite

As of v7.0 the full suite is green: **`python3 -m pytest tests/ -v` → 130+ passed, 0 failed**
(including `test_extended.py`, `test_regime_guard.py`, and `test_guardrails.py`). Verify all pass in
your own environment before trusting results.

---

## E2E Testing Against Live Accounts

End-to-end testing (actual order execution on exchange) was not performed. All validation uses:
- Offline mock adapter (`--test-mock`)
- Dry-run mode against real exchange APIs (no orders placed)

**Recommendation:** Run dry-run for 24–48 hours before switching `dry_run: false`.

---

## VPN Required for Geo-Restricted Regions

Exchange APIs may be geo-blocked without VPN. Tested with Russian and Netherlands exit nodes.
