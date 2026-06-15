# Changelog

## v7.0 — Regime Guard (slow-bleed / cumulative-drawdown protection)

**The problem v7.0 solves:** the v6.0 fast-crash stages require a sharp move (≥4.5%/15m …). On a real
deposit a **−50.7% drawdown accumulated as a slow grind** and the kill-switch fired **zero** times.
v7.0 adds a second, **additive** contour — a portfolio/regime-level guard that runs every cycle
alongside (not instead of) the existing fast stages.

### New: the Regime Guard (`regime_guard.py`)
Account-level, **close-only** (never flips, no blanket per-trade stop). It acts only when the *market
regime* turns against the book, and closes the positions unfavourable for that regime.

- **L0 — catastrophe cap.** Any single position losing more than `max_loss_pct_equity` (default **8%**)
  of account equity is closed immediately, in any regime. Concentration/ruin protection — catches an
  idiosyncratic single-name blow-up (e.g. a coin pumped against a short) that the portfolio triggers
  miss. NOT a per-trade stop: it fires ~once in 180 positions in real history.
- **L2 — portfolio peak-drawdown.** Equity ≤ −5% from its rolling 48h peak, sustained → close the
  dominant losing side.
- **L3 — correlated cluster.** ≥4 same-side positions all ≤ −6% on margin → close that side. A single
  position is left for its own bot; a *cluster* is the regime signal.
- **L4 — daily loss.** Equity ≤ −6% from the UTC day-open → close the dominant losing side.
- **Drawdown-velocity selection.** When a trigger fires it closes the **top-3 fastest-bleeding**
  positions (blended 15m+1h+4h bleed-rate), not all and not the biggest-loss — validated as the best
  selector.
- **Side-aware adverse-macro gate.** L2/L3 fire only when the macro is confirmed against the side:
  longs when BTC 7d < 0, shorts when BTC 7d > 0. Stops the guard flushing recoverable dips inside a
  slow trend. L4 (daily hard loss) is always-on.
- **Symmetric.** Longs and shorts handled identically (selection + gates mirrored).

### New: deploy guardrails
- **`log_only`** — the guard computes the full decision and logs / Telegrams "WOULD close X" but
  executes nothing (the fast stages keep their own `dry_run`). For a safe live observation period.
- **`reentry_cooldown_min`** — after the guard *actually* closes a `(symbol, side)`, it re-closes that
  position if a bot re-opens it within the window. Stops the guard and the bots fighting each other.
  **Live-only by design:** in `log_only` mode nothing is actually flushed, so the re-entry guard is
  disabled there (otherwise it would mistake an always-open position for a re-opened one and emit a
  "WOULD close" every cycle). The window anchors to the first flush and never self-renews.
- **`max_closes_per_day`** — caps guard-initiated closes per UTC day (runaway backstop).

### Validation
Backtested by **counterfactual replay of the account's own 60-second history** (not synthetic): MDD
−50.7% → ~−26%, positive/neutral across uptrend / chop / downtrend (the uptrend checked on a
reconstructed window incl. a +8% BTC bull). Fail-safe: any guard error is swallowed so the fast stages
are never affected.

### Other
- `position_store.py`: added `continuous_underwater_hours` + `composite_bleed_rate`.
- Adapter: `get_macro_returns` (BTC 24h/6h/7d, 5-min cache) for the gates.
- Tests: +`tests/test_regime_guard.py`, +`tests/test_guardrails.py` (now 130+ tests total).
- `requirements.txt`: add `pytest`.
- L1 (a blanket per-position margin stop) exists but is **OFF by default** — it overrides each bot's
  own exit logic; use L0 (the account-level cap) instead.

### Backward compatibility
Fully additive. If `regime_guard` is absent from the config, behaviour is identical to v6.0. The
existing `stage_1/2/3` (and legacy `tier_a/tier_b`) fast-crash logic is unchanged.

---

## v6.0 — Stage-based risk-attribution engine
- Replaced the 2-tier blunt logic with 3 stages + PnL-**delta** attribution (which side is bleeding),
  composite risk ranking, surgical top-N partial closes, one-way stage escalation, trading-lock file
  for external bots, safe spot multi-hop (delta-only) routing. See `README_RU.md` / `README_EN.md`.
