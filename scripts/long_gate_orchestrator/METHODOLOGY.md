# Methodology — building a regime-gate orchestrator for any signal family

A repeatable recipe for answering *"when should signals of family X be traded, sized down, or suppressed?"* with a causal market-state panel and a thin per-strategy decision layer. This is the generalized version of the long-gate orchestrator in this repo.

---

## 0. Thesis — what an orchestrator is, and when it's worth building

A **causal market-state feature panel** plus a thin per-strategy decision layer:

```
decide(strategy, ts, [signal_features]) -> size_mult ∈ {0, 0.5, 1.0}
```

It does **not** touch detection or the execution engine — it sizes/suppresses at signal-emit. Build one when **regime is the dominant lever** for a family of strategies (exit / universe / indicator tuning already exhausted) and several strategies can share one feature layer. The payoff is a single validated, monitorable gate that converts regime knowledge into sizing across the whole book.

---

## 1. The pipeline (P0 → P5)

| Phase | Goal | The discipline |
|-------|------|----------------|
| **P0** recon + feature inventory | Commit an *expected-favorable-axis per strategy* table **before** any sweep. | Anti-cherry-pick: pre-register 3–5 hypotheses. |
| **P1** regenerate + settle | Re-run each strategy's detector on the **real execution venue** corpus; settle with its **deployed** exit engine into one unified `{strategy, symbol, ts, realized_R, native_features}` table. | Regenerate — don't reuse legacy CSVs settled on a different venue. Reuse live detectors/engines for faithful settlement. |
| **P2** regime panel | One causal market-state timeseries, `shift(1)`, keyed by `known_at = bar_open + interval`. | Universal — build it once, reuse across families. Never reference a signal's own entry indicator. |
| **P3** sweep + refine | Cache `realized_R` (P1); vectorized gate lookups; finetune continuous thresholds + per-strategy 2-D combos. | Rank by a **bounded maximin-mean-R objective + retention guard**, never pooled PF. Plateau-not-point; dimensions separate, no cross-product. |
| **P4** validate | WFO-select-on-train → score next-unseen fold; clean last-25% holdout; label-permutation placebo; episode honesty (remove-best-month). | Must beat ungated **and** each existing gate out-of-sample; placebo p<0.05; not one-episode-carried; the OOS haircut is real. |
| **P5** wire + monitor | `decide()` API (parity-verified vs P4) → gate at the source → A/B monitor + kill-switch. | Shadow-first for real money; gate-at-source so consumers inherit; fail-open; instant rollback. |

---

## 2. The regime panel (P2)

One market-state timeseries on a fixed cadence (here 4h). Every value is tagged with `known_at` — the wall-clock time it becomes causally available (bar close) — and attached to signals downstream via `merge_asof(known_at <= signal_ts)`. Representative features:

- **`btc_r3`** — BTC trailing 3-day return. A smooth trend-state axis; needs no extra lag. The *backbone*.
- **`riskoff_z`** — a continuous cross-sectional risk-off composite (dispersion − BTC drift + negative breadth, each trailing-z-scored). Higher = more risk-off.
- **`btc_ema55_up`** — BTC in an EMA55 uptrend (price > EMA55 and EMA55 rising). The breakout self-gate.
- **`breadth`** — fraction of the universe trading well below its short SMA (a stress proxy).
- **`|btc_trend_z|`** — trend-strength magnitude.

Principle: smooth state axes (like `btc_r3`) need no lag; noisy flags do — prefer the smooth axis, and lag the noisy ones explicitly (the sweep tests lags 4/8/12/16/24h).

---

## 3. The validation battery (P4) — the overfit killer

A gate config is only real if it passes **all** of:

- **(a) Walk-forward** — the threshold chosen on train folds beats ungated + existing on the next-unseen fold, and the chosen threshold is **stable** (drift across folds is an overfit tell).
- **(b) Clean holdout** — positive on the last 25% by time, which the tuner never saw.
- **(c) Placebo** — permute `realized_R` within strategy; the gate's traded subset must beat random same-size subsets (p<0.05). This isolates genuine selection skill from luck.
- **(d) Episode honesty** — per-month gated vs ungated, and remove-best-month still beats ungated. Count *episodes* (a 6-month window is only ~4–8 BTC swings), not days.

A negative worst-fold is normal — the gate shrinks the bad-regime loss, it doesn't erase it. A clean placebo + holdout + a stable WFO pick is the bar. Expect out-of-sample to be ~40–50% of the in-sample headline, and say so.

---

## 4. The deploy pattern (P5) — gate at the source

- **Gate before the signal leaves the scanner.** Suppressed = never emitted = no consumer trades it. Consumers install nothing.
- **Publish cross-cutting regime features once** (one publisher writes a small `regime_state.json`); a fail-**open** helper reads it (if the panel is stale/missing, trade ungated rather than block everything). Native per-signal features stay in the bot.
- **Real money → shadow-first.** An env flag logs `would_skip` on every signal (the live A/B) before it enforces; flip to enforce once the live A/B confirms; instant rollback by flipping the flag back.
- **A/B monitor (daily).** Settle the last-N-days signals, report kept vs skipped-and-avoided vs ungated, plus a kill-switch (gated PF < 1.0 over ≥ N closes). The *skipped/avoided* arm is how you adjudicate "is the gate wrongly cutting good signals?" live.

---

## 5. Meta-lessons

1. Regenerate on the real venue; reuse live detectors/engines for faithful settlement.
2. The regime panel is a durable, reusable asset — build it once, gate many families.
3. Rank by a bounded objective + retention guard, never pooled PF (it rewards over-narrow overfit gates).
4. In-sample tuning is free; the out-of-sample battery is the gate. Expect (and report) a ~40–50% haircut.
5. Compose with a specialist's native edge feature — don't let return-maximization delete it.
6. Smooth state axes need no lag; noisy flags do — prefer the smooth axis.
7. Gate at the source so consumers inherit; bridge cross-cutting features once; fail-open.
8. Build the A/B monitor's skipped/avoided arm from day one — it's how you catch a gate cutting good signals.
