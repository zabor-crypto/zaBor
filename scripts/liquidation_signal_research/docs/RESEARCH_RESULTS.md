# Liquidations v1 — Full Project Deconstruction

**Status:** Phase 3 closed 2026-05-27. Cross-venue execution committed in parallel. Internal architecture redesign initiated (same signal, new execution model).

**Purpose of this document:** complete, single-source post-mortem of the v1 paper-trading system. Captures architecture as built, every phase of validation, what works, what doesn't, why gross PnL is structurally zero, and what to try next. Read sequentially; sections are self-contained.

---

## 1. Executive summary (one page)

The v1 liquidation paper bot has been live since 2026-03-22. It collected ~22,000 paper trades over the 7-day regime-soft window (May 20–27), plus ~457 trades over the pre-regime-soft baseline (Mar 22 – Apr 25). Total >22,000 trades, statistically conclusive.

**Headline numbers (7-day, n=21,816):**
- WR = 62.7%
- Gross = -0.061 bps (95% CI [-0.143, +0.022]) — statistical zero
- Net = -$1,717
- Fees = $1,702 (∼equal to net loss; fees ARE the loss)
- Daily gross stays within ±0.16 bps across all 7 consecutive days

**Core finding:**

> The v1 entry signal has no measurable forward-return edge at any horizon between 500 ms and 5 min. Position management (trailing stop) lifts realized WR from 47% (random) to 63%, but doesn't generate gross PnL — gains from trailing stops are exactly cancelled by losses from hard stops. The 8 bps Binance fee wall therefore equals the entire net cost.

**What this implies:**

| Lever | Evaluated? | Outcome |
|---|---|---|
| Looser/tighter score threshold | Yes (Recipe 5) | r=-0.006 — score is binary gate only |
| Funding/OI conditioning | Yes (Recipe 3) | <0.15 bps spread — no signal |
| Maker mode (saves ~4 bps RT) | Yes (Recipe 4) | 0.141 bps savings — wouldn't matter |
| Per-symbol filtering | Yes | HYPE 70% +0.03 bps; SOL hurts; BTC/ETH small + |
| Hard_stop loosening | Calibrated Mar–Apr | tighter = more stops, looser = bigger losses |
| Cross-venue (Bybit/Bitget) | Phase 2.5 Bybit done | Mechanically viable; basis noise 4–7 bps > edge |
| ML quality gate | Disabled in test | Not enough ENTERs to retrain |

**All Binance-internal levers exhausted.** Two open tracks:

1. **Cross-venue execution** (Phase 5): use lower-fee venue. Mechanically viable but basis noise ≈ 5 bps ≫ v1 edge ≈ 0. Cross-venue helps only if fee advantage absorbs both basis penalty AND dead signal.
2. **Architecture redesign** (this branch): same liquidation+flow signal, new execution model — reduce fee count by aggregation, asymmetric exits, event-driven holds, symbol-portfolio optimization. Goal: find a structural way to extract edge from data we already have.

Both tracks run in parallel. v1 paper bot continues as data collector.

---

## 2. Project history & timeline

### Pre-history (2025–Feb 2026)
- LiveBot_2 (live trading on Binance, paused in 2026): fees + WR fatal
- liq_v2 WFO: walk-forward optimization, invalid backtest bug
- liq_v3 through liq_v6: each strategy iteration superseded by the next
- liq_v8 scalp: separate scalping experiment, abandoned

### v1 evolution (Mar 2026 → present)

| Date | Milestone |
|---|---|
| 2026-03-22 | v1 paper bot deployed. Liquidation cascade detection + flow-imbalance + depth-collapse. Strict regime gates. ML quality gate enabled. |
| 2026-03–04 | Position-manager calibration sweep. Multiple exit parameter iterations. |
| 2026-04-25 | 457-trade baseline complete: WR=5.3%, gross≈0, fees=-$24, net=-$24.87. Conclusion: fees structural. |
| 2026-04-28 | Major exit recalibration: HARD_STOP=8bps, TRAILING_ACTIVATE=2bps, TRAIL=1bps, FLOW_REVERSAL_COUNT=20, liq_notional≥$300K entry filter |
| 2026-05-11 | FAPI fix (Binance API stability) |
| 2026-05-11 | Strategic assessment: gross≈0, fee wall structural, ML not ready (19 features vs 19 ENTERs), recommend pivot |
| 2026-05-18 → 20 | Deploy flap: removed one filter thinking it was THE liq gate, missed 2 upstream gates. 53 hours of silent zero ENTERs. Recovered with regime-soft + ML-off env flags. |
| 2026-05-20 09:57 UTC | "Regime-soft" deploy: REGIME_LIQ_OPTIONAL=1, REGIME_SESSION_OPTIONAL=1, DISABLE_ML_QUALITY_GATE=1. Bot starts generating trades at ~3,000/day. |
| 2026-05-20 12:57 UTC | Data enrichment deployed: TOB-at-close, funding/OI poller, forward-return backfill, Bybit capture. |
| 2026-05-21 10:44 UTC | KILL_SWITCH_MAX_DRAWDOWN_PCT env var added (drawdown_pct unit-mismatch bypass). Bot back to continuous run. |
| 2026-05-23 | Phase 2 (72h, n=9,269): cross-venue track soft-locked. |
| 2026-05-25 | Phase 3 5-day soft lock (n=15,808). Verdict: cross-venue. |
| 2026-05-26 | Phase 2.5 Bybit gap analysis: BTC/ETH viable, SOL not. Bitget capture deployed. |
| 2026-05-27 | Phase 3 7-day HARD lock (n=21,816). Cross-venue confirmed; redesign branch opened in parallel. |

---

## 3. Architecture as built

### 3.1 Process layout (VPS, May 27 state)
```
[liq_paper screen]       v1.app.paper_runner    ~3,000 trades/day, ~1.1 GB/day telemetry
[liq_market_meta screen] v1.analysis.funding_oi_poller  ~2 MB/day
[bitget_capture screen]  v1.analysis.bitget_capture  ~0.5-1 GB/day
                          (bybit_capture killed May 26 after Phase 2.5)
[on demand]              v1.analysis.backfill_forward_returns
```

### 3.2 Gate stack (current regime-soft mode)
```
raw_event
  ↓ capture_writer.enqueue  ← runs ALWAYS, independent of pipeline
normalize → features
  ↓
decision_interval throttle (250 ms per symbol)
  ↓
regime_gate.evaluate()
    SKIPPED when REGIME_LIQ_OPTIONAL=1:  insufficient_liquidation_stress,
                                          liq_flow_misaligned, liq_in_dead_zone
    SKIPPED when REGIME_SESSION_OPTIONAL=1:  outside_us_session
    STILL CHECKS: stream_unhealthy, book_unhealthy, weak_flow_imbalance,
                  spread_too_wide, no_depth_collapse
  ↓ (if tradable)
signal_engine.evaluate()  ← v1/signal/continuation_score.py
    NO_TRADE if score < 0.42
  ↓ (if ENTER)
paper_calibration_gate.evaluate()  (off unless ENABLE_CALIBRATION_GATE=1)
  ↓
ml_quality_gate.evaluate()  (SKIPPED when DISABLE_ML_QUALITY_GATE=1)
  ↓
risk_engine.evaluate()
  ↓
order placed → executor → fill → position_manager → close
  ↓
close_order event includes tob_best_bid/ask/mid/spread_bps  (May 20 enrichment)
```

### 3.3 Data flow
```
Binance WS (4 symbols: BTC/ETH/SOL/HYPE)
  ├── !forceOrder@arr (liquidations)
  ├── aggTrade
  └── depth diff
       ↓
   capture_writer  → v1_data/raw/date=YYYY-MM-DD/hour=HH/symbol=*/stream=*/event_v1.jsonl.zst
       ↓
   normalize → features (flow imbalance, depth collapse score, liq intensity)
       ↓
   regime gate → signal scoring → ML gate → risk engine → executor
       ↓
   v1_data/telemetry/
     ├── decision_log.jsonl       (every decision, full mode)
     ├── feature_log.jsonl        (60s baseline + always during liq active)
     ├── order_lifecycle_log.jsonl  (order/fill/close events)
     ├── funding_oi_log.jsonl    (60s OI + 300s funding)
     └── risk_health_log.jsonl    (heartbeat)
```

### 3.4 Signal definition (what v1 "thinks" is tradable)
- **Liquidation cascade trigger:** liq_notional ≥ $300K within rolling window
- **Flow alignment:** aggressive trade imbalance same direction as liquidation
- **Depth collapse:** order book thin on the cascade side
- **Continuation score:** continuous score ∈ [-0.2, +1.1], threshold 0.42 → ENTER
- **Direction:** continuation (LONG when buy-side liqs cascade, SHORT when sell-side) — fade was tested in May 20 prototype and confirmed dead (-0.46 bps)

### 3.5 Exit logic (April 28 calibration, unchanged since)
- **Trailing stop:** activate at +2 bps unrealized, trail by 1 bps
- **Hard stop:** -8 bps
- **Flow reversal:** count of opposite-flow events ≥ 20 → exit
- **Time limit:** ~5 min (varies by symbol)
- **Hard TP:** +30 bps (rare, only 0.3% of exits)
- **Depth recovered:** book health returns → exit (3.3% of exits)

### 3.6 Storage budget
- Raw: 1.75 GB/day Binance + 0.5–1 GB/day Bitget = ~2.5 GB/day
- Telemetry (full mode): ~1.1 GB/day
- Combined ~3.6 GB/day = ~108 GB / 30 days
- VPS: 193 GB total, 89 GB used (47%) as of May 27

---

## 4. Validation phases (chronological)

### Phase 0 — Stability (ongoing)
- Daily watchdog (`liq-v1-daily-health`) checks: 3 screens alive, decision_log mtime <5 min, no `book_resync_failure_total > 0`.
- 0 bot incidents since the 2026-05-21 KILL_SWITCH_MAX_DRAWDOWN_PCT fix.

### Phase 1 — 24h checkpoint (May 21)
- Initial 100 min post-restart: n=122, WR=72.1%, gross=+0.90 bps. **Looked promising.**
- Same-day auto-24h: n=336, WR=67.3%, gross=+0.358 bps. Then a sticky `kill_switch:drawdown_breach` blocked 100% of ENTERs for 5h.
- Fix: added env-tunable `KILL_SWITCH_MAX_DRAWDOWN_PCT` (default 0.08, set to 10.0 in paper). Bot restarted; 81 closes in 16 min; 0 kill switch hits.
- **Lesson:** initial promising windows can be small-sample noise. Wait for n>1000 before believing.

### Phase 2 — 72h checkpoint (May 23, n=9,269)
- WR=62.6%, gross=-0.057 bps (≈0), fees=$721, net=-$728
- Entry has zero forward edge: 5s mean +0.02 bps
- Maker savings only 0.16 bps spread
- No funding/OI discrimination
- Score r=-0.005
- Hard_stop 21.9% at -9.50 bps is structural drag
- **Verdict:** Cross-venue track soft-locked (Bybit gap analysis prereq for hard commit)

### Phase 2.5 — Bybit basis analysis (May 26)
- 5 full UTC days of Bybit aggTrade collected (May 21–25, n_minutes ≈ 7,200 per symbol)
- BTC: μ=+1.36 bps, σ=4.55 bps → viable with basis-aware sizing
- ETH: μ=+0.20 bps, σ=4.27 bps → viable
- SOL: μ=+0.22 bps, σ=6.59 bps → NOT viable (σ > 5 bps threshold)
- Peak corr r=0.627 (BTC) / 0.589 (ETH) at lag=0±200ms. Binance leads Bybit sub-200ms. Mechanically synchronous.
- **But basis stdev 4–7 bps ≫ v1's ≈0 bps edge.** Cross-venue is mechanically viable but does not "fix" zero-edge signal.

### Phase 3 — 5-day soft lock (May 25, n=15,808)
- WR=62.9%, gross=-0.056 bps (95% CI [-0.154, +0.043]), fees=$1,232, net=-$1,242
- All recipes confirm: entry random, exits convert WR not gross, score gate binary only
- **Verdict:** Cross-venue track soft-locked, awaiting May 26 gap

### Phase 3' — 7-day HARD lock (May 27, n=21,816)
- WR=62.7%, gross=-0.061 bps (95% CI [-0.143, +0.022]), fees=$1,702, net=-$1,717
- Daily gross stable within ±0.16 bps across all 7 days
- Same exit mix, same per-symbol distribution, same Recipe 2–7 conclusions
- **Verdict:** Cross-venue HARD-LOCKED. Architecture redesign branch ALSO opened (this document).

---

## 5. What works (positive findings to preserve)

### 5.1 Signal direction (continuation, not fade)
Tested in May 20 prototype: fade returns -0.46 bps gross, continuation +0.46–0.66 bps gross. Continuation is the right direction. **Don't re-test this.**

### 5.2 Position management transforms WR
The Apr 28 exit recalibration moved trailing_stop from "11% WR fail" to "93% WR winner" and made flow_reversal extinct. Realized WR 62.7% vs entry forward-return WR 47.9%. The exit machinery works *as a WR converter*; it just doesn't add gross because hard_stop losses cancel trailing wins.

### 5.3 Data pipeline is rock-solid
21,816 trades over 7 days with 0 incidents. Telemetry joinable by `decision_id`. Forward-return backfill produces 130K+ enriched rows in 5 minutes. Bybit and Bitget captures work via the Binance-compatible RawMarketEvent schema — cross-venue analytics are uniform. **Keep all data plumbing.**

### 5.4 Decision logging is comprehensive
`TELEMETRY_LOG_LEVEL=full` writes 99.9% of decisions to disk. Recipe 7 confirms 882 ENTERs/h reach signal gate. Pipeline-up is monitorable in real time.

### 5.5 Cross-venue mechanically viable
Bybit basis tight enough for BTC/ETH execution. Latency synchronous. Bitget capture seeding for parallel confirmation. **This is real optionality.**

---

## 6. What doesn't work (dead ends, documented to avoid revisiting)

### 6.1 Entry signal has no forward-return edge
At every horizon (500 ms, 1s, 5s, 30s, 5 min), forward returns cluster around zero:
- 5s: WR=47.9%, mean=+0.018 bps
- 30s: WR=47.9%, mean=-0.129 bps
- 5 min: WR=48.9%, mean=-0.411 bps (mild reversion)

The signal is not predictive. It is a "trigger" that conditions on liquidation cascades and produces random direction over the holding window.

### 6.2 Maker mode is irrelevant
Mean spread savings: 0.141 bps (Recipe 4 across 21,946 closes). Even with full maker rebate, gross stays ≈ 0 bps, so net stays negative.

### 6.3 Score thresholding doesn't separate winners
Pearson r(score, gross_pnl_bps) = -0.006. The lowest-score bucket (0.42–0.45) has the best gross. Score is a binary gate, not a quality measure.

### 6.4 Funding/OI is not a useful filter
Tailwind vs headwind spread: <0.15 bps. Below useful threshold (≥2 bps).

### 6.5 Symbol filtering doesn't fix it
- HYPE 70% of volume, gross +0.03 bps
- SOL 11%, gross -1.11 bps (the only true negative outlier)
- BTC 10%, +0.23 bps
- ETH 9%, +0.21 bps

Dropping SOL would slightly improve gross but the equilibrium math (trailing wins ≈ hard_stop losses) holds across all symbols. SOL drop is worth doing in the redesign but won't generate edge.

### 6.6 Hard_stop loosening / tightening
- Current 8 bps: 21.1% hit rate at -9.42 bps avg
- Tightening to 5 bps: more stops, but smaller avg loss → fee count up, gross unchanged
- Loosening to 12 bps: fewer stops, bigger avg loss → fee count down, gross probably unchanged or worse

Symmetric balance is structural — can't tune out.

### 6.7 ML quality gate (current state)
The pre-May 18 ML calibration used liquidation features that became irrelevant under regime-soft. Retraining requires ≥500 ENTERs in new feature distribution (Phase 6, back-burner). Even if retrained, ML can only re-rank an already-zero-edge signal — won't generate gross.

### 6.8 Fee venue (Binance VIP-0)
~8 bps RT taker. At ≈0 bps gross, net = -8 bps × volume. This is the fee wall.

---

## 7. Root cause: why gross PnL is structurally zero

Decomposing the 7-day exit distribution:

| Exit | Share | Avg gross (bps) | Contribution (bps) |
|---|---|---|---|
| trailing_stop | 60.4% | +3.40 | +2.05 |
| hard_stop | 21.1% | -9.42 | -1.99 |
| time_limit | 14.7% | -0.99 | -0.15 |
| depth_recovered | 3.5% | -1.20 | -0.04 |
| hard_tp | 0.3% | +27.06 | +0.08 |
| **Total** | **100%** | | **-0.05 bps** ≈ zero |

The trailing_stop wins (+2.05) and hard_stop losses (-1.99) are nearly exactly equal. This is not noise; it has held within ±0.05 bps across every checkpoint (May 21, 23, 25, 27).

**Why is this the equilibrium?**

Because the entry has no forward edge (Recipe 2), each trade starts at expected zero gross. The exit mechanics then:
- Capture small wins via trailing stop when the trade drifts favorably (small +)
- Eat fixed-size losses via hard stop when the trade drifts unfavorably (fixed -)

For a zero-mean random walk over the holding window, the expectation of (asymmetric trailing exit at +K) × P(reach +K) + (hard stop at -L) × P(reach -L) ≈ 0 when the parameters K, L are calibrated to a "balanced" point. The April 28 recalibration found that balanced point.

To break the equilibrium without adding entry edge, you need one of:
- **Asymmetric stop math** that biases the integral positive (e.g. wider hard_stop + wider trailing_activate; risk: bigger drawdowns when wrong)
- **Reduce fee count** (batch signals into fewer larger positions)
- **Hold-to-event** exits (exit on next liq pulse, not time)
- **Discover a real entry filter** (none found from internal Binance data so far)

---

## 8. Redesign hypotheses (priority-ordered candidates)

All hypotheses can be tested *offline* against the 7d trade tape. Implementation budget: 1–3 weeks per hypothesis, paper-only validation, no live capital.

### H1 — Position aggregation (highest expected leverage)
**Idea:** Currently each liquidation signal becomes its own trade. 21,816 trades × 8 bps fees = $1,702 in fees alone. If we aggregate N consecutive signals into 1 net position (held until reversal or fixed window), fee count drops ~N×.

**Test:** simulate aggregating signals per symbol within rolling W-second windows (W ∈ [5, 30, 60, 300]s). Compare:
- N_trades reduction
- Realized gross (sum of per-event gross, weighted by remaining-position direction)
- Net-of-fees

**Expected outcome:** If N reduction is 5× and aggregate gross stays ≈ same per-position, net moves from -8 bps RT to -1.6 bps RT. Could be the single biggest lever.

**Risk:** aggregating opposite-direction signals cancels exposure mid-window; sizing must handle the netting cleanly.

### H2 — Asymmetric exit math
**Idea:** Current symmetric balance (trailing +3.4 / hard_stop -9.4) cancels exactly. Sweep wider trailing_activate (e.g. 4 bps instead of 2) and wider hard_stop (e.g. 15 bps instead of 8) to bias the integral.

**Test:** offline replay of all 21,816 trades' TOB/aggTrade history against new exit parameters. Use existing `v1/analysis/exit_ab_replay.py` framework (already built per Mar–Apr calibration sweeps).

**Expected outcome:** Probably -1 to +1 bps gross shift, possibly worse drawdown profile. Worth running but unlikely to be the silver bullet on its own.

### H3 — Event-driven exits (hold-to-next-liq)
**Idea:** Replace time/trailing exits with "exit on next liquidation pulse" (regardless of direction). Aligns hold time with the data-generating process instead of clock.

**Test:** offline replay using `decision_log` next-event timestamps. Compute realized gross at "next liq event" exit per trade.

**Expected outcome:** Unknown. Hold-time distribution will change dramatically. Could surface a positive edge if the post-cascade impulse decays before the next cascade.

### H4 — Symbol-portfolio redesign
**Idea:** Drop SOL (-1.11 bps drag), stress-test HYPE (70% of volume, near-zero gross), concentrate on BTC/ETH (small positive). Optional: introduce per-symbol score thresholds.

**Test:** subset the existing 21,816 trades by symbol mix, recompute net. Compare {BTC+ETH only} vs {+ HYPE} vs current. Trivial; can run in 5 min.

**Expected outcome:** Drop SOL → small gross improvement, fee proportional drop. {BTC+ETH only} fee count drops ~5× → net loss might drop 5×. Combined with H1 could be material.

### H5 — Multi-event signal trigger (require N events in window)
**Idea:** Instead of single-event triggers, require ≥N liquidation events within window M seconds. Filters out isolated liq-cascade triggers that are pure noise; retains true cascades.

**Test:** offline filter on decision_log — count liq events in W-sec window before each ENTER, drop ENTERs with count < N. Recompute net for ENTERs that survive.

**Expected outcome:** Sample drops dramatically (probably 3–10×). If surviving sample has clearly positive gross, this is the real signal. If still ~0, the cascade-count hypothesis is dead.

### H6 — Combine H1+H4 (the realistic first deploy)
**Idea:** Most likely first-deploy combo: aggregate signals per symbol within a 30s window, BTC+ETH only, current exit math.

**Test:** offline only. If gross ≥ +3 bps net of expected ~2 bps fees, deploy to paper as v1.1.

---

## 9. Open questions for the next iteration

1. What is the empirical distribution of inter-cascade times per symbol? Drives H3 and H5 design.
2. Does the 30s/300s mean-reversion in Recipe 2 (-0.41 bps at 5min) imply there IS a fade-the-cascade edge if hold is long enough? Counter to May 20 fade-is-dead finding — needs explicit re-test now that we have larger sample.
3. Is the basis (Binance vs Bitget vs Bybit) stable enough to use cross-venue as a *complement* to redesign rather than an alternative? (e.g. signal on Binance, hold on Bitget for cheaper fees, but still apply H1+H4 batching)
4. Are there features in `feature_log.jsonl` that we haven't used as entry filters yet (depth-slope, orderbook microstructure, post-cascade impulse decay rate)? The score model uses some but not all.
5. What does the cross-symbol correlation look like? If BTC liq predicts ETH return, cross-symbol signals could 2× the data without 2× the fee cost.

---

## 10. Data assets preserved

All data is kept on VPS. Preserved indefinitely until disk pressure forces rotation (currently 47% used).

| Asset | Path | Use |
|---|---|---|
| Binance raw | `v1_data/raw/date=*/hour=*/symbol=*/stream=*/event_v1.jsonl.zst` | Offline backtest replay, feature recomputation |
| Bybit raw (May 20–26) | `v1_data/raw/venue=bybit_um/` | Cross-venue basis analysis, completed |
| Bitget raw (May 26+) | `v1_data/raw/venue=bitget_um/` | Phase 5 cross-venue analysis, ongoing |
| decision_log | `v1_data/telemetry/decision_log.jsonl` | Every decision (full mode) |
| feature_log | `v1_data/telemetry/feature_log.jsonl` | All inputs to signal model |
| order_lifecycle_log | `v1_data/telemetry/order_lifecycle_log.jsonl` | Every order/fill/close with TOB |
| funding_oi_log | `v1_data/telemetry/funding_oi_log.jsonl` | 60s OI + 300s funding for 4 symbols |
| risk_health_log | `v1_data/telemetry/risk_health_log.jsonl` | Heartbeat |

---

## 11. Lessons for the dev/testing process

(Also propagated to `memory/v1_dev_testing_process.md`.)

### L9 — Daily metric stability beats sample size for decisive verdicts
At 9k trades (Phase 2) the verdict was already there: gross ≈ 0, daily ±0.16 bps. The 22k-trade Phase 3 confirmation moved nothing material. Whenever daily metrics are stable across ≥5 consecutive days within ±0.2 bps of the mean, more sample is wasted compute. Move to the next decision instead.

### L10 — Successful local optimization can hide global zero-edge
The April 28 recalibration was a structural success — flow_reversal extinct, trailing_stop dominant, WR 47% → 63%. Looked like winning. The mistake was conflating "this knob is now well-tuned" with "this strategy now makes money." Always re-test the headline (gross PnL net of fees) after every local optimization, no matter how clean the diagnostic looks.

### L11 — Multiple internal levers can all individually rule themselves out
Score thresholding, funding/OI, maker mode, symbol filter, hard_stop sweep — all five returned independent "not the answer" results within 2 weeks of regime-soft deploy. Plan parallel ruling-out, not sequential one-at-a-time tests. Recipe 1–7 in `v1_analytics_playbook.md` is exactly this pattern; codify it.

### L12 — Cross-venue is not a signal fix
Cheaper fees only help if there is gross edge to preserve. With gross ≈ 0, even free trading would still give net ≈ 0. Venue choice is downstream of signal; never pitch it as a strategy rescue.

### L13 — Fade vs continuation re-tests are cheap; re-do when sample 10×
May 20 fade prototype said -0.46 bps at small n. We have 22k trades now; fade should be re-tested explicitly before assuming continuation is forever the right direction. Cheap, decisive, worth a one-hour offline run.

### L14 — Symmetric exit equilibrium is a known failure mode
When trailing_wins × P(reach trailing) ≈ -hard_stop_losses × P(reach hard_stop), gross is structurally zero for any zero-mean random walk. Detect this by computing the contribution table (Section 7). When the two big bars balance to ±0.1 bps, the equilibrium is calibrated and entry must add edge for the strategy to work.

---

## 12. Where to start (concrete next-step plan)

(Detailed implementation plan in `memory/v1_redesign_kickoff_may27.md` and the IMPLEMENTATION_PLAN section of this doc.)

**Week of May 27 – June 2:**
1. Implement H1 + H4 offline simulator using existing trade tape (3–5 days)
2. Run H5 (multi-event filter) on decision_log (1 day)
3. June 2: `liq-v1-bitget-binance-gap` fires, finalizes cross-venue branch
4. Decide H1+H4 simulator results → if gross-of-fees ≥ +2 bps, paper-deploy v1.1

**Week of June 2–9:**
5. If v1.1 candidate exists: deploy paper-side-by-side with v1 baseline. Don't kill v1 — run both.
6. Bitget executor implementation begins (Phase 5) if June 2 gap confirms.
7. Run H2 (asymmetric exit sweep) offline.
8. Run H3 (event-driven exit) offline.

**Week of June 9–16:**
9. Compare paper v1.1 vs v1 baseline. If v1.1 gross net of fees > +1 bps, scale up.
10. Begin Phase 5 paper deploy on Bitget if executor ready.

**June 16+:**
11. Decision point: v1.1 winning → archive v1 baseline, focus on v1.1 + cross-venue.
12. v1.1 losing → re-evaluate; possibly retire v1 entirely, preserve data corpus for next-gen strategy.

---

*End of post-mortem. Companion files:*
- `memory/MEMORY.md` — index of all v1 memories
- `memory/v1_validation_roadmap.md` — current scheduled-task roadmap
- `memory/v1_analytics_playbook.md` — 7 reusable analysis recipes
- `memory/v1_redesign_kickoff_may27.md` — short ledger entry pointing here
- `v1/analysis/cross_venue_design.md` (on VPS) — Phase 5 design
