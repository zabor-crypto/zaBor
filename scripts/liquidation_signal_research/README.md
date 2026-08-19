# liquidation-signal-research

**Binance liquidation cascade signal — event-sourced paper-trading pipeline and research results.**

This module documents a complete research cycle: hypothesis, an event-sourced implementation with
deterministic replay, paper-trading validation over 22,000+ simulated trades, and honest conclusions.

**Status:** v1 paper validation complete. Research continues on cross-venue execution and architecture redesign.

> **Reproducibility note.** The figures below come from a private paper-trading run (7-day window,
> May 2026) against live Binance market data. **No orders were placed** — the pipeline was in paper
> mode throughout. The raw event log and trade records are not published, so these numbers are not
> reproducible from this repository alone. What is published is the pipeline, the features, the
> replay machinery and the full post-mortem reasoning. Dataset, period and method are stated with
> every figure; treat them as a documented research record, not a verifiable benchmark.

---

## What this is

A microstructure-based trading system that monitors Binance futures liquidation events in real time and trades the continuation:

- Liquidation cascade detected → flow imbalance + depth collapse confirmed → enter in cascade direction
- Exit via trailing stop / hard stop / time limit

The system is built as a proper research pipeline: shadow mode → paper mode → (future) live, with deterministic replay to verify that paper decisions exactly match what a live system would have done.

---

## What we found

After 22,000+ paper trades (7-day window, May 2026):

| Metric | Value |
|--------|-------|
| Win rate | 62.7% |
| Gross PnL | −0.061 bps (95% CI [−0.143, +0.022]) |
| Net PnL | −$1,717 |
| Fees | $1,702 |
| Conclusion | **No measurable edge at Binance fee levels** |

The position manager (trailing stop) lifts WR from ~47% (random) to 63%, but gains from trailing stops are exactly cancelled by hard stop losses. The 8 bps Binance fee wall absorbs everything.

All internal levers tested: score threshold tuning, funding/OI conditioning, maker mode, per-symbol filtering, hard stop sweep — each individually ruled out.

**Open tracks:** cross-venue execution (lower-fee venue) and architecture redesign (reduce trade count via event aggregation, asymmetric exits, event-driven holds).

Full post-mortem with all phases, hypotheses, and data: [docs/RESEARCH_RESULTS.md](docs/RESEARCH_RESULTS.md)

---

## Architecture

```
binance_ws_client  →  raw_capture_writer  →  jsonl.zst partitions
                                                    ↓
                                              normalizer
                                                    ↓
                                           local_book  +  feature_engine
                                                    ↓
                                            regime_gate
                                                    ↓
                                       continuation_score_engine
                                                    ↓
                                            risk_engine
                                                    ↓
                                         binance_executor
                                                    ↓
                                    telemetry / rollout_gates / replay
```

### Replay invariant

Every raw event is written to immutable partitioned `.jsonl.zst` files before processing. The `replayer` can feed the same raw stream back through the same decision path and produce bit-identical output. The `parity_runner` compares live decision logs against replay output and flags any divergence.

### Rollout gates

Paper mode tracks:
- Order-lineage completeness ratio (`feature → decision → order_intent`)
- Realized slippage diagnostics (mean, p95)
- Reject ratio and residual-flatten events

All gates must pass before promoting to live.

---

## Quickstart

```bash
git clone https://github.com/zabor-crypto/zaBor.git
cd zaBor/scripts/liquidation_signal_research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # add your Binance read-only API key
```

### Shadow mode (no orders, live market data)

```bash
PYTHONPATH=. python3 -m v1.app.shadow_runner \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --duration-seconds 3600 \
  --data-root v1_data/raw \
  --telemetry-root v1_data/telemetry \
  --report-path v1_data/telemetry/shadow_report.json
```

### Paper mode (order lifecycle simulation, no real fills)

```bash
PYTHONPATH=. python3 -m v1.app.paper_runner \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --duration-seconds 3600 \
  --data-root v1_data/raw \
  --telemetry-root v1_data/telemetry \
  --report-path v1_data/telemetry/paper_report.json \
  --min-decisions 10 --min-fills 1 \
  --min-lineage-ratio 0.95 \
  --max-mean-realized-slippage-bps 2.0 \
  --require-gates-pass
```

### Run tests

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q v1/tests
```

20 test modules, 55 tests. The reporting/analysis layer (`v1/analysis/`) is not part of this public
release, so its tests are not shipped either.

One test currently fails on a clean clone —
`test_shadow_mode_engine.py::test_shadow_mode_blocks_execution_even_when_trade_is_approved`: the
stubbed aggTrade event is dropped before a decision is produced, so the shadow-mode assertion never
runs. It is recorded here rather than deleted or skipped, because this module is active research and
the failure is a real signal about event handling, not a test-harness artifact.

---

## Module map

| Module | What it does |
|--------|-------------|
| `v1/ingest/` | Binance WebSocket client (`forceOrder`, `aggTrade`, `depth`, `markPrice`), raw capture writer, stream supervisor |
| `v1/normalize/` | Converts raw payloads to canonical event contracts with validators |
| `v1/contracts/` | Immutable event type definitions and deterministic ID generation |
| `v1/book/` | Local order book with desync detection and snapshot resync |
| `v1/features/` | Microstructure feature engine: flow imbalance, depth collapse, liq direction, spread, impact |
| `v1/regime/` | Regime gate: blocks trading outside defined market conditions |
| `v1/signal/` | Continuation score engine: weighted feature combination → ENTER/NO_TRADE decision |
| `v1/risk/` | Pre-trade risk engine, calibration gate, ML quality gate, kill switch |
| `v1/execution/` | Binance aggressive-limit executor, order state machine, position manager, slippage guard |
| `v1/telemetry/` | Structured logs, metrics, rollout gates, attribution, shadow monitors, alerts |
| `v1/store/` | Append-only partitioned JSONL store with manifest |
| `v1/replay/` | Raw-event replayer and live/replay parity checker |
| `v1/app/` | Shadow runner, paper runner, runtime configuration, parameter governance |

---

## Signal

The continuation score is a weighted combination of microstructure features computed over a 6-second rolling window:

```
score = 0.20 × flow_imbalance_1s
      + 0.10 × flow_imbalance_3s
      + 0.15 × liq_flow_alignment
      + 0.15 × depth_collapse_ratio
      + 0.15 × price_impact
      + 0.10 × failed_recovery
      + 0.05 × flow_acceleration
      + 0.10 × liq_notional_normalized
      − 0.15 × spread_penalty
      − 0.05 × exhaustion_penalty
```

Entry requires `0.42 ≤ score ≤ 0.55`. Scores above the ceiling historically underperform (calibration replay: 0.55–0.65 → −1.57 bps, 0.65+ → −4.04 bps).

**Finding:** score is a binary gate only. Within the 0.42–0.55 band, score has zero correlation with forward returns (r = −0.006 across 21,816 trades).

---

## What is NOT in this repository

- Raw market data (`v1_data/`)
- Live telemetry and trade logs
- Analysis scripts (one-off calibration experiments)
- VPS deployment configuration
- Ongoing backtest iterations (liq_v10, liq_v11) — active research

---

## Requirements

- Python 3.11+
- Binance Futures API key (read-only sufficient for shadow/paper mode)
- See `requirements.txt`

---

## Disclaimer

This is a research system. Paper trading results do not guarantee live performance. Automated trading carries substantial risk of loss. Nothing in this repository constitutes financial advice.

---

## License

MIT — see [LICENSE](LICENSE).
