# Binance Event-Sourced Continuation v1

New architecture path for Binance-only continuation-under-liquidity-vacuum stress.

## Scope Locks
- Symbols: `BTCUSDT, ETHUSDT, SOLUSDT, HYPEUSDT`
- Streams: `forceOrder, aggTrade, depth, markPrice`
- Raw source of truth: immutable partitioned `jsonl.zst`
- Replay invariant: deterministic ordering on `recv_mono_ts_ns, ingest_seq, stream_local_seq`
- Strategy branch: continuation only (no production fade branch)
- Stream health policy: `aggTrade/depth/markPrice` are freshness-gated; `forceOrder` is treated as sparse context (no per-symbol no-data stale gate)

## Runtime Chain
1. `ingest/binance_ws_client.py` captures raw websocket events.
2. `store/jsonl_partition_store.py` writes append-only raw partitions + manifest.
3. `normalize/normalizer.py` converts raw payloads to canonical event contracts.
4. `book/local_book.py` maintains local depth state with desync guards.
5. `features/microstructure_features.py` computes second-level flow/depth/spread/impact/recovery features.
6. `regime/regime_gate.py` enforces tradable/no-trade states.
7. `signal/continuation_score.py` produces continuation intents.
8. `risk/risk_engine.py` applies hard pre-trade/session gates.
9. `execution/binance_executor.py` creates aggressive-limit intents, reconciles user-stream order updates, and enforces ACK/TTL timeout handling.
10. `telemetry/*` records feature/decision/risk/order/replay diagnostics.
11. `replay/replayer.py` replays raw partitions through the same decision path.
12. `replay/parity_runner.py` compares live decision logs against deterministic replay output.

## Test Command
```bash
PYTHONPATH=. pytest -q v1/tests
```

## Shadow Mode Command
```bash
PYTHONPATH=. python3 -m v1.app.shadow_runner \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --duration-seconds 3600 \
  --heartbeat-seconds 30 \
  --checkpoint-every-heartbeats 10 \
  --data-root v1_data/raw \
  --telemetry-root v1_data/telemetry \
  --report-path v1_data/telemetry/shadow_session_report.json
```

This runs live ingest/feature/decision/risk with execution blocked (`shadow_mode=true`) and writes a summary report containing:
- decision action distribution
- no-trade reason distribution
- risk reason distribution
- score bucket distribution
- desync/resync counts

## Paper Mode Command (Rollout Gates + Parity)
```bash
PYTHONPATH=. python3 -m v1.app.paper_runner \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --duration-seconds 3600 \
  --heartbeat-seconds 30 \
  --data-root v1_data/raw \
  --telemetry-root v1_data/telemetry \
  --report-path v1_data/telemetry/paper_session_report.json \
  --min-decisions 10 \
  --min-fills 1 \
  --min-lineage-ratio 0.95 \
  --max-mean-realized-slippage-bps 2.0 \
  --max-p95-realized-slippage-bps 4.0 \
  --max-reject-ratio 0.15 \
  --max-residual-flatten-events 0 \
  --require-gates-pass
```

This runs paper execution (`shadow_mode=false`, `execution_paper_mode=true`) and writes:
- rollout gate pass/fail and failed gate list
- order-lineage completeness ratio from `feature -> decision -> order_intent`
- realized slippage diagnostics (`mean`, `p95`) from order lifecycle logs
- reject ratio and residual-flatten event counts
- replay parity summary for the same session window

Optional paper-only calibration gate:
```bash
PYTHONPATH=. python3 -m v1.app.paper_runner \
  ... \
  --enable-calibration-gate \
  --calibration-overrides-path analysis/phase3_experiments/<run-id>/per_asset_calibration_overrides.json
```

When enabled, entries failing per-asset overrides are blocked with risk reasons prefixed by
`paper_calibration_*` (paper mode only; shadow/live remain unchanged).

## Phase 3 Experiment Runner (Per-Asset Calibration)
```bash
PYTHONPATH=. python3 v1/analysis/phase3_experiment_runner.py \
  --telemetry-root v1_data/telemetry \
  --out-dir analysis/phase3_experiments \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT \
  --rolling-hours 48 \
  --friction-bps 8.0
```

Outputs:
- `analysis/phase3_experiments/<run-id>/phase3_experiment_report.json`
- `analysis/phase3_experiments/<run-id>/per_asset_calibration_overrides.json`

One-shot wrapper:
```bash
./v1/analysis/run_phase3_experiments.sh "$(pwd)" 48
```

## Phase 4A Monitoring Pack (Post-Restart)
Generate restart->now monitoring reports with calibration-block attribution:
```bash
./v1/analysis/run_post_restart_monitoring_pack.sh "$(pwd)" <restart-ts-ms>
```

Outputs:
- `analysis/monitoring_packs/<label>/reports/post_restart_window.json`
- `analysis/monitoring_packs/<label>/reports/context_rolling_24h.json`
- `analysis/monitoring_packs/<label>/summary.json`

`daily_report_generator.py` now emits `schema_version=daily_report_v2` and includes
`calibration_gate` attribution from both `risk_health_log` events and decision `risk_reasons`.
It also emits `calibration_config_drift` (runtime override path/hash vs latest recommended
override path/hash) so config drift is visible in every report.

Automate every 4 hours on VPS:
```bash
./v1/analysis/install_phase4a_cron.sh /root/trading-bot/Liquidations_websocker
```

Run one auto pack immediately:
```bash
./v1/analysis/auto_phase4a_pack.sh /root/trading-bot/Liquidations_websocker
```

Notes:
- `auto_phase4a_pack.sh` uses a lock (`logs/.auto_phase4a_pack_lock`) to prevent overlap.
- Analysis scripts auto-prefer `<project>/.venv/bin/python` when present.

## Phase 4B Exit Parameter Sweep
```bash
./v1/analysis/run_exit_parameter_sweep.sh "$(pwd)" 72
```

Outputs:
- `analysis/exit_sweeps/<label>.json`
- `analysis/exit_sweeps/<label>.stdout.log`

Notes:
- Runner is lock-protected (`analysis/exit_sweeps/.exit_sweep_lock`) to avoid concurrent sweeps.
- Wrapper output includes `elapsed_seconds` and `python_bin` for diagnostics.
- Sweep wrapper now includes stale-lock cleanup + runtime watchdog:
  - `SWEEP_STALE_LOCK_SECONDS` (default `43200`)
  - `SWEEP_MAX_RUNTIME_SECONDS` (default `21600`)
  - watchdog events: `analysis/exit_sweeps/watchdog_events.jsonl`
  - status file: `analysis/exit_sweeps/<label>.status.json`
- Heavy analysis wrappers run CPU/IO isolated by default:
  - `ANALYSIS_NICE_LEVEL` (default `15`)
  - `ANALYSIS_IONICE_CLASS` (default `3`, idle)
  - `ANALYSIS_IONICE_CLASSDATA` (default `7`)
- `exit_calibration.py` now has:
  - window-aware depth seed lookup
  - depth seed UID cache (`v1_data/cache/depth_uid_cache.json`)
  - per-hour replay progress + ETA logs

## Phase 4C Exit A/B Replay (Baseline vs Candidate)
Replay entry windows and compare candidate static exit profile against the current
live price-profile approximation (hard stop + dynamic TP + trailing + time limit),
with friction applied.

Using candidate from sweep output:
```bash
./v1/analysis/run_exit_ab_replay.sh "$(pwd)" \
  analysis/exit_sweeps/<sweep-label>.json \
  ab_$(date -u +%Y%m%d_%H%M%S)
```

Using manual candidate parameters:
```bash
CANDIDATE_TP_BPS=18 \
CANDIDATE_SL_BPS=10 \
CANDIDATE_TIME_LIMIT_MS=90000 \
./v1/analysis/run_exit_ab_replay.sh "$(pwd)"
```

Outputs:
- `analysis/ab_replay/<label>.json`
- `analysis/ab_replay/<label>.stdout.log`

Notes:
- Default windows: `72h,168h` (override with `WINDOWS_HOURS`, e.g. `WINDOWS_HOURS=24,72`).
- Default friction: `8.0 bps` (override with `FRICTION_BPS`).
- Baseline replay excludes non-price exits (`flow_reversal`, `depth_recovered`) because
  those decision-path signals are not persisted in calibration replay entries.

## Step 2 Batch A/B Matrix (Top Candidates + Promotion Gates)
Quick + full windows in one run:
```bash
./v1/analysis/run_step2_ab_matrix.sh "$(pwd)" \
  analysis/exit_sweeps/<sweep-label>.json \
  step2_ab_$(date -u +%Y%m%d_%H%M%S)
```

Targeted loss-cluster experiments (manual candidate set + sweep top candidates):
```bash
./v1/analysis/run_exit_targeted_experiments.sh "$(pwd)" \
  analysis/exit_sweeps/<sweep-label>.json
```

Batch output includes per-symbol promotion gates:
- non-degrading symbol net expectancy (for symbols with enough trades)
- drawdown proxy limit vs baseline
- trade retention floor
- absolute candidate net-mean floor

## Baseline Freeze (R2 Anchor)
Freeze current rolling reports + latest monitoring pack:
```bash
./v1/analysis/freeze_baseline_anchor.sh "$(pwd)" baseline_R2_$(date -u +%Y%m%d_%H%M%S)
```

Output:
- `analysis/baselines/<baseline-id>/baseline_manifest.json`

## Entry False-Positive Audit (48h + 7d)
```bash
PYTHONPATH=. python3 v1/analysis/entry_false_positive_audit.py \
  --telemetry-root v1_data/telemetry \
  --windows-hours 48,168 \
  --out analysis/entry_audits/entry_false_positive_audit_latest.json
```

## Calibration Gate A/B Harness (Offline)
Compare `no_gate` vs runtime current overrides vs latest recommended overrides:
```bash
./v1/analysis/run_calibration_gate_ab.sh "$(pwd)" \
  <runtime-overrides.json> \
  <recommended-overrides.json>
```

## Retention Automation (Daily)
Run once:
```bash
./v1/analysis/run_retention_cleanup.sh "$(pwd)"
```

Install daily cron (03:17 UTC):
```bash
./v1/analysis/install_retention_cron.sh /root/trading-bot/Liquidations_websocker
```

Defaults (configurable via env):
- `RAW_RETENTION_DAYS=7`
- `TELEMETRY_REPORT_RETENTION_DAYS=14`
- `ANALYSIS_PACK_RETENTION_DAYS=30`

Retention report output:
- `analysis/retention_reports/retention_<timestamp>.json`

For VPS deployment, adapt `v1/app/paper_runner.py` or `v1/app/shadow_runner.py` to your environment — see the quickstart in the root README.
