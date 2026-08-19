# zaBor

> Engineering-focused toolkit for market-data recording, execution safety and reproducible research.

[![CI](https://github.com/zabor-crypto/zaBor/actions/workflows/ci.yml/badge.svg)](https://github.com/zabor-crypto/zaBor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What zaBor is

A collection of self-contained modules for two problems that are easy to get wrong in automated
market systems: **operational safety** — controls that must fail closed rather than open — and
**reproducible research** — findings published together with the code and method that produced them.

Everything here installs and tests offline. No API key is required to clone, install, run the test
suite, or read any result.

It is **not** a trading strategy, and it makes no claim of profitability.

---

## Three flagship components

### 1. Kill-Switch — [`scripts/killswitch/`](scripts/killswitch/)

Emergency risk-control engine with two independent contours.

The **fast-crash stages** identify which side caused a sharp loss, rank positions by a composite risk
score, and close surgically — escalating only if the situation worsens. The **Regime Guard** adds a
portfolio-level contour for the failure the fast stages structurally cannot see: the slow bleed —
positions held underwater for days, or one coin steadily squeezed against the book.

- **Stages:** close top-N risk contributors → close dominant losing direction → full stop (manual reset)
- **Regime Guard:** catastrophe cap · peak-drawdown · correlated cluster · daily loss · log-only rollout
- **Exchanges:** Binance, Bybit, Bitget · Futures + Spot · SQLite state
- **153 tests**, offline, no credentials

### 2. Hyperliquid Microstructure Recorder + Toxicity Toolkit — [`scripts/hl_microstructure_recorder/`](scripts/hl_microstructure_recorder/)

An append-only recorder for **wallet-tagged** order flow, plus a pure-standard-library toolkit that
measures whether a market maker actually profits against the flow crossing its spread.

Hyperliquid publishes both counterparty addresses on every trade. That turns adverse selection from an
assumption into a per-counterparty measurement — the methodological point of the module.
No API keys, no strategy logic, no execution.

- **Records:** L2 book (20 levels) · BBO · trades with counterparty wallets → hourly-rotated gzip JSONL
- **Toxicity suite:** per-coin and per-wallet maker net edge · intraday persistence (ACF/half-life) ·
  depth-markout and λ(δ) fill-intensity for Avellaneda–Stoikov · train→test wallet-stability check

Findings from a single-day capture are reported in the
[module README](scripts/hl_microstructure_recorder/README.md), with sample size, period and method
stated alongside them. The raw capture is not published.

### 3. Funding Rate Arbitrage Research — [`scripts/funding_arb_research/`](scripts/funding_arb_research/)

Multi-venue delta-neutral funding-carry research stack: six async collectors (Binance, Bybit, OKX,
Hyperliquid, Bitget, GMX v2), a strategy-agnostic no-look-ahead backtest engine with an explicit cost
model, proper funding-settlement timing and margin simulation.

Dependencies are exact-pinned so parquet/CSV outputs are byte-identical across runs on the same
Python version. Pins move for published security advisories — the guarantee holds within a pin set,
not across pin sets.

> **Verdict: REJECT.** Five carry strategies and three event-driven variants all fail at current fee
> levels. The verdict tables and the reasoning ship with the code. Understanding *why* they fail — and
> where the fee/spread boundary sits — is the research output; the infrastructure is the artifact.

---

## Related repository

[**research-intelligence-platform**](https://github.com/zabor-crypto/research-intelligence-platform)
sits upstream of this one. It turns external research — papers, repositories, notes — into ranked,
backtest-ready strategy hypotheses, and documents the process layer that decides whether such a
hypothesis survives contact with data: pre-freeze market-identity gates, preregistration, execution
accounting, independent reconciliation, and terminal closure that cannot be undone.

Its published outcomes are two preregistered backtests, zero gross-positive, zero net-positive, both
terminally closed. The division of labour is deliberate: that repository decides **what is worth
building**, this one contains **what was built**.

---

## Safety model

This repository contains **live-trading-capable** components. The design rule is **fail closed**:

- `dry_run` defaults to **true**. A config that omits the key, or sets it to null or a non-boolean
  value, runs in dry-run and logs why. Live execution requires an explicit boolean `dry_run: false`.
- Enabling live mode prints a prominent warning before anything runs.
- There is no CLI flag that can enable live execution — it is config-only, by design.
- Stage 3 (full stop) requires a **manual** reset; the system does not resume trading on its own.
- Credentials come from environment variables only; nothing is read from committed config.
- Exchange keys should be **trade-only** (no withdrawal permission) and IP-whitelisted.

**Never run in live mode without a dry-run validation period of at least 24–48 hours.**

You remain responsible for API-key security, position sizing, exchange-specific rules (min size, lot
size, tick, leverage caps) and local regulatory compliance.

---

## Architecture overview

```
                 ┌──────────────────────────────────────────┐
   venue APIs ──▶│  collectors        REST / WebSocket       │
                 │  · async, retrying, rate-limit aware      │
                 └───────────────────┬──────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │  normalization     unified schema         │
                 │  · per-venue quirks isolated at the edge  │
                 └───────────────────┬──────────────────────┘
                          ┌──────────┴───────────┐
                          ▼                      ▼
        ┌─────────────────────────┐   ┌─────────────────────────┐
        │  research               │   │  operational controls   │
        │  · no-look-ahead engine │   │  · risk scoring         │
        │  · explicit cost model  │   │  · staged closure       │
        │  · PASS/REJECT verdicts │   │  · SQLite state machine │
        └─────────────────────────┘   └───────────┬─────────────┘
                                                  ▼
                                      ┌─────────────────────────┐
                                      │  exchange adapters      │
                                      │  typed, dry-run gated   │
                                      └─────────────────────────┘
```

Each module under `scripts/` is self-contained with its own dependencies — deliberately, so a research
module can pin exact versions for reproducibility while an operational module tracks security updates.

---

## Quickstart

Fully offline. No API key, no exchange account, no live connection.

```bash
git clone https://github.com/zabor-crypto/zaBor.git
cd zaBor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Verify the install — 153 tests, no network, no credentials:

```bash
pip install -r scripts/killswitch/requirements.txt
python -m pytest scripts/killswitch/tests -q
```

Run the offline demo — an end-to-end pass against mock exchange state, no orders, no keys:

```bash
python scripts/killswitch/killswitch.py --test-mock
```

Only after this offline path works should you configure credentials for any live-capable component.

---

## Tests and CI

| Module | Tests | Offline | In CI |
|---|---|---|---|
| [`scripts/killswitch/`](scripts/killswitch/) | 153 | ✅ | ✅ |
| [`scripts/liquidation_signal_research/v1/`](scripts/liquidation_signal_research/) | 20 modules, 55 tests (1 failing, documented) | ✅ | ✅ 54 of 55 |
| [`scripts/hl_microstructure_recorder/`](scripts/hl_microstructure_recorder/) | none | — | error-gate only |
| [`scripts/funding_arb_research/`](scripts/funding_arb_research/) | none | — | error-gate only |
| other `scripts/` modules | none | — | error-gate only |

CI runs five jobs on every push:

| Job | What it actually checks |
|---|---|
| `lint` | ruff + mypy on the shared root **only** (`exchanges/`, `tests/` — 2 source files under mypy) |
| `install` | clean venv, install via the documented command, import the public package |
| `tests` | Kill-Switch suite; fails if fewer than 130 tests are collected |
| `scripts-errors` | error-only ruff across **all** of `scripts/` (undefined names, unreachable bindings, syntax errors, invalid comparisons) |
| `liquidation-tests` | event-sourced pipeline suite — replay, reconciliation, fault injection, snapshot resync, rollout gates |

**What CI does not check.** Style and typing are not enforced across `scripts/`: those modules do not
pass the repository's full ruff profile, and enforcing it would be a large unrelated reformat. The
`scripts-errors` job is a correctness gate, not a style gate. Two of the three flagship modules ship
no test suite at all — the recorder is I/O against a live venue and the funding stack is a research
pipeline whose verdict tables are its output. Read the badge as "the two suites that exist pass, and
nothing in `scripts/` has a genuine runtime error", not as full coverage.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Complete project catalog

| Path | Description |
|------|-------------|
| [`scripts/killswitch/`](scripts/killswitch/) | **Flagship.** Emergency risk-control engine — PnL attribution, staged closure, portfolio-level Regime Guard. Binance/Bybit/Bitget. 153 tests |
| [`scripts/hl_microstructure_recorder/`](scripts/hl_microstructure_recorder/) | **Flagship.** Wallet-tagged microstructure recorder + per-counterparty adverse-selection toolkit |
| [`scripts/funding_arb_research/`](scripts/funding_arb_research/) | **Flagship.** Multi-venue funding-carry research stack — 6 collectors, no-look-ahead engine, REJECT verdict |
| [`scripts/alt_4h_scanner/`](scripts/alt_4h_scanner/) | Binance spot scanner — 4H accumulation breakout detection with a 9-filter stack, computes structured limit-order levels for pullback entries |
| [`scripts/alt_4h_reversal_scanner/`](scripts/alt_4h_reversal_scanner/) | Binance/Bitget perps — 4H LONG reversal detection, three independent detectors, filter stack, 1H MTF confirmation, regime-adaptive R-multiple TP ladders. Signal generation only |
| [`scripts/liquidation_signal_research/`](scripts/liquidation_signal_research/) | Binance liquidation cascade signal — event-sourced paper-trading pipeline, microstructure features. Research in progress |
| [`scripts/long_gate_orchestrator/`](scripts/long_gate_orchestrator/) | Causal regime-gating layer for swing-long strategies — shared regime panel, per-strategy thresholds, WFO/placebo validation battery |
| [`scripts/lighter_mm/`](scripts/lighter_mm/) | Market-making simulator for Lighter.xyz DEX — ATR-scaled spread, inventory skew, toxicity filter, walk-forward optimization |
| [`exchanges/bitget/`](exchanges/bitget/) | Typed Bitget REST client adapter — order lifecycle, credentials, retries |
| [`indicators/tradingview/`](indicators/tradingview/) | Pine Script — entry-signal overlay and a descriptive market-context dashboard |
| [`configs/`](configs/) | YAML configuration templates |

---

## Configuration

All sensitive values come from environment variables — never hardcoded, never committed.

```bash
export BINANCE_API_KEY="..."      export BINANCE_API_SECRET="..."
export BYBIT_API_KEY="..."        export BYBIT_API_SECRET="..."
export BITGET_API_KEY="..."       export BITGET_API_SECRET="..."   export BITGET_PASSWORD="..."
```

See [`.env.example`](.env.example) for the full list. Requires Python 3.10+.

---

## Limitations and scope

- **No strategy and no profitability claim.** Signal-generation and research modules are published
  without the exit-management and sizing layers that would be required to trade them.
- **Research findings are not all reproducible from this repository.** Where a result depends on a
  private dataset or a private capture, the module documentation says so and states the period,
  sample size and method. Nothing here should be read as a forward-looking claim.
- **Modules are independent and unevenly mature.** The three flagship components are the most
  developed; others are research artifacts at varying stages, labelled accordingly.
- **Live-capable code is not proven in production.** It is tested offline and designed to fail closed;
  that is not the same as an operational track record.
- **Exchange behaviour under stress** — partial fills, rate limits, API degradation exactly when it
  matters — is handled defensively but cannot be fully tested against live venues.

### What is intentionally not in this repository

Private account-level performance, proprietary parameters and live trading records are not included.
Reproducible research findings based on public or included sample data may be reported in module
documentation.

Also excluded: live strategy logic and signal parameters; position history, trade logs and runtime
state; private configuration files (`.env`, `config.yaml` with real keys).

---

## License and contributing

MIT — see [LICENSE](LICENSE). TradingView Pine Script components carry their own license; see the
indicator subdirectory.

Contributions and issue reports are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md). Please do not open public issues for security matters.

**Disclaimer.** For research and educational purposes. Automated trading carries substantial risk of
financial loss. Nothing here is investment advice. Test thoroughly in dry-run mode before deploying
capital.
