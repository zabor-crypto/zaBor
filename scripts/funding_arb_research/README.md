# funding-arb-research

> Multi-venue delta-neutral funding rate arbitrage research stack — async data collectors, strategy-agnostic backtest engine, 8 strategies tested. No edge found yet; the infrastructure is the artifact.

Collects funding rates from **Binance, Bybit, OKX, Hyperliquid, Bitget, and GMX v2**, normalizes them into a unified schema, and backtests delta-neutral carry strategies against a realistic cost model with proper funding settlement timing, margin simulation, and honest PASS/REJECT verdicts.

---

## Why this exists

Funding rate arbitrage is the most obvious carry trade in crypto — go long on the venue paying, go short on the venue receiving, collect the spread. The pitch falls apart when you run the numbers: taker fees (~17 bps round-trip across most CEX pairs) eat the median cross-venue spread, and the rare large spikes mean-revert before you can exit at maker rates.

This repo is the full research process: six async collectors, a no-look-ahead backtest engine, five carry strategies and three event-driven variants, and the verdict table showing all of them fail. Understanding *why* they fail — and where the fee/spread boundary actually sits — is the research output.

**Honest result:** every configuration tested returns `REJECT`. The highest-throughput strategy (cross-DEX dispersion at 0.2 APR threshold) generates 2,423 trades at -81% net APR. The most selective (new-listing spike at 2× APR, ≤30d age) achieves 45 trades at -1% net APR — closest to breakeven, not enough data for a verdict.

---

## Quickstart

```bash
git clone https://github.com/zabor-crypto/funding-arb-research.git
cd funding-arb-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Collect 90 days of funding history (CEX venues, no API key needed)
python -m funding_arb.main_collect \
    --venues binance bybit okx hyperliquid \
    --symbols BTC ETH SOL \
    --start 2025-01-01 --end 2025-04-01

# 2. Run a backtest strategy
python -m funding_arb.main_backtest --strategy hl_binance_interval
python -m funding_arb.main_backtest --strategy cross_cex_persistence
python -m funding_arb.main_backtest --strategy gmx_imbalance_feasibility

# Results appear in outputs/runs/<timestamp>_<strategy>/
```

No API keys are required for the public endpoints used in backtesting. The optional recorder (`src/recorder/`) for live data collection requires CEX keys — see `.env.example`.

---

## Backtest results

All results use `$1,000,000` starting capital, taker fees throughout, and the slippage model from `config/fees.yaml`. Runs cover Jan–Apr 2026 (90-day window) except cross-dispersion and new-listing which span Nov 2024–Apr 2026.

### Core carry strategies

| Strategy | Trades | Net APR | Max DD | Verdict |
|----------|-------:|--------:|-------:|---------|
| HL vs Binance interval arb | 78 | -11.1% | -2.8% | **REJECT** |
| Cross-CEX funding persistence | 0 | — | — | **REJECTED** (no signal) |
| Cross-CEX residual | 0 | — | — | (no trades) |
| GMX v2 imbalance carry | 0 candidates | — | — | **REJECT** |
| HL cross-venue dispersion | 91 | -10.6% | -2.7% | **REJECT** |

### Event-driven strategies (threshold grid)

**Bitget extreme funding spikes** (hedge on Binance/Bybit/OKX/HL):

| Entry threshold APR | Trades | Net APR | Win rate | Avg hold | Max DD | Verdict |
|--------------------:|-------:|--------:|---------:|---------:|-------:|---------|
| 0.50× | 199 | -24.5% | 12.6% | 22.8h | -7.3% | **REJECT** |
| 1.00× | 134 | -16.5% | 17.9% | 26.6h | -4.3% | **REJECT** |
| 2.00× | 98 | -10.5% | 17.3% | 25.6h | -3.3% | **REJECT** |

**Cross-DEX dispersion** (30 coins, 5 venues):

| Min carry APR | Trades | Net APR | Win rate | Max DD | Verdict |
|--------------:|-------:|--------:|---------:|-------:|---------|
| 0.20 | 2,423 | -80.9% | 1.9% | -119.7% | **REJECT** |
| 0.50 | 1,155 | -34.1% | 4.3% | -51.8% | **REJECT** |
| 1.00 | 547 | -12.7% | 10.2% | -21.7% | **REJECT** |
| 2.00 | 255 | -4.8% | 12.2% | -10.0% | **REJECT** |

**New-listing funding spike** (29 coins with known launch date):

| Threshold APR | Max age | Trades | Net APR | Win rate | Max DD | Verdict |
|--------------:|--------:|-------:|--------:|---------:|-------:|---------|
| 0.50 | 30d | 97 | -2.8% | 7.2% | -4.1% | **REJECT** |
| 1.00 | 30d | 68 | -2.2% | 8.8% | -2.9% | **REJECT** |
| 2.00 | 30d | 45 | -1.0% | 11.1% | -1.4% | **REJECT** |
| 1.00 | 7d | 27 | -0.8% | 7.4% | -1.1% | **NEEDS MORE DATA** |

**Primary failure mode:** fee drag. The HL interval strategy's per-symbol breakdown shows fees of $9–10k on SOL alone over 54 trades — dwarfing any funding capture.

---

## Architecture

```
main_collect.py
  └── async collectors (Binance/Bybit/OKX/HL/Bitget/GMX v2)
        └── normalization (venue-specific → canonical schema)
              └── data/normalized/*.parquet

main_backtest.py
  └── BacktestEngine (fixed-step clock loop)
        ├── funding settlement at actual settlement timestamps
        ├── cost model: fees + slippage (from depth or fallback)
        ├── margin proxy: free/used ratio with liq buffer
        ├── strategy(state) → Action (OpenLeg list | CloseAll)
        └── BacktestResult → outputs/runs/<ts>_<strategy>/
                              metrics.json, equity_curve, trades, charts
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full layer contracts, canonical schemas, engine invariants, and data sources.

---

## Strategy mechanics

### HL vs Binance interval (MVP-1)

Hyperliquid funds hourly; Binance funds every 8 hours. When HL predicted funding diverges from Binance's next-settlement rate by more than `entry_threshold_apr`, open a delta-neutral pair: long the negative leg, short the positive leg. Exit when spread converges below `exit_threshold_apr`. The position must survive at least one Binance settlement to realize any carry.

**Why it fails:** a ~6 bps maker round-trip (HL + Binance) is achievable, but the spread rarely persists through a full Binance 8h window. Most trades close before a settlement fires.

### Cross-CEX persistence (MVP-2)

Z-score the rolling funding spread between pairs of CEX venues (Binance/Bybit/OKX). Enter when `z > z_entry` and the spread's AR(1) coefficient exceeds `ar_min` (high persistence). Exit when `z < z_exit`.

**Why it fails:** cross-CEX funding on liquid pairs (BTC/ETH/SOL) is tightly arbitraged. The spread barely exists at 1h resolution; it doesn't survive the round-trip cost at taker rates.

### GMX v2 imbalance carry (MVP-3)

GMX v2 applies a directional funding factor when one side dominates open interest. If the factor exceeds the round-trip cost amortized over `carry_horizon_days`, enter a position on the subsidized side hedged on a CEX.

**Why it fails at the scan stage:** GMX borrowing factors are computed dynamically on-chain and difficult to backtest historically. The available imbalance proxies don't clear the minimum net-APR gate even before fees.

### Event-driven strategies

Three event triggers attempt to exploit temporary funding dislocations:
- **Bitget extreme:** enter when Bitget funding exceeds an APR threshold, hedge on cheapest available venue.
- **Cross-DEX dispersion:** trade the widest cross-venue spread across 30 coins on 5 venues.
- **New-listing spike:** enter when a newly-listed coin shows extreme funding in the days after launch.

All three fail for the same reason: the funding spike is the signal, but by the time a position is opened and the first settlement fires, the spike has mean-reverted. Win rates below 20% with negative expectancy.

---

## Cost model

Defined in `config/fees.yaml`. Default maker/taker bps per venue:

| Venue | Maker | Taker |
|-------|------:|------:|
| Binance futures | 2.0 | 4.0 |
| Bybit | 2.0 | 5.5 |
| OKX | 2.0 | 5.0 |
| Hyperliquid | 1.0 | 2.5 |
| Bitget | 2.0 | 6.0 |

Slippage formula (when depth is available):
```
slippage_bps = max(min_slippage_bps, k × √(order_notional / depth_notional))
```
Falls back to `slippage.fallback_bps` if no depth snapshot is loaded.

---

## Data sources

All public endpoints, no auth required for backtesting:

| Venue | Endpoint | Auth |
|-------|----------|------|
| Binance | `fapi.binance.com /fapi/v1/fundingRate` | none |
| Bybit | `api.bybit.com /v5/market/funding/history` | none |
| OKX | `www.okx.com /api/v5/public/funding-rate-history` | none |
| Hyperliquid | `api.hyperliquid.xyz /info` (POST) | none |
| GMX v2 | `arbitrum-api.gmxinfra.io/markets/info` | none |
| Bitget | `api.bitget.com /api/mix/v1/market/history-fundRate` | none |

The live recorder (`src/recorder/`) polls Binance and Bitget mark prices; those collectors require API keys.

---

## CLI reference

**Collect:**
```bash
python -m funding_arb.main_collect --venues binance bybit okx hyperliquid \
    --symbols BTC ETH SOL --start 2025-01-01 --end 2025-04-01

python -m funding_arb.main_collect --venues gmx_v2 --gmx-snapshot

python -m funding_arb.main_collect --venues binance --symbols BTC ETH SOL \
    --depth --depth-limit 100
```

**Backtest:**
```bash
python -m funding_arb.main_backtest --strategy hl_binance_interval
python -m funding_arb.main_backtest --strategy cross_cex_persistence
python -m funding_arb.main_backtest --strategy cross_cex_residual
python -m funding_arb.main_backtest --strategy gmx_imbalance_feasibility
python -m funding_arb.main_backtest --strategy hl_cross_venue_disp
```

**Event-driven verdict scripts:**
```bash
python scripts/run_bitget_extreme_verdict.py
python scripts/run_cross_dispersion_verdict.py
python scripts/run_new_listing_verdict.py
```

Each backtest run writes to `outputs/runs/<YYYYMMDD_HHMMSS>_<strategy>/` containing `metrics.json`, `equity_curve.{csv,parquet}`, `trades.parquet`, and `charts/*.png`.

---

## Configuration

All strategy parameters are in `config/strategy_params.yaml` — no values baked into code:

```yaml
global:
  capital_usd: 1_000_000
  per_trade_capital_pct: 0.10
  max_strategy_dd: 0.15
  liquidation_buffer_min: 0.30

hl_binance_interval:
  universe: ["BTC", "ETH", "SOL"]
  entry_threshold_apr: 0.30
  exit_threshold_apr: 0.10
  role: "maker"          # maker-only execution to hit ~6bp round-trip
```

See [config/strategy_params.yaml](config/strategy_params.yaml) for all parameters.

---

## What is NOT in this repo

- **Live execution** — no order placement, account management, or cancel/replace logic
- **Historical data** — `data/` parquet panels are excluded; collect your own with `main_collect.py`
- **Backtest run artifacts** — `outputs/` is gitignored; results described above are from local runs
- **Production-grade margin model** — the engine uses a free/used approximation adequate for research, not for live sizing
- **Portfolio-level risk** — no cross-strategy capital allocation, no correlation-aware sizing

---

## Known limitations

1. **Margin is approximate.** Worst-leg adverse-move proxy; not venue-specific isolated/cross margin.
2. **Predicted funding** uses realized history as fallback for most CEX strategies (predicted-rate collectors are a planned extension).
3. **GMX funding factors** are dynamically computed on-chain and not backfillable via public APIs; the feasibility module uses a carry proxy.
4. **Slippage** falls back to a flat bps estimate unless you collect orderbook depth first.

---

## Possible next directions

The negative results are informative. The narrowest loss is the new-listing strategy at 1× APR, 7-day window (-0.8% APR, 27 trades, `NEEDS MORE DATA`). Candidates for further research:
- Maker-only execution to cut round-trip from ~17 bps to ~6 bps
- Predicted funding as entry signal rather than realized (reduces adverse selection on entries)
- Longer hold target — waiting for a second settlement before exiting
- Pairs with wider structural spreads (smaller-cap perps on Bitget vs Binance)

---

## Disclaimer

Research simulator only. Results do not reflect live trading performance. The strategies have not been validated as profitable. Do not deploy capital without independent validation on live data.

---

## License

MIT — see [LICENSE](LICENSE).
