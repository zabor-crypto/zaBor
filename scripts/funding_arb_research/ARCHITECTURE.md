# funding_arb — Architecture

This document is the load-bearing reference for the stack. It describes
*what each layer is allowed to do*, the shape of data flowing between
layers, and the operational invariants the engine relies on.

## Goal

A research-grade backtester for delta-neutral crypto funding arbitrage.
Live execution is explicitly out of scope. The whole stack is built around
one principle: **a strategy result you can trust** — no look-ahead, real
costs, deterministic outputs, honest verdicts.

## Layers

```
┌───────────────────────────────────────────────────────────────────────┐
│  main_collect.py                                                       │
│      ├── async run-loop                                                │
│      └── per-venue collectors  ─────────▶ raw API/RPC                 │
│              │                                                         │
│              ▼                                                         │
│         normalization (per-venue → canonical schema)                   │
│              │                                                         │
│              ▼                                                         │
│         data/normalized/*.parquet                                      │
└───────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────┐
│  main_backtest.py                                                      │
│      └── _build_panels  (load funding+predicted+price+depth)           │
│                                                                       │
│           BacktestEngine                                              │
│           ├── clock loop (fixed step)                                 │
│           ├── per-tick mark-to-market                                 │
│           ├── funding settlement at actual settlement timestamps       │
│           ├── forced-exit logic (max DD, liq buffer, data stale)      │
│           ├── strategy(state) ─▶ Action  (OpenLeg list | CloseAll)    │
│           ├── entry/exit cost model (fees + slippage)                 │
│           └── BacktestResult (equity curve, trades, schema_issues)    │
│                                                                       │
│           ReportBuilder ─▶ outputs/runs/<ts>_<strategy>/              │
└───────────────────────────────────────────────────────────────────────┘
```

### Layer contracts

| Layer            | Reads                | Writes                 | Forbidden       |
|------------------|----------------------|------------------------|-----------------|
| Collectors       | venue API/RPC        | raw payloads + issues  | pnl, strategy   |
| Normalization    | raw payloads         | canonical-schema rows  | network IO      |
| Engine           | normalized panels    | trades, equity, marks  | venue specifics |
| Strategy         | `StrategyState`      | `Action`               | network IO, parquet, pnl |
| Cost model       | leg + depth          | bps                    | strategy state  |
| Reports          | `BacktestResult`     | `outputs/runs/...`     | engine internals|

If you find yourself reaching across layers, stop — there's a missing
interface.

## Canonical schemas

### Funding panel (`src/normalization/schema.py:FUNDING_COLUMNS`)

```
timestamp_utc          datetime64[ns, UTC]
venue                  string             ("binance"|"bybit"|"okx"|"hyperliquid"|"gmx_v2")
symbol                 string             venue-specific (BTCUSDT / BTC-USDT-SWAP / BTC)
base_asset, quote_asset string
instrument_type        string             ("perp")
funding_rate           float64            realized; NaN for predicted-only rows
funding_interval_hours Int16              1 for HL, 8 for CEXes
predicted_funding_rate float64            in-progress estimate
mark_price, index_price float64
open_interest, long_oi, short_oi float64
borrow_rate, collateral_yield  float64
source                 string             "<venue>/<endpoint>"
raw_payload_hash       string             SHA1 of the raw row
```

### Price panel — same idea, with bid/ask/mid_price.

### Depth panel (per-tick `depth_notional_usd`)

```
timestamp_utc, venue, symbol,
best_bid, best_ask, mid_price,
bid_notional_usd, ask_notional_usd,    # within band_bps of inside price
n_bid_levels, n_ask_levels,
raw_payload_hash
```

The engine reads the latest row `≤ clock` and uses
`min(bid_notional, ask_notional)` as the binding constraint for a
delta-neutral leg.

## Engine invariants

1. **No look-ahead.** `state.funding[(v,s)]` and `state.prices[(v,s)]`
   passed to the strategy are sliced strictly `< clock`. Predicted-funding
   rows whose `timestamp_utc` is in the future are filtered out by
   `_data_window` and only become visible after that timestamp passes.

2. **Funding pnl is booked at actual settlement timestamps.** The engine
   walks each leg's funding panel within `(prev_clock, clock]` and applies
   the realized rate exactly once per settlement. PnL convention:
   `pnl = -direction × rate × notional` (long pays, short receives when
   rate > 0).

3. **Fees + slippage charged at entry and exit, not at every tick.**
   Slippage uses depth: `bps = max(min, k × √(notional/depth)) × 1e4`.
   Falls back to `slippage.fallback_bps` if depth is missing.

4. **Forced exits are unconditional.** A position is closed regardless of
   strategy preference if any of: `drawdown ≥ max_strategy_dd`,
   `free_margin_ratio < liquidation_buffer_min`, or any open leg's price
   panel is `> data_staleness_seconds` stale.

5. **Margin model is approximate.** `free / used` margin where
   `used = sum(notional / leverage)`. Worst-leg adverse-move
   approximation; accurate enough for research, not for live sizing.

6. **Determinism.** Given identical inputs, the output is bit-identical.
   `requirements.txt` pins exact pandas/numpy/pyarrow versions for byte-
   exact parquet diffs.

## Data sources of record

| Data                    | Source                                                  | Auth |
|-------------------------|---------------------------------------------------------|------|
| Binance funding history | `fapi.binance.com /fapi/v1/fundingRate`                  | none |
| Binance premium index   | `fapi.binance.com /fapi/v1/premiumIndex`                 | none |
| Binance L2 depth        | `fapi.binance.com /fapi/v1/depth`                        | none |
| Bybit funding history   | `api.bybit.com /v5/market/funding/history`               | none |
| Bybit L2 depth          | `api.bybit.com /v5/market/orderbook`                     | none |
| OKX funding history     | `www.okx.com /api/v5/public/funding-rate-history`        | none |
| OKX L2 depth            | `www.okx.com /api/v5/market/books`                       | none |
| Hyperliquid funding     | `api.hyperliquid.xyz /info {type:fundingHistory}`        | none |
| HL predicted funding    | `api.hyperliquid.xyz /info {type:predictedFundings}`     | none |
| HL L2 book              | `api.hyperliquid.xyz /info {type:l2Book}`                | none |
| GMX v2 funding factors  | Arbitrum `DataStore.getInt(SAVED_FUNDING_FACTOR_PER_SECOND, market)` | RPC |
| GMX v2 markets          | Arbitrum `Reader.getMarkets(DataStore, 0, N)`            | RPC |
| GMX v2 OI / liquidity   | `arbitrum-api.gmxinfra.io/markets/info`                  | none |

The legacy Satsuma subgraph (`subgraph.satsuma-prod.com/.../synthetics-arbitrum-stats/api`)
is **dead**. Anything that needed the subgraph now uses `arbitrum-api.gmxinfra.io`.

## Configuration files

- `config/venues.yaml` — REST bases, endpoint paths, funding interval per venue.
- `config/strategy_params.yaml` — global capital, per-strategy thresholds & universes.
- `config/fees.yaml` — maker/taker bps per venue, slippage formula constants.

All three are read at runtime; no values are baked into code.

## Outputs (per-run dir)

```
outputs/runs/<YYYYMMDD_HHMMSS>_<strategy>/
  config_snapshot.yaml      # exact config used for this run
  metrics.json              # aggregate performance metrics
  equity_curve.{csv,parquet}
  trades.{csv,parquet}
  charts/
    equity.png
    drawdown.png
    funding_spread.png      # if applicable
    fee_drag.png
  feasibility_table.csv     # MVP-3 only
  rejection_report.md       # if verdict ∈ {REJECTED, REJECT}
  run.log
```

## Strategy verdict semantics

| Verdict       | Meaning                                                          |
|---------------|------------------------------------------------------------------|
| ACCEPTED      | Net APR ≥ threshold, MaxDD ≤ ceiling, ≥ N trades.                |
| REJECTED      | Survived a fair test and failed honestly. Do *not* parameter-sweep. |
| NEEDS MORE DATA | Not enough trades or data completeness for a verdict.          |
| PASS / REJECT (MVP-3) | Feasibility-only. PASS = ≥ 1 candidate market clears all gates. |

## Known limitations (not bugs — explicit scope)

1. Margin is approximate — see invariant (5).
2. GMX borrowing factor is read but tends to come back 0 (computed
   dynamically on-chain). Carry estimate uses funding + best-effort
   borrow.
3. Order-book depth is point-in-time at collection. Strategy slippage uses
   the most recent depth row `≤ clock`.
4. Live execution is not implemented and is out of scope for this stack.

## Bug-class checklist (from prior incidents)

When extending the stack, audit against these:

- [ ] No `pd.Timestamp.utcnow().tz_localize("UTC")` — pandas 2.2 returns
      tz-aware. Use `pd.Timestamp.now(tz="UTC")`.
- [ ] No hardcoded venue/contract addresses. Resolve from the canonical
      registry at runtime, with env-var override.
- [ ] No raw `keccak256(text)` for GMX-style storage labels — must be
      `keccak256(eth_abi.encode(["string"], [label]))`.
- [ ] No `Optional[X | Y]` runtime type aliases on Python 3.9 — use
      `Optional[Union[X, Y]]`.
- [ ] All venue/symbol canonicalization happens in normalization, not in
      strategy lookup hacks.

## Where to start

- **New strategy:** implement `decide(state: StrategyState) -> Action` and register it in `main_backtest.py`. See any file in `src/strategies/` for the pattern.
- **Run a backtest:** `python main_backtest.py --strategy <name>`. Results land in `outputs/runs/`.
- **Refresh data:** `python main_collect.py --venues binance bybit okx hyperliquid --symbols BTC ETH SOL --start YYYY-MM-DD --end YYYY-MM-DD`.
- **Understand current results:** see the backtest results table in [README.md](README.md).
