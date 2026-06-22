# Crypto Context Dashboard (TradingView)

A clean **visual context dashboard** for crypto charts. It reads the current chart and shows, in a compact table, **what the market looks like right now** — trend, multi-timeframe agreement, volatility, extension from the mean, and candle noise.

It is **not a strategy**. It makes no trade calls, gives no advice, and does not forecast price. It is a present-state context panel you read alongside your own analysis.

> Built from a research-validated state model. Every formula is **causal and trailing-only**; higher-timeframe context uses **confirmed candles only** (non-repainting by design).

---

## What it shows

| Row | Values | Meaning |
|-----|--------|---------|
| **Current** | Low Vol / High Vol / Extended / Choppy / MTF Conflict / Clean Up / Clean Down / Mixed | The dominant present-state context, in plain words |
| **Reading** | one-line summary | Human-readable explanation of the current context |
| **Drivers** | compact tags | Which components formed the current state (e.g. `Noise + Low Vol + MTF Mixed`) |
| **Last Change** | `A → B · N bars ago` | When the context last changed |
| **Trend** | Up / Down / Mixed | Price vs a 50 EMA and its slope |
| **MTF** | Aligned / Conflict / Mixed | Whether the chart timeframe agrees with a higher timeframe (confirmed candles) |
| **Volatility** | Low / Normal / High | Trailing ATR and Bollinger-width percentiles |
| **Extension** | Normal / Extended / Very Extended | Distance from the EMA in ATR units |
| **Noise** | Clean / Mixed / Choppy (+ Wicky / Inefficient) | Recent candle / path structure |

**UI modes:** `Beginner` (a simple card), `Trader`, and `Full` (every row). Optional compact event markers (`Dots` / `Labels`), a subtle per-bar context strip, and a muted background — all off or minimal by default to keep the chart clean.

---

## Installation

1. Open TradingView → **Pine Editor**
2. Paste [`crypto_context_dashboard.pine`](crypto_context_dashboard.pine)
3. **Save → Add to chart**

If TradingView rejects `//@version=6`, change only that line to `//@version=5` — no v6-exclusive syntax is used.

**Best starting setup:** chart `4H`, HTF `1D`. Works on any liquid market with normal OHLC and a reasonable history (some windows use up to 200 bars).

---

## How to use it

```
Setup → Context → Risk management → Execution
```

The dashboard owns only the **Context** step: is the chart clean or mixed, is there a timeframe conflict, is price far from the EMA, is volatility compressed or elevated, did the state just change. Entries, stops, sizing, and the decision to trade come from **your** system.

See [USER_GUIDE.md](USER_GUIDE.md) for a detailed walkthrough with examples, and [TRADINGVIEW_DESCRIPTION.md](TRADINGVIEW_DESCRIPTION.md) for the publication description.

---

## What it is NOT

- No buy/sell signals, no advice, no trade calls.
- No price or condition forecasts.
- Not a strategy and makes no performance claims.
- Alerts are limited to neutral **state-change** notifications — there are no order-related alerts.

---

## Design notes

- **Causal / trailing-only formulas** — no lookahead.
- **Non-repainting HTF** — `request.security` with `lookahead_off`; the higher-timeframe row updates once per completed HTF candle.
- `ta.percentrank` over ~200 bars needs that much history before the volatility/noise percentiles are fully meaningful.

---

## License

MIT — see [LICENSE](LICENSE).
