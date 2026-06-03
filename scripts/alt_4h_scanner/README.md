# ALT 4H Pullback Scanner

Scans Binance spot pairs at every 4H candle close. Detects accumulation breakouts against a 9-filter stack and computes structured limit-order levels for a pullback entry strategy.

This is a **signal generation and logging tool** — it produces structured output for manual or automated review. Execution, position sizing, and risk management are not included.

---

## Strategy logic

Large-cap altcoins frequently consolidate under resistance before breaking out on volume. The scanner identifies the first 4H candle that closes above a 100-bar high with momentum, then computes 5 limit-order entry levels spaced between the breakout close and the EMA55/cloud support zone.

### Filter stack (applied in order)

| # | Filter | What it checks |
|---|--------|---------------|
| 1 | **EMA55 trend** | Price > EMA55 on 4H |
| 2 | **Ichimoku cloud** | Price > Kumo cloud top |
| 3 | **100-bar breakout** | Current 4H sets new 100-bar high |
| 4 | **Momentum** | Close ≥ 8% above close 10 candles ago (40H) |
| 5 | **Bullish body** | Breakout candle: close > open |
| 6 | **Prev-candle quality** | Prev candle closes in top 40% of range |
| 7 | **1H RSI** | 1H RSI(14) ≤ 80 at breakout |
| 8 | **BTC regime** | BTC 4H close vs EMA55 |
| 9 | **Liquidity** | Median daily volume ≥ $300k USDT |

### Entry level computation

5 limit levels computed from breakout close to EMA55/cloud support:

```
entry[0] = 90% of span  (closest to current price)
entry[1] = 75%
entry[2] = 55%
entry[3] = 35%
entry[4] = 15%  (deepest, near EMA55/cloud)
```

### Take-profit computation

R-multiples from `entry[0]`, where `R = entry[0] − stop_level`:

| Level | R-mult | Fraction |
|-------|--------|---------|
| TP1 | 0.3R | 60% |
| TP2 | 0.6R | 20% |
| TP3 | 1.0R | 10% |
| TP4 | 1.5R | 7% |
| TP5 | 2.5R | 3% |

### Stop-loss computation

EMA55 value at signal time, floored to tick. Reference level for a **4H candle close** exit — not an intrabar wick.

### Entry-3 recalculation

When price reaches `entry[2]`, TPs are recalculated from the average of entries 1–3 and logged. An optional webhook notification is triggered.

### BTC regime adaptation

- BTC above EMA55: risk range 10–25%
- BTC below EMA55: risk range 12–25%

---

## Output format

Each detected setup is logged to file with the following fields:

```
pair: CETUSUSDT
entries:  [0.02831, 0.02735, 0.02607, 0.02479, 0.02351]
targets:  [0.03004, 0.03177, 0.03407, 0.03695, 0.04271]
fractions:[0.60, 0.20, 0.10, 0.07, 0.03]
stop_level: 0.02255
regime: btc_uptrend
```

Optionally, the same data is posted to a configured Telegram endpoint (webhook mode).

---

## Quickstart

```bash
cd scripts/alt_4h_scanner
pip install -r requirements.txt

export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"   # optional webhook
export TELEGRAM_CHAT_ID="your_chat_id"        # optional webhook

python signal_bot.py
```

The scanner aligns to the 4H candle grid and runs indefinitely.
Output is written to `signal_bot_spot_long_pullback.log`.

---

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | — | Read-only Binance API key |
| `BINANCE_API_SECRET` | — | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | — | Telegram webhook token (optional) |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID for webhook (optional) |
| `MAX_1H_RSI` | `80` | Max 1H RSI at signal time. `0` = disable |
| `REQUIRE_BULLISH_BODY` | `true` | Breakout candle must be bullish |
| `PREV_CLOSE_POS` | `0.60` | Min close position of prev candle. `0` = disable |
| `BTC_REGIME_ENABLED` | `true` | Adapt risk range to BTC trend |
| `MOMENTUM_MIN_PCT` | `8.0` | Min 40H momentum % |
| `LIQ_MIN_MEDIAN_USDT` | `300000` | Min median daily volume |
| `SIGNAL_COOLDOWN_H` | `24` | Hours before re-triggering same pair |
| `E3_RECALC_ENABLED` | `true` | Enable entry-3 TP recalculation |
| `ENTRY_WINDOW_HOURS` | `24` | Cancel window communicated in output |

---

## What is NOT included

- **Execution layer** — order placement, fill handling, cancel/replace logic
- **Position sizing** — this tool does not recommend allocation sizes
- **Portfolio risk controls** — max concurrent positions, exposure caps
- **Live performance tracking** — no P&L tracking in this tool
- **Backtesting framework** — signal logic only; backtest infra is separate

---

## Disclaimer

This tool generates structured output based on technical filters. It is not financial advice and does not represent a recommendation to buy or sell any asset. Past filter performance does not predict future results. Use at your own risk.

---

## License

MIT — see repo root [LICENSE](../../LICENSE).
