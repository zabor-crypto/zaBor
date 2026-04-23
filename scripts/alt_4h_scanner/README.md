# ALT 4H Pullback Scanner

Scans Binance spot altcoin pairs at every 4H candle close. Detects accumulation breakouts and generates structured limit-entry signals for buying the first pullback after an impulse move.

**Backtested on 3 months / 636 trades (Binance spot, all USDT pairs):**

| Filter stack | N signals | Win Rate | Profit Factor | Max DD |
|-------------|-----------|----------|---------------|--------|
| Full (RSI ≤ 80) | 41 | 95.7% | 11.81 | −18.7% |
| Relaxed (all)   | 70 | 90.0% | 7.06  | −21.1% |

> **Disclaimer:** Backtest results do not guarantee future performance. This is a signal generation tool — execution, position sizing, and risk management are your responsibility.

---

## Signal format

```
#longalt
Pair: CETUSUSDT
Entries:  0.02831, 0.02735, 0.02607, 0.02479, 0.02351
Targets:  0.03004 (60%), 0.03177 (20%), 0.03407 (10%), 0.03695 (7%), 0.04271 (3%)
Stop-loss: 4H close below 0.02255
```

Signals are logged to file and optionally sent to a Telegram bot.

---

## Strategy logic

### Why accumulation breakouts on 4H?

Large-cap altcoins frequently consolidate under resistance for weeks before breaking out on volume. The first candle that closes above a 100-bar high with momentum is the signal event. The edge is in **not chasing** — instead placing limit orders on the inevitable pullback into the new support zone (former resistance, now EMA55/cloud).

### Filter stack (applied in order)

| # | Filter | What it checks | Why |
|---|--------|---------------|-----|
| 1 | **EMA55 trend** | Price > EMA55 on 4H | Only trade in uptrend |
| 2 | **Ichimoku cloud** | Price > Kumo cloud top | Confirms structural support below |
| 3 | **100-bar breakout** | Current 4H sets new 100-bar high | Accumulation exit confirmed |
| 4 | **Momentum** | Close ≥ 8% above close 10 candles ago (40H) | Filters weak/slow breakouts |
| 5 | **Bullish body** | Breakout candle: close > open | Real buying pressure, not a wick |
| 6 | **Prev-candle quality** | Candle before breakout closes in top 40% of range | Momentum build-up confirmed |
| 7 | **1H RSI** | 1H RSI(14) ≤ 80 at breakout | Rejects already-overbought setups |
| 8 | **BTC regime** | BTC 4H close vs EMA55 | Tightens risk range in downtrend |
| 9 | **Liquidity** | Median daily volume ≥ $300k USDT | Prevents illiquid pair signals |

### Entry ladder

5 limit levels from breakout close down to EMA55/cloud support:
```
entry[0] = 90% of the breakout-to-support span  (closest to current price)
entry[1] = 75%
entry[2] = 55%
entry[3] = 35%
entry[4] = 15%  (deepest, near EMA55/cloud)
```

**Key insight from backtest:** Trades filling only entries 1–3 have 100% win rate and +4.8–8.7% avg gain. Trades filling all 5 entries drop to ~50% WR. Use limit orders only — let price come to you.

### Take-profit levels

R-multiples from entry[0], where R = entry[0] − stop-loss:

| Level | R-mult | Position % |
|-------|--------|-----------|
| TP1 | 0.3R | 60% |
| TP2 | 0.6R | 20% |
| TP3 | 1.0R | 10% |
| TP4 | 1.5R | 7% |
| TP5 | 2.5R | 3% |

Front-loaded exits (60% at TP1) preserve capital while keeping a runner.

### Stop-loss

EMA55 value at signal time, floored to tick. Triggered on a **4H candle close** below this level — not on an intrabar wick.

### Position monitoring

When price reaches entry[2], TPs are recalculated from the actual average entry price of entries 1–3. This tightens targets relative to real position cost and improves hit probability. A Telegram update is sent automatically.

### BTC regime adaptation

- BTC above EMA55 (uptrend): risk range 10–25%
- BTC below EMA55 (downtrend): risk range 12–25%
- Backtest: BTC downtrend + RSI < 80 → PF 26.07, WR 88.9%

---

## Quickstart

```bash
cd scripts/alt_4h_scanner
pip install -r requirements.txt

export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"   # optional
export TELEGRAM_CHAT_ID="your_chat_id"        # optional

python signal_bot.py
```

The scanner aligns itself to the 4H candle grid and runs indefinitely.
Signals are logged to `signal_bot_spot_long_pullback.log`.

---

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | — | Read-only Binance API key |
| `BINANCE_API_SECRET` | — | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token (optional) |
| `TELEGRAM_CHAT_ID` | — | Telegram chat/channel ID (optional) |
| `MAX_1H_RSI` | `80` | Max 1H RSI at signal time. `0` = disable |
| `REQUIRE_BULLISH_BODY` | `true` | Breakout candle must be bullish |
| `PREV_CLOSE_POS` | `0.60` | Min close position of prev candle. `0` = disable |
| `BTC_REGIME_ENABLED` | `true` | Adapt risk range to BTC trend |
| `MOMENTUM_MIN_PCT` | `8.0` | Min 40H momentum % |
| `LIQ_MIN_MEDIAN_USDT` | `300000` | Min median daily volume |
| `SIGNAL_COOLDOWN_H` | `24` | Hours before re-signaling same pair |
| `E3_RECALC_ENABLED` | `true` | Enable entry-3 TP recalculation monitor |
| `ENTRY_WINDOW_HOURS` | `24` | Communicated in signal: cancel unfilled entries after N hours |

---

## How to use signals

1. Set limit orders at all 5 entry levels after signal arrives
2. Cancel unfilled orders after 24 hours if price did not pull back
3. Stop-loss triggers on a **4H candle close** — not an intrabar wick
4. Take 60% off at TP1, trail the rest 6% below the highest 4H high after TP1
5. If entry-3 is filled: wait for the Telegram update with recalculated TPs

---

## What is NOT included

This tool generates signals only. To build a complete trading system you would need:

- **Execution layer** — order placement, partial fill handling, cancel/replace logic
- **Position sizing** — fixed-fraction or Kelly based on account equity
- **Portfolio risk controls** — max concurrent positions, correlation filter, total exposure cap
- **Live performance tracking** — P&L per signal, slippage vs backtest comparison
- **Backtesting framework** — the results above were produced from a separate backtest suite
- **Multi-exchange routing** — Bybit, OKX adapters (Binance adapter included)

---

## License

MIT — see repo root [LICENSE](../../LICENSE).
