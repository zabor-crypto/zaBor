# zaBor RSI + AO + Stochastic Entry System  
## User Manual — Version 0.2

---

## 1. Purpose

This TradingView indicator identifies **BUY / SELL entry points** using a structured multi-stage confirmation system based on:

- RSI divergences (regular & hidden)
- RSI momentum confirmation
- Optional Awesome Oscillator (AO) confirmation
- Stochastic crossover with **reversal continuation filter**
- Anti-spam cooldown logic

The system is designed as an **entry framework**, not a standalone trading strategy.

Optimized for:
- **1H** — balanced frequency
- **4H** — high-confidence reversal entries

---

## 2. Indicator Structure

The system consists of **two Pine Script indicators**:

1. **RSI + AO + Stochastic System**  
   Lower panel: RSI, Stochastic, AO, divergences, setups

2. **Entry Signals (Chart Overlay)**  
   BUY / SELL arrows on the main price chart

Both indicators must be added to the chart.

---

## 3. Installation (TradingView)

### Step 1 — Panel Indicator

1. Open TradingView → **Pine Editor**
2. New → **Blank indicator**
3. Paste code from  
   `zaBor_RSI_AO_Stoch_indicator.pine`
4. Save → **Add to chart**

### Step 2 — Chart Overlay

1. Pine Editor → New → **Blank indicator**
2. Paste code from  
   `zaBor_RSI_AO_Stoch_chart.pine`
3. Save → **Add to chart**

---

## 4. Signal Logic Overview (v0.2)

### Stage 1 — Setup Creation

A setup is created when **at least one condition** is met:

- Regular RSI divergence
- Hidden RSI divergence (trend-filtered)
- RSI oversold / overbought
- RSI rebound toward mid-level (50)

Setup validity window: **50 bars**  
If no trigger occurs — setup expires automatically.

---

### Stage 2 — Momentum Confirmation (RSI)

New in **v0.2**

After setup:
- RSI must confirm momentum **in signal direction**
- Allowed tolerance: **1–2 bars**
- Prevents early counter-trend entries

---

### Stage 3 — Optional AO Confirmation

Disabled by default.

If enabled:
- **Slope mode**: AO must move in signal direction
- **Zero-cross mode**: AO crosses zero within setup window

---

### Stage 4 — Stochastic Trigger + Reversal Confirmation

Final entry trigger:

- BUY:
  - %K crosses above %D
  - %K < 35
  - %K continues upward after crossover

- SELL:
  - %K crosses below %D
  - %K > 65
  - %K continues downward after crossover

This **reversal continuation filter (v0.2)** removes premature stochastic signals.

---

## 5. Divergence Types

### Regular Divergence (Reversal)

- Bullish: price LL + RSI HL
- Bearish: price HH + RSI LH

### Hidden Divergence (Trend Continuation)

- Hidden bullish: price HL + RSI LL
- Hidden bearish: price LH + RSI HH

Hidden divergences are enabled **only when trend aligns**
(EMA20 vs EMA50).

---

## 6. Anti-Spam Protection

- Cooldown: **15 bars** after each signal
- Separate cooldown for BUY and SELL
- Setup auto-expiry if not triggered

---

## 7. Recommended Presets

### 1H (Balanced)

- Signal Mode: Divergence + Momentum
- Hidden Divergences: ON
- RSI Levels: 35 / 65
- Stochastic K: 35 / 65
- AO Filter: OFF

Expected: 4–8 signals per week

---

### 4H (High Confidence)

- Signal Mode: Divergence Only
- Min Price Divergence: 1.0
- Min RSI Divergence: 5.0
- Cooldown: 20 bars

Expected: 1–3 signals per week

---

## 8. Alerts

1. Click indicator → **Create Alert**
2. Condition: Entry BUY or Entry SELL
3. Trigger: **Once Per Bar Close**
4. Enable desired notifications

---

## 9. Interpretation

- Green arrow + green panel marker → BUY
- Red arrow + red panel marker → SELL
- Divergence lines show setup origin
- Best signals occur near range extremes or trend pullbacks

---

## 10. Risk Notes

- Always use stop-loss
- Recommended R:R ≥ 1:2
- Do not trade blindly against higher timeframe trend
- Indicator provides **signals, not advice**

---

## 11. Version Notes

**v0.2 improvements**
- Reversal confirmation added
- RSI momentum filter
- Reduced premature entries
- Cleaner signal clustering

---
