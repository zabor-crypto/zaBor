# Crypto Context Dashboard — TradingView Publication Description

## Summary

A clean **visual context dashboard** for crypto charts. It shows **present-state
conditions only** in a compact table — it makes no trade calls and does not forecast
anything.

## Full description

**Crypto Context Dashboard** is a visual context panel. It reads the current chart
and shows, in a compact table, **what the market looks like right now**.

The table is organized for quick reading:

- **Crypto Context** — title row.
- **Current** — the grouped present-state context, in plain words (e.g. *Low Vol
  Context*, *Choppy Context*, *Clean Up Context*).
- **Reading** — a one-line, human-readable summary of the current context.
- **Drivers** — the active context components behind the current state, shown as compact
  neutral tags (e.g. *Trend Up + MTF Aligned*, *Noise + Low Vol + MTF Mixed*).
- **Last Change** — what the context changed from and how many bars ago
  (e.g. *Choppy → Low Vol · 3 bars ago*).
- Details (trend, MTF, volatility, extension, noise) below.

Rows explained:

- **Trend** — `Up` / `Down` / `Mixed` (price vs a 50 EMA and its slope)
- **MTF** — `Aligned` / `Conflict` / `Mixed` (whether the chart timeframe agrees with a
  higher timeframe, computed on **confirmed** higher-timeframe candles)
- **Volatility** — `Low` / `Normal` / `High` (trailing ATR and Bollinger-width percentiles)
- **Extension** — `Normal` / `Extended` / `Very Extended` (distance from the EMA in ATR units)
- **Noise** — `Clean` / `Mixed` / `Choppy`, with `Wicky` / `Inefficient` sub-labels
  (recent candle/path structure)

**Display modes:**
- **Minimal** — Current, Reading, Last Change.
- **Simple** (default) — Current, Reading, Drivers, Last Change, MTF, Volatility.
- **Full** — Current, Reading, Drivers, Last Change, Trend, MTF, Volatility, Extension, Noise, State.

**What it is:**
- A descriptive context panel — a quick read of present conditions.
- *Reading* summarizes the current context in one neutral line.
- *Drivers* shows which context components are active right now.
- Present-state labels only, computed from price (OHLC) with trailing-only formulas.
- Higher-timeframe context is non-repainting (confirmed candles only).

**What it is NOT:**
- It gives **no advice** and makes **no trade calls**.
- It does **not forecast** future price or future conditions.
- It makes **no strategy claims** and is not a strategy.

**Visuals (kept clean):**
- **Compact event markers** — when the Current context changes, a tiny neutral marker is
  placed on the chart. *Dots* mode (default) shows only a 2-letter code (e.g. `LV`, `CH`,
  `UP`); *Labels* mode shows a brief "Context → …" tag; *Off* shows none. No arrows, no
  drivers on the chart, capped to the most recent markers to keep the chart clean.
- **Context strip** (on by default) — a subtle per-bar dot at the bottom of the pane, tinted
  by the current category. It is a low-visibility trace, not a full-height background.
- Full **background** stays **off by default**; when enabled it is very transparent and uses
  muted category colors (no green/red coding).

Use it as a context glance alongside your own analysis. Inputs (display mode, EMA / ATR
lengths, Bollinger settings, lookbacks, higher timeframe, HTF EMA length) are adjustable.

Designed for crypto charts, but it works on **any liquid market**.

Alerts are limited to neutral **state-change** notifications (context state, MTF
context, volatility, extension, noise). There are no order-related alerts.
