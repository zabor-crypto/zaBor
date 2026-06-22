# Crypto Context Dashboard v1.3 — Detailed Installation, Setup, and Usage Guide

## 1. What this dashboard is

Crypto Context Dashboard v1.3 is a visual indicator for TradingView that shows the current state of the chart in an easy-to-read table and through compact visual markers.

It is not a trading strategy. It does not give buy or sell signals. It does not forecast future price. It does not tell you where to enter or where to exit.

Its job is different: to quickly show what state the chart is in right now.

The dashboard helps you see:

* whether there is a simple up, down, or mixed context;
* whether the current timeframe agrees with the higher timeframe;
* whether the market is in low, normal, or high volatility;
* how far price has moved away from the EMA;
* how clean or noisy the recent candles look;
* when the overall context last changed.

The core idea: it is not a direction compass, it is a state dashboard for the chart.

## 2. Which coins it suits

The indicator technically works on any TradingView chart that has normal OHLC data. It contains no ticker-specific settings and is not optimized for a single coin.

It is best used on:

* BTC;
* ETH;
* SOL;
* BNB;
* large liquid altcoins;
* liquid perpetual futures;
* liquid spot pairs;
* instruments with a reasonably long history.

Best timeframes:

* 1H;
* 4H;
* 1D.

Best starting setup:

```text
Chart timeframe: 4H
HTF timeframe: 1D
```

On new listings, micro-caps, illiquid pairs, and tokens with ragged candles you can use it, but carefully. Some calculations use windows of up to 200 bars, so on a short history the readings will be less stable.

## 3. Installation in TradingView

1. Open TradingView.
2. Open the chart you need, for example `ETHUSDT 4H` or `BTCUSDT 4H`.
3. Go to the Pine Editor.
4. Open the file:

```text
indicator/crypto_context_dashboard_v1.pine
```

5. Copy the whole code.
6. Paste the code into the Pine Editor.
7. Click Add to chart.

If TradingView does not accept the line:

```text
//@version=6
```

change only that line to:

```text
//@version=5
```

The rest of the code does not need to be changed.

## 4. Recommended settings for the first run

For first use, keep the default settings:

```text
Display mode: Simple
Context markers: Dots
Show context strip: true
Show background: false
EMA length: 50
ATR length: 14
Percentile lookback: 200
BB length: 20
BB multiplier: 2.0
Noise lookback: 10
HTF timeframe: 1D
HTF EMA length: 20
```

Why these:

* `Simple` shows only the essentials and does not overload the table.
* `Dots` shows context changes as compact codes.
* `Context strip` gives a soft visual trace of the state.
* `Background` is off so the whole chart is not flooded with color.
* `HTF = 1D` works well for analyzing a 4H chart.

## 5. Display modes

The dashboard has three modes.

Minimal — shows only:

```text
Current
Reading
Last Change
```

Suits you if you want the cleanest possible panel.

Simple — the default mode:

```text
Current
Reading
Drivers
Last Change
MTF
Volatility
```

This is the best mode for most users.

Full — shows all rows:

```text
Current
Reading
Drivers
Last Change
Trend
MTF
Volatility
Extension
Noise
State
```

Suits a detailed breakdown of the chart.

## 6. What the Current row means

`Current` is the main current context of the chart.

Possible states:

```text
Low Vol Context
High Vol Context
Extended Context
Choppy Context
MTF Conflict Context
Clean Up Context
Clean Down Context
Mixed Context
```

This is not a signal. It is the name of the dominant visual state.

Examples:

`Low Vol Context` means the main feature of the chart right now is compressed volatility.
`Extended Context` means price is far from the EMA.
`Choppy Context` means the recent candles look noisy.
`MTF Conflict Context` means the current timeframe and the higher timeframe diverge.
`Clean Up Context` means the current context points visually upward.
`Clean Down Context` means the current context points visually downward.
`Mixed Context` means there is no single clean dominant state.

## 7. What the Reading row means

`Reading` is a brief, human-readable explanation of the current state.

Examples:

```text
Low-volatility context. Direction is mixed.
```
```text
Recent candles are noisy. Direction is mixed.
```
```text
Lower and higher timeframe differ.
```
```text
Price is far from EMA. Present-state only.
```

This row helps you quickly understand what exactly the dashboard is showing.

Important: `Reading` is not a recommendation. It does not tell you what to do. It only explains the current context.

## 8. What the Drivers row means

`Drivers` shows which components formed the current context.

Examples:

```text
Low Vol + Mixed Trend
```
```text
Noise + Low Vol + MTF Mixed
```
```text
Extended + High Vol
```
```text
Trend Up + MTF Aligned
```
```text
MTF Conflict + Trend Up
```

This is one of the most useful rows. It answers the question: why is the dashboard showing this particular state?

If `Current = Choppy Context` and `Drivers = Noise + Low Vol + MTF Mixed`, then the state is caused not by a trend but by a noisy structure, low volatility, and a mixed MTF.

## 9. What the Last Change row means

`Last Change` shows the last change of the overall context.

Examples:

```text
Choppy → Low Vol · 3 bars ago
```
```text
Mixed → MTF Conflict · 1 bar ago
```
```text
Initializing
```

This helps you understand whether the context just changed or has been holding for several candles.

How to read it:

* `now` or `1 bar ago` — the state is fresh;
* `5 bars ago` and more — the state has been holding for some time;
* `Initializing` — the dashboard has not yet recorded a state change on the available history.

Important: this is not a signal. A fresh context change does not mean you should open a trade.

## 10. What the MTF row means

`MTF` shows whether the current timeframe agrees with the higher timeframe.

Possible values:

```text
Aligned
Conflict
Mixed
```

`Aligned` means the current timeframe and the higher timeframe point the same way.
`Conflict` means the current timeframe and the higher timeframe diverge.
`Mixed` means there is no clear agreement or conflict.

Example:

```text
4H context = Up
1D context = Down
MTF = Conflict
```

Important: `Aligned` does not mean the move will continue. `Conflict` does not mean a trade should be canceled. It is only a description of the current relationship between the timeframes.

Technically, MTF is computed on the confirmed higher-timeframe candle. For example, if the chart is 4H and HTF = 1D, the dashboard uses the closed daily candle, not the currently forming one.

## 11. What the Volatility row means

`Volatility` shows the current volatility relative to the instrument's own history.

Possible values:

```text
Low
Normal
High
```

`Low` means ATR and Bollinger Bands width are in the lower zone of their historical range.
`High` means ATR or Bollinger Bands width are in the upper zone.
`Normal` means an in-between state.

Important: `Low` does not mean a breakout is coming soon. `High` does not mean the move will continue.

## 12. What the Extension row means

This row is visible in Full mode.

`Extension` shows how far price has moved from the EMA, in ATR units.

Possible values:

```text
Normal
Extended Up
Very Extended Up
Extended Down
Very Extended Down
```

`Extended Up` means price is noticeably above the EMA.
`Very Extended Up` means price is far above the EMA.
`Extended Down` means price is noticeably below the EMA.
`Very Extended Down` means price is far below the EMA.

Important: a strong extension does not mean a pullback will happen. It is only a description of the current distance of price from the average.

## 13. What the Noise row means

This row is visible in Full mode.

`Noise` shows the structure of the recent candles.

Possible values:

```text
Clean
Mixed
Choppy
Wicky
Inefficient
```

`Clean` means the recent candles look relatively clean.
`Mixed` means a mixed structure.
`Choppy` means a noisy structure.
`Wicky` means an elevated share of wicks.
`Inefficient` means the move was ragged and less direct.

Important: `Choppy` does not mean a whipsaw will necessarily follow. It is only a description of the recent candles.

## 14. What the bottom Context Strip means

`Context strip` is a thin visual trace of the state at the bottom of the chart.

It helps you quickly see how the context changed over the visible part of the history.

It is not trade markup. It is not entry points. It is not signals.

If the strip looks too dense, you can turn it off:

```text
Show context strip: false
```

## 15. What the markers on the chart mean

`Context markers` show the moments when the overall context changed.

Modes:

```text
Off
Dots
Labels
```

Recommended mode:

```text
Dots
```

Marker codes:

```text
LV = Low Vol
HV = High Vol
EX = Extended
CH = Choppy
MC = MTF Conflict
UP = Clean Up
DN = Clean Down
MX = Mixed
```

Markers are limited to the most recent changes. This is normal. The limit keeps the chart from being overloaded with historical labels.

If you want the cleanest possible chart:

```text
Context markers: Off
```

If you want to see full labels:

```text
Context markers: Labels
```

## 16. How to use the dashboard in real work

The correct sequence:

```text
1. First, your own trading idea or setup.
2. Then the dashboard as a context check.
3. Then a decision based on your own system.
```

The dashboard should not be the first source of a decision. It does not tell you what to do. It helps you quickly understand what state the chart is in.

## 17. Practical chart-reading checklist

Open the chart and look at the rows in this order.

Step 1. Current — first look at the main context.

```text
Current: Low Vol Context
```

This says the main feature right now is compressed volatility.

Step 2. Reading — check the human-readable explanation.

```text
Reading: Low-volatility context. Direction is mixed.
```

This immediately explains that the trend direction is not obvious.

Step 3. Drivers — look at what formed the state.

```text
Drivers: Low Vol + Mixed Trend
```

So the dashboard does not see a clean trend context and instead highlights low volatility.

Step 4. Last Change — check whether the state is fresh or has been holding.

```text
Last Change: Choppy → Low Vol · 3 bars ago
```

This helps you understand how recently the chart structure changed.

Step 5. MTF — check whether there is a conflict between the current and higher timeframe.

```text
MTF: Conflict
```

This does not cancel your idea, but it tells you the current and higher context diverge.

Step 6. Volatility — check which volatility regime it is.

```text
Volatility: Low
```

This is important for understanding the character of the market, but it is not a breakout forecast.

## 18. Examples of reading states

Example 1

```text
Current: Low Vol Context
Reading: Low-volatility context. Direction is mixed.
Drivers: Low Vol + Mixed Trend
MTF: Mixed
Volatility: Low
```

How to read it: the market is quiet right now. There is no clear direction. The higher and current timeframe do not give a strong agreement. The dashboard simply shows that the chart is in a calm, mixed state.

How to use it: compare it with your own trading system. If your system requires a trending move, the context does not yet look cleanly trending. If your system works with ranges or pre-move preparation, you can keep this chart in a watchlist. But the dashboard itself does not say there will be a breakout.

Example 2

```text
Current: MTF Conflict Context
Reading: Lower and higher timeframe differ.
Drivers: MTF Conflict + Trend Up
MTF: Conflict
Volatility: Normal
```

How to read it: on the current timeframe there is an up context, but the higher timeframe does not agree with it. This is a timeframe-conflict state.

How to use it: check whether you are trading a lower-timeframe impulse against the higher-timeframe context. The decision depends on your system: for a short-term setup a conflict may be acceptable; for a medium-term one it may require an extra check.

Example 3

```text
Current: Extended Context
Reading: Price is far from EMA. Present-state only.
Drivers: Extended + High Vol
Volatility: High
```

How to read it: price is far from the EMA right now, and volatility is elevated. This is a description of the current extension.

How to use it: do not automatically conclude that a pullback will happen. Just understand that price is far from the average and take that into account when evaluating your idea.

Example 4

```text
Current: Choppy Context
Reading: Recent candles are noisy. Direction is mixed.
Drivers: Noise + Low Vol + MTF Mixed
Noise: Choppy
```

How to read it: the recent candles are noisy. Direction is mixed. The higher context gives no clear confirmation.

How to use it: look at the structure more carefully. If your system requires a clean impulse, such a chart may be less convenient to analyze. But the dashboard does not forbid a trade and does not say a whipsaw will follow.

Example 5

```text
Current: Clean Up Context
Reading: Upward chart context. Present-state only.
Drivers: Trend Up + MTF Aligned
MTF: Aligned
```

How to read it: the current chart points visually upward, and the higher timeframe does not conflict.

How to use it: this is not a long signal. It only says the context is visually cleaner and points upward. Beyond that, your own trading logic is required.

## 19. How to make decisions with the dashboard

The dashboard does not make decisions for you.

The correct scheme:

```text
Setup → Context → Risk management → Execution
```

Where the dashboard is responsible only for the second part:

```text
Context
```

It helps you ask:

* is the current chart clean or mixed;
* is there a timeframe conflict;
* is price far from the EMA or not;
* is volatility compressed or elevated;
* do the recent candles look noisy or clean;
* did the state just change or has it been holding for several bars.

But the answers to:

* where to enter;
* where to place the stop;
* what position size to use;
* where to place the take-profit;
* whether to open a trade at all;

should come from your trading system, not from the dashboard.

## 20. How not to use the dashboard

Wrong:

```text
Current = Clean Up Context → open a position up
Current = Clean Down Context → open a position down
Low Vol Context → wait for a breakout
Extended Context → wait for a pullback
Choppy Context → definitely do not trade
MTF Conflict Context → always cancel the trade
```

There is no such logic in the indicator.

Right:

```text
Current = Clean Up Context → the chart currently points visually upward
Low Vol Context → volatility is currently compressed
Extended Context → price is currently far from the EMA
Choppy Context → the recent candles are currently noisy
MTF Conflict Context → the current and higher timeframe currently diverge
```

## 21. Best practices

Use the dashboard:

* as a quick context filter for a watchlist;
* as a visual panel before manual analysis;
* as part of a trading journal;
* as a way to quickly compare several coins;
* as a neutral layer on top of your strategy.

Do not use the dashboard:

* as a trading system;
* as a source of signals;
* as a replacement for risk management;
* as proof of an edge;
* as a forecast of a move.

## 22. Recommended workflow

For a watchlist:

1. Open several liquid coins on 4H.
2. Keep `Display mode = Simple`.
3. Keep `Context markers = Dots`.
4. Keep `Context strip = true`.
5. Quickly scan `Current`, `Reading`, `Drivers`.
6. Mark the charts whose context is clear to you.
7. Then analyze only the charts where your trading system actually sees a setup.

For a trade journal: record the dashboard state at the moment of an idea or entry:

```text
Current:
Reading:
Drivers:
MTF:
Volatility:
Last Change:
```

Over time you can compare which contexts more often accompanied your good and bad decisions. That is your own personal statistic, not a built-in claim of the indicator.

## 23. Final idea

Crypto Context Dashboard v1.3 does not show what the market will do next. It shows how the chart looks right now.

Use it as a dashboard:

```text
quickly read the context,
understand the structure,
compare coins,
record the conditions,
but do not replace your trading system.
```
