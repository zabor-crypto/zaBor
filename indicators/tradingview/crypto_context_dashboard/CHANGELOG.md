# Changelog — Crypto Context Dashboard

All UI/UX iterations are visual only — the indicator formulas and the final-state logic are unchanged from the validated spec.

## v2 — Market Lens UI
- Simplified default UI: a compact "Market Lens" card.
- UI modes: **Beginner** (card), **Trader**, **Full** (every row).
- Context markers (Off / Dots / Labels) with a 2-letter code per context change; capped to the most recent markers.
- Optional per-bar context strip and muted-color background (off/minimal by default).

## v1.3
- **Compact event markers** — `Context markers`: Off / Dots / Labels. Dots shows a 2-letter code per context change; Labels shows a brief "Context → …" tag. No arrows; capped at the most recent 25 markers.
- **Last Change** row — `<previous> → <current> · N bars ago`, or `Initializing`.
- **Context strip** — optional subtle per-bar dot at the bottom of the pane.
- Muted category colors; full background off by default.

## v1.2
- Added the **Reading** and **Drivers** rows.

## v1.1
- Added display modes + human-readable labels.

## v1.0
- Initial visual-only present-state context dashboard: Trend, MTF, Volatility, Extension, Noise, final State. Causal/trailing-only formulas; non-repainting HTF context.
