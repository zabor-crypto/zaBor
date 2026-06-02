"""Core event-loop backtest engine.

Design choices:

  * The engine is *strategy-agnostic*: it receives a callable
    ``decide(state) -> Action`` and a set of normalized data panels.
  * The simulation clock advances on a fixed grid (default: 5 min). Funding
    is settled lazily — at every funding timestamp, all open positions of
    the corresponding venue/symbol receive the funding pnl exactly once.
  * Trading fees and slippage are charged at entry and exit, not at
    rebalance ticks.
  * No look-ahead: ``state`` only ever exposes data with timestamp <
    current clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Union

import pandas as pd

from ..utils.io import write_parquet
from ..utils.logging import get_logger
from .costs import CostModel
from .funding_settlement import funding_pnl as compute_funding_pnl
from .margin import LegState, free_margin_ratio


# ---- Action types ------------------------------------------------------- #

@dataclass(frozen=True)
class OpenLeg:
    venue: str
    symbol: str
    direction: int
    notional_usd: float
    leverage: float = 1.0
    role: str = "taker"
    depth_notional: Optional[float] = None
    tag: str = ""


@dataclass(frozen=True)
class CloseAll:
    reason: str = "exit"


# ``Action`` is a *list of legs* representing one atomic position change, or
# CloseAll. We deliberately keep the interface tiny.
Action = Optional[Union[list[OpenLeg], CloseAll]]


# ---- State exposed to the strategy -------------------------------------- #

@dataclass
class StrategyState:
    now: pd.Timestamp
    funding: dict[tuple[str, str], pd.DataFrame]   # (venue, symbol) -> truncated funding history
    prices: dict[tuple[str, str], pd.DataFrame]    # (venue, symbol) -> truncated price history
    open_legs: list[LegState]                      # currently open
    equity_usd: float
    capital_usd: float
    config: dict
    # Optional: per (venue, symbol) latest depth_notional_usd, used by the
    # cost model. Strategies do not need to populate this; the engine reads
    # it from a depth panel passed to ``run``.
    depth_notional: dict[tuple[str, str], float] = field(default_factory=dict)


class StrategyFn(Protocol):
    def __call__(self, state: StrategyState) -> Action: ...


# ---- Engine ------------------------------------------------------------- #

@dataclass
class EngineConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    step: timedelta = timedelta(minutes=5)
    capital_usd: float = 1_000_000.0
    liquidation_buffer_min: float = 0.30
    data_staleness_seconds: int = 600
    max_strategy_dd: float = 0.15


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    venue: str
    symbol: str
    direction: int
    entry_price: float
    exit_price: Optional[float]
    notional_usd: float
    funding_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    slippage_bps: float = 0.0
    net_pnl_usd: float = 0.0
    liquidation_buffer_min: float = float("inf")
    exit_reason: str = ""
    tag: str = ""


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    closed_count: int = 0
    open_count: int = 0
    schema_issues: list[dict] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, cfg: EngineConfig, costs: CostModel, strategy_config: dict):
        self.cfg = cfg
        self.costs = costs
        self.strategy_config = strategy_config
        self.logger = get_logger("funding_arb.backtest.engine")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _slice_until(self, df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
        if df.empty:
            return df
        return df.loc[df["timestamp_utc"] < t]

    def _last_price(self, prices_panel: pd.DataFrame, t: pd.Timestamp) -> float | None:
        df = self._slice_until(prices_panel, t)
        if df.empty:
            return None
        # Prefer mid_price, then mark_price, then bid/ask average.
        for col in ("mid_price", "mark_price"):
            if col in df.columns:
                v = df[col].dropna()
                if not v.empty:
                    return float(v.iloc[-1])
        if {"bid", "ask"}.issubset(df.columns):
            ba = df[["bid", "ask"]].dropna()
            if not ba.empty:
                return float((ba["bid"].iloc[-1] + ba["ask"].iloc[-1]) / 2.0)
        return None

    def _settle_funding_into_legs(
        self,
        legs: list[LegState],
        funding_panels: dict[tuple[str, str], pd.DataFrame],
        prev: pd.Timestamp,
        now: pd.Timestamp,
    ) -> float:
        """Apply funding pnl to each leg for funding events strictly within (prev, now]."""
        total_pnl = 0.0
        for leg in legs:
            panel = funding_panels.get((leg.venue, leg.symbol))
            if panel is None or panel.empty:
                continue
            entries = compute_funding_pnl(
                panel,
                direction=leg.direction,
                notional_usd=leg.notional_usd,
                venue=leg.venue,
                symbol=leg.symbol,
                start=prev, end=now,
            )
            if not entries:
                continue
            pnl = sum(e.pnl_usd for e in entries)
            leg.cumulative_funding_pnl += pnl
            total_pnl += pnl
        return total_pnl

    def _data_stale(
        self,
        legs: list[LegState],
        prices_panels: dict[tuple[str, str], pd.DataFrame],
        now: pd.Timestamp,
        funding_panels: dict[tuple[str, str], pd.DataFrame] | None = None,
    ) -> bool:
        """Detect data staleness for any leg's primary data feed.

        We prefer the price panel; if missing or empty, fall back to the
        leg's funding panel (HL realized-funding rows have no markPrice
        column, so a funding-only venue would otherwise look perpetually
        stale and force-exit on every tick).
        """
        max_age = pd.Timedelta(seconds=self.cfg.data_staleness_seconds)
        for leg in legs:
            panel = prices_panels.get((leg.venue, leg.symbol))
            df = self._slice_until(panel, now) if panel is not None else None
            if df is None or df.empty:
                if funding_panels is None:
                    return True
                fp = funding_panels.get((leg.venue, leg.symbol))
                df = self._slice_until(fp, now) if fp is not None else None
                if df is None or df.empty:
                    return True
                # Funding cadence is up to ``funding_interval_hours`` so we
                # tolerate up to 1.5× that in addition to ``max_age``.
                interval_h = 1
                if "funding_interval_hours" in df.columns:
                    last_iv = df["funding_interval_hours"].dropna()
                    if not last_iv.empty:
                        interval_h = int(last_iv.iloc[-1])
                tolerance = pd.Timedelta(hours=int(interval_h * 1.5)) + max_age
                if (now - df["timestamp_utc"].iloc[-1]) > tolerance:
                    return True
                continue
            if (now - df["timestamp_utc"].iloc[-1]) > max_age:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _latest_depth_notional(
        self,
        depth_panels: dict[tuple[str, str], pd.DataFrame],
        now: pd.Timestamp,
    ) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for key, df in depth_panels.items():
            if df.empty:
                continue
            sub = df.loc[df["timestamp_utc"] <= now]
            if sub.empty:
                continue
            last = sub.iloc[-1]
            # Use the smaller side as the binding constraint for a delta-neutral leg
            bid = float(last.get("bid_notional_usd") or 0.0)
            ask = float(last.get("ask_notional_usd") or 0.0)
            band = min(b for b in (bid, ask) if b > 0) if (bid > 0 and ask > 0) else max(bid, ask)
            if band > 0:
                out[key] = band
        return out

    def run(
        self,
        strategy: StrategyFn,
        funding_panels: dict[tuple[str, str], pd.DataFrame],
        price_panels: dict[tuple[str, str], pd.DataFrame],
        depth_panels: Optional[dict[tuple[str, str], pd.DataFrame]] = None,
    ) -> BacktestResult:
        depth_panels = depth_panels or {}
        cfg = self.cfg
        equity = cfg.capital_usd
        peak_equity = equity
        legs: list[LegState] = []
        open_record_by_leg: dict[int, TradeRecord] = {}   # id(leg) -> record
        closed: list[TradeRecord] = []
        equity_curve: list[dict] = []
        marks: dict[tuple[str, str], float] = {}

        clock = cfg.start
        prev_clock = cfg.start
        steps = 0
        while clock <= cfg.end:
            steps += 1
            # --- 1. mark legs to last observed price ----------------------
            for leg in legs:
                last = self._last_price(price_panels.get((leg.venue, leg.symbol), pd.DataFrame()), clock)
                if last is not None:
                    marks[(leg.venue, leg.symbol)] = last

            # --- 2. settle funding for the elapsed window -----------------
            funding_step_pnl = self._settle_funding_into_legs(
                legs, funding_panels, prev_clock, clock,
            )
            equity += funding_step_pnl

            # --- 3. compute equity (cash + unrealized) --------------------
            unreal = 0.0
            for leg in legs:
                m = marks.get((leg.venue, leg.symbol))
                if m is not None:
                    unreal += leg.mark_to_market_pnl(m)
            mark_equity = equity + unreal
            peak_equity = max(peak_equity, mark_equity)
            drawdown = 1.0 - mark_equity / peak_equity if peak_equity > 0 else 0.0

            # update per-leg liquidation_buffer_min
            if legs:
                fmr = free_margin_ratio(legs, marks)
                for leg in legs:
                    rec = open_record_by_leg.get(id(leg))
                    if rec is not None:
                        rec.liquidation_buffer_min = min(rec.liquidation_buffer_min, fmr)
            else:
                fmr = float("inf")

            # --- 4. forced exits ------------------------------------------
            forced_close = False
            forced_reason = ""
            if legs:
                if drawdown >= cfg.max_strategy_dd:
                    forced_close = True
                    forced_reason = f"max_dd_breach({drawdown:.3f})"
                elif fmr < cfg.liquidation_buffer_min:
                    forced_close = True
                    forced_reason = f"liq_buffer_breach({fmr:.3f})"
                elif self._data_stale(legs, price_panels, clock, funding_panels):
                    forced_close = True
                    forced_reason = "data_stale"

            # --- 5. invoke strategy ---------------------------------------
            action: Action = None
            depth_lookup = self._latest_depth_notional(depth_panels, clock)
            if not forced_close:
                state = StrategyState(
                    now=clock,
                    funding={k: self._slice_until(v, clock) for k, v in funding_panels.items()},
                    prices={k: self._slice_until(v, clock) for k, v in price_panels.items()},
                    open_legs=list(legs),
                    equity_usd=mark_equity,
                    capital_usd=cfg.capital_usd,
                    config=self.strategy_config,
                    depth_notional=depth_lookup,
                )
                try:
                    action = strategy(state)
                except Exception as e:  # never let a strategy bug silently kill a run
                    self.logger.exception("strategy raised at %s: %s", clock, e)
                    action = None

            # --- 6. apply exits -------------------------------------------
            if forced_close or isinstance(action, CloseAll):
                reason = forced_reason or (action.reason if isinstance(action, CloseAll) else "exit")
                for leg in legs:
                    mark = marks.get((leg.venue, leg.symbol)) or leg.entry_price
                    notional = leg.notional_usd
                    exit_fee_bps = self.costs.exit_cost_bps(
                        leg.venue, role="taker",
                        order_notional=notional,
                        depth_notional=depth_lookup.get((leg.venue, leg.symbol)),
                    )
                    fee_usd = notional * exit_fee_bps * 1e-4
                    leg.cumulative_fee_usd += fee_usd
                    equity -= fee_usd
                    rec = open_record_by_leg.pop(id(leg), None)
                    if rec is None:
                        continue
                    rec.exit_time = clock
                    rec.exit_price = mark
                    rec.fees_usd += fee_usd
                    rec.exit_reason = reason
                    mtm = leg.mark_to_market_pnl(mark)
                    rec.funding_pnl_usd = leg.cumulative_funding_pnl
                    rec.net_pnl_usd = mtm + leg.cumulative_funding_pnl - leg.cumulative_fee_usd
                    closed.append(rec)
                # After closing, mtm of legs should already be in `equity`
                for leg in legs:
                    m = marks.get((leg.venue, leg.symbol)) or leg.entry_price
                    equity += leg.mark_to_market_pnl(m)
                legs = []
                marks = {}

            # --- 7. apply entries -----------------------------------------
            elif isinstance(action, list) and action:
                # Atomic multi-leg open: if any leg has no price, reject
                # the whole action. A partial fill on a delta-neutral
                # action would leave us directional.
                resolved_prices: list[float | None] = [
                    self._last_price(
                        price_panels.get((spec.venue, spec.symbol), pd.DataFrame()),
                        clock,
                    )
                    for spec in action
                ]
                if any(p is None for p in resolved_prices):
                    missing = [
                        f"{a.venue}/{a.symbol}"
                        for a, p in zip(action, resolved_prices) if p is None
                    ]
                    self.logger.warning(
                        "skipping atomic action at %s: missing price for %s",
                        clock, ",".join(missing),
                    )
                    # carry on without opening — the strategy will be re-asked next tick
                    pass
                for spec, entry_price in zip(action, resolved_prices):
                    if any(p is None for p in resolved_prices):
                        break
                    if entry_price is None:
                        self.logger.warning(
                            "skipping leg open %s/%s at %s: no price",
                            spec.venue, spec.symbol, clock,
                        )
                        continue
                    fee_bps = self.costs.entry_cost_bps(
                        spec.venue, role=spec.role,
                        order_notional=spec.notional_usd,
                        depth_notional=(
                            spec.depth_notional
                            if spec.depth_notional is not None
                            else depth_lookup.get((spec.venue, spec.symbol))
                        ),
                    )
                    fee_usd = spec.notional_usd * fee_bps * 1e-4
                    equity -= fee_usd
                    leg = LegState(
                        venue=spec.venue, symbol=spec.symbol,
                        direction=spec.direction,
                        entry_price=entry_price,
                        notional_usd=spec.notional_usd,
                        leverage=spec.leverage,
                        cumulative_fee_usd=fee_usd,
                    )
                    legs.append(leg)
                    rec = TradeRecord(
                        entry_time=clock, exit_time=None,
                        venue=spec.venue, symbol=spec.symbol,
                        direction=spec.direction,
                        entry_price=entry_price, exit_price=None,
                        notional_usd=spec.notional_usd,
                        fees_usd=fee_usd,
                        slippage_bps=fee_bps - self.costs.fee_bps(spec.venue, spec.role),
                        tag=spec.tag,
                    )
                    open_record_by_leg[id(leg)] = rec

            # --- 8. record equity -----------------------------------------
            unreal = sum(
                leg.mark_to_market_pnl(marks.get((leg.venue, leg.symbol), leg.entry_price))
                for leg in legs
            )
            equity_curve.append({
                "timestamp_utc": clock,
                "equity_usd": equity + unreal,
                "n_open_legs": len(legs),
                "drawdown": drawdown,
            })

            prev_clock = clock
            clock = clock + cfg.step

        # --- 9. close any still-open legs at the final clock ---------------
        final_depth = self._latest_depth_notional(depth_panels, cfg.end)
        for leg in legs:
            mark = marks.get((leg.venue, leg.symbol)) or leg.entry_price
            notional = leg.notional_usd
            exit_fee_bps = self.costs.exit_cost_bps(
                leg.venue, role="taker", order_notional=notional,
                depth_notional=final_depth.get((leg.venue, leg.symbol)),
            )
            fee_usd = notional * exit_fee_bps * 1e-4
            equity -= fee_usd
            rec = open_record_by_leg.pop(id(leg), None)
            if rec is not None:
                rec.exit_time = cfg.end
                rec.exit_price = mark
                rec.fees_usd += fee_usd
                rec.exit_reason = "end_of_backtest"
                rec.funding_pnl_usd = leg.cumulative_funding_pnl
                rec.net_pnl_usd = (
                    leg.mark_to_market_pnl(mark)
                    + leg.cumulative_funding_pnl
                    - (leg.cumulative_fee_usd + fee_usd)
                )
                closed.append(rec)

        # Build dataframes
        equity_df = pd.DataFrame(equity_curve)
        if equity_df.empty:
            equity_df = pd.DataFrame(columns=["timestamp_utc", "equity_usd", "n_open_legs", "drawdown"])
        trades_df = pd.DataFrame([t.__dict__ for t in closed])

        return BacktestResult(
            equity=equity_df,
            trades=trades_df,
            closed_count=len(closed),
            open_count=len(legs),
        )


def write_result(result: BacktestResult, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(result.equity, run_dir / "equity_curve.parquet")
    result.equity.to_csv(run_dir / "equity_curve.csv", index=False)
    write_parquet(result.trades, run_dir / "trades.parquet")
