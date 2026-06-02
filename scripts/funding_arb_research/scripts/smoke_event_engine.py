"""End-to-end smoke for the event-driven engine.

Builds a synthetic funding panel (Bitget BTCUSDT funding -50% APR,
Binance BTCUSDT +5% APR) over 30 days at 8h cadence, runs the engine
with the synthetic trigger, and prints the trade tape.

Expected outcome: the engine should open Bitget-long (negative funding
receiver) + Binance-short positions at each trigger fire, accrue
positive funding pnl every 8h on the Bitget leg (roughly net +45% APR),
and close on either the carry-falls-below-threshold rule or max-hold.

Run from the funding_arb dir:

    python -m scripts.smoke_event_engine
"""
from __future__ import annotations

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.event_engine import EngineConfig, EventEngine
from src.events.triggers.synthetic import SyntheticTrigger
from src.routing.hedge_router import HedgeRouter
from src.routing.venue_capabilities import VenueCapabilities


def _build_panel(venue: str, symbol: str, rate_per_period: float,
                 start: pd.Timestamp, end: pd.Timestamp,
                 interval_hours: int = 8) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq=f"{interval_hours}h", tz="UTC", inclusive="left")
    return pd.DataFrame({
        "timestamp_utc": ts,
        "venue": venue,
        "symbol": symbol,
        "funding_rate": rate_per_period,
        "mark_price": 70_000.0,
    })


def main() -> None:
    start = pd.Timestamp("2026-04-01", tz="UTC")
    end = pd.Timestamp("2026-04-30", tz="UTC")
    # Target APRs:
    #   Bitget -50% APR @ 8h = -50% / (365*24/8) = -0.000457 / settle
    #   Binance  +5% APR @ 8h = +0.0000457 / settle
    bitget_rate = -0.50 / (365 * 24 / 8)
    binance_rate = +0.05 / (365 * 24 / 8)
    panels = {
        ("bitget", "BTCUSDT"): _build_panel("bitget", "BTCUSDT", bitget_rate, start, end),
        ("binance", "BTCUSDT"): _build_panel("binance", "BTCUSDT", binance_rate, start, end),
    }
    contracts = pd.DataFrame([
        {"venue": "bitget", "symbol": "BTCUSDT", "base_asset": "BTC",
         "quote_asset": "USDT", "fund_interval_hours": 8},
        {"venue": "binance", "symbol": "BTCUSDT", "base_asset": "BTC",
         "quote_asset": "USDT", "fund_interval_hours": 8},
    ])

    cfg = EngineConfig(
        start=start, end=end, capital_usd=1_000_000,
        per_event_max_pct=0.10, exit_threshold_apr=0.05,
        max_hold_hours=72, exit_check_interval_minutes=60, role="taker",
    )
    caps = VenueCapabilities()
    router = HedgeRouter(caps=caps, allowed_hedge_venues=["binance", "bybit", "okx"])
    costs = CostModel.from_yaml()
    trig = SyntheticTrigger(coins=["BTC"], check_interval_hours=24)
    engine = EventEngine(cfg=cfg, triggers=[trig], router=router, costs=costs,
                         funding_panels=panels, contracts_meta=contracts)
    res = engine.run()
    df = res.to_dataframe()
    print(f"trades: {len(df)}, skipped: {res.skipped_events}, final equity: ${res.final_equity_usd:,.2f}")
    if not df.empty:
        cols = ["coin", "long_venue", "short_venue", "opened", "closed", "hold_hours",
                "close_reason", "long_funding_usd", "short_funding_usd", "fees_usd",
                "realized_pnl_usd"]
        print(df[cols].to_string(index=False))
        print(f"\nsum realized_pnl: ${df.realized_pnl_usd.sum():,.2f}")
        print(f"sum long_funding (Bitget receives): ${df.long_funding_usd.sum():,.2f}")
        print(f"sum short_funding (Binance pays): ${df.short_funding_usd.sum():,.2f}")
        print(f"sum fees: ${df.fees_usd.sum():,.2f}")


if __name__ == "__main__":
    main()
