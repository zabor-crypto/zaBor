"""Canonical event + position records for the event-driven engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd


# What the strategy is harvesting.
TriggerType = Literal[
    "bitget_extreme",        # |predicted_funding| > threshold on Bitget perp
    "new_listing_spike",     # newly listed perp with extreme funding in first hours
    "cross_dex_dispersion",  # large funding gap between two venues
    "synthetic",             # only used for engine smoke tests
]


@dataclass(frozen=True)
class FundingEvent:
    """A trigger firing, with everything the router needs to act.

    ``venue_signal`` is the per-venue snapshot the trigger considered
    when firing (``{venue: predicted_apr}``). The router uses it to pick
    the long and short legs without re-reading market data.
    """
    coin: str
    timestamp_utc: pd.Timestamp
    trigger_type: TriggerType
    venue_signal: dict[str, float]            # {venue: predicted_funding_apr}
    expected_funding_pnl_apr: float           # net APR after rough cost guess
    confidence: float = 1.0                   # 0..1; ranks events when capital is scarce
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if not isinstance(self.timestamp_utc, pd.Timestamp) or self.timestamp_utc.tz is None:
            raise ValueError("timestamp_utc must be tz-aware UTC pd.Timestamp")


@dataclass
class Leg:
    """One side of a delta-neutral pair."""
    venue: str
    symbol: str
    direction: int                            # +1 long, -1 short
    notional_usd: float
    leverage: float
    entry_price: float
    entry_ts_utc: pd.Timestamp
    funding_interval_hours: int
    accrued_funding_usd: float = 0.0
    last_funding_settle_ts: Optional[pd.Timestamp] = None
    fees_paid_usd: float = 0.0
    slippage_paid_usd: float = 0.0
    slippage_source: Literal["recorder", "fallback"] = "fallback"


@dataclass
class Position:
    """A delta-neutral position: long_leg + short_leg, born from one event."""
    event: FundingEvent
    long_leg: Leg
    short_leg: Leg
    opened_ts_utc: pd.Timestamp
    closed_ts_utc: Optional[pd.Timestamp] = None
    close_reason: Optional[str] = None
    realized_pnl_usd: float = 0.0
    peak_equity_usd: float = 0.0              # for per-position drawdown
    trough_equity_usd: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.closed_ts_utc is None

    def gross_notional_usd(self) -> float:
        return self.long_leg.notional_usd + self.short_leg.notional_usd

    def used_capital_usd(self) -> float:
        return (self.long_leg.notional_usd / self.long_leg.leverage
                + self.short_leg.notional_usd / self.short_leg.leverage)
