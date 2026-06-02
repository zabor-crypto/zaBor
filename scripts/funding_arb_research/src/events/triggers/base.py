"""Trigger abstract base class.

Triggers are stateless w.r.t. the engine — they receive a snapshot of
the market state at time ``t`` and emit zero or more ``FundingEvent``s.
The engine asks each trigger when it next wants to be checked, so a
trigger that only fires at funding-publication boundaries can opt out
of being polled every tick.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..event import FundingEvent


@dataclass
class MarketSnapshot:
    """What every trigger sees at evaluation time.

    ``funding_panels[(venue, symbol)]`` is the venue's funding history
    panel sliced to ``timestamp_utc < t`` (no look-ahead). The trigger
    is responsible for picking the right rows (predicted vs realized).
    """
    t: pd.Timestamp
    funding_panels: dict[tuple[str, str], pd.DataFrame]
    contracts_meta: pd.DataFrame              # per (venue, symbol): cadence, base/quote, fees
    listing_archive: pd.DataFrame             # for the new-listing trigger


class Trigger(ABC):
    name: str = "base"

    @abstractmethod
    def next_check_at(self, t: pd.Timestamp) -> pd.Timestamp:
        """When the engine should next call ``evaluate``."""

    @abstractmethod
    def evaluate(self, snap: MarketSnapshot) -> list[FundingEvent]:
        """Emit zero or more events given the snapshot at ``snap.t``."""

    def cooldown_until(self, coin: str) -> pd.Timestamp | None:
        """Optional: a coin-specific cooldown. Default: no cooldown."""
        return None
