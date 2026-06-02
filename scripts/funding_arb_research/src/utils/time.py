"""UTC time helpers. The whole stack uses tz-aware UTC datetimes."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Iterable, Optional

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def to_utc(ts: datetime | pd.Timestamp | int | float | str) -> pd.Timestamp:
    """Coerce arbitrary input into a UTC pandas Timestamp.

    Integer/float inputs are interpreted as ms epoch if they look like ms,
    seconds otherwise. This avoids the very common silent ms/sec mix-up.
    """
    if isinstance(ts, (int, float)):
        # Heuristic: > 1e12 means ms (after ~2001 in ms), otherwise s.
        unit = "ms" if abs(ts) > 1e12 else "s"
        return pd.Timestamp(ts, unit=unit, tz="UTC")
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def to_ms(ts: datetime | pd.Timestamp) -> int:
    return int(to_utc(ts).timestamp() * 1000)


def floor_to_interval(ts: pd.Timestamp, interval_hours: int) -> pd.Timestamp:
    ts = to_utc(ts)
    epoch_hours = int(ts.timestamp() // 3600)
    floored_hours = (epoch_hours // interval_hours) * interval_hours
    return pd.Timestamp(floored_hours * 3600, unit="s", tz="UTC")


def date_range_utc(start: str | datetime, end: str | datetime, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range(start=to_utc(start), end=to_utc(end), freq=freq, tz="UTC")


def chunk_time_range(
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    chunk: timedelta,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    s = to_utc(start)
    e = to_utc(end)
    cursor = s
    while cursor < e:
        nxt = min(cursor + chunk, e)
        yield cursor, nxt
        cursor = nxt


def annualize_funding(funding_rate: float, interval_hours: int) -> float:
    """Funding rate per interval -> APR (simple compounding off)."""
    intervals_per_year = (365 * 24) / interval_hours
    return funding_rate * intervals_per_year
