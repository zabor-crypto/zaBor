"""Canonical schema for normalized data. All venues map into these tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---- Funding panel ------------------------------------------------------- #
FUNDING_COLUMNS: list[str] = [
    "timestamp_utc",
    "venue",
    "symbol",
    "base_asset",
    "quote_asset",
    "instrument_type",
    "funding_rate",
    "funding_interval_hours",
    "predicted_funding_rate",
    "mark_price",
    "index_price",
    "open_interest",
    "long_oi",
    "short_oi",
    "borrow_rate",
    "collateral_yield",
    "source",
    "raw_payload_hash",
]

FUNDING_DTYPES: dict[str, str] = {
    "timestamp_utc": "datetime64[ns, UTC]",
    "venue": "string",
    "symbol": "string",
    "base_asset": "string",
    "quote_asset": "string",
    "instrument_type": "string",
    "funding_rate": "float64",
    "funding_interval_hours": "Int16",
    "predicted_funding_rate": "float64",
    "mark_price": "float64",
    "index_price": "float64",
    "open_interest": "float64",
    "long_oi": "float64",
    "short_oi": "float64",
    "borrow_rate": "float64",
    "collateral_yield": "float64",
    "source": "string",
    "raw_payload_hash": "string",
}


# ---- Price panel --------------------------------------------------------- #
PRICE_COLUMNS: list[str] = [
    "timestamp_utc",
    "venue",
    "symbol",
    "base_asset",
    "quote_asset",
    "instrument_type",
    "mid_price",
    "bid",
    "ask",
    "mark_price",
    "index_price",
    "source",
    "raw_payload_hash",
]

PRICE_DTYPES: dict[str, str] = {
    "timestamp_utc": "datetime64[ns, UTC]",
    "venue": "string",
    "symbol": "string",
    "base_asset": "string",
    "quote_asset": "string",
    "instrument_type": "string",
    "mid_price": "float64",
    "bid": "float64",
    "ask": "float64",
    "mark_price": "float64",
    "index_price": "float64",
    "source": "string",
    "raw_payload_hash": "string",
}


@dataclass(frozen=True)
class SchemaIssue:
    """Recorded mismatch between an API payload and our canonical schema."""
    venue: str
    endpoint: str
    field: str
    expected: str
    observed: str
    sample: Optional[str] = None


def empty_funding_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=FUNDING_DTYPES[c]) for c in FUNDING_COLUMNS})
    return df


def empty_price_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=PRICE_DTYPES[c]) for c in PRICE_COLUMNS})
    return df


def conform_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex/cast a venue-specific funding frame to the canonical schema."""
    for col in FUNDING_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FUNDING_COLUMNS].copy()
    for col, dtype in FUNDING_DTYPES.items():
        try:
            if dtype.startswith("datetime64"):
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Leave the column as-is; the schema-mismatch logger should
            # have already flagged the issue upstream.
            pass
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def conform_price(df: pd.DataFrame) -> pd.DataFrame:
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PRICE_COLUMNS].copy()
    for col, dtype in PRICE_DTYPES.items():
        try:
            if dtype.startswith("datetime64"):
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return df.sort_values("timestamp_utc").reset_index(drop=True)
