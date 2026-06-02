"""Bitget MMR (maintenance-margin rate) model.

Loads the per-symbol tier ladder Bitget publishes on
``/api/v2/mix/market/query-position-lever`` (cached to
``data/static/bitget_mmr.parquet`` by ``mmr_archive.py``) and exposes
two operations the engine needs:

  * ``mmr_for(symbol, notional_usdt)`` — the maintenance-margin rate the
                                          venue would apply at that
                                          notional, used to compute liq
                                          price for the leg.
  * ``max_leverage(symbol, notional_usdt)`` — Bitget's tier-leverage cap;
                                              the engine clips the
                                              strategy's requested
                                              leverage to this.

Resolution order, per the v1 working contract:

  1. Cached API tier table (``data/static/bitget_mmr.parquet``) — primary.
  2. ``config/bitget_mmr_fallback.yaml`` ``generic_altcoin`` table — only
     used when the cache is missing the symbol entirely. The fallback is
     deliberately conservative (overstates MMR) so backtests err pessimistic.

Schema mismatches (e.g. tier rows with NaN ``mmr``) are logged via the
honest-engineering protocol; we never silently default a tier value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import load_yaml, project_root
from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.margin.bitget_mmr")
_DEFAULT_TIERS_PARQUET = project_root() / "data" / "static" / "bitget_mmr.parquet"
_DEFAULT_FALLBACK_YAML = project_root() / "config" / "bitget_mmr_fallback.yaml"


@dataclass(frozen=True)
class Tier:
    level: int
    start_unit: float       # position notional lower bound (USDT)
    end_unit: float         # upper bound (USDT)
    max_leverage: float
    mmr: float              # decimal (0.004 = 0.4%)


class BitgetMMR:
    def __init__(
        self,
        tiers_parquet: Path | None = None,
        fallback_yaml: Path | None = None,
    ) -> None:
        self._tiers_by_symbol: dict[str, list[Tier]] = {}
        self._fallback: list[Tier] = []
        self._fallback_used: set[str] = set()
        self._load_api_cache(tiers_parquet or _DEFAULT_TIERS_PARQUET)
        self._load_fallback(fallback_yaml or _DEFAULT_FALLBACK_YAML)

    def _load_api_cache(self, path: Path) -> None:
        if not path.exists():
            _LOG.warning("bitget MMR cache missing at %s — fallback YAML will be used for every symbol", path)
            return
        df = pd.read_parquet(path)
        required = {"symbol", "level", "start_unit", "end_unit", "max_leverage", "mmr"}
        missing = required - set(df.columns)
        if missing:
            _LOG.error("bitget MMR cache schema mismatch: missing %s; ignoring cache", missing)
            return
        bad = df[df["mmr"].isna() | df["max_leverage"].isna()]
        if not bad.empty:
            _LOG.warning("bitget MMR cache has %d rows with NaN mmr/max_leverage — these tiers will be skipped",
                         len(bad))
        for sym, grp in df.dropna(subset=["mmr", "max_leverage"]).groupby("symbol"):
            ladder = [
                Tier(int(r["level"]), float(r["start_unit"]), float(r["end_unit"]),
                     float(r["max_leverage"]), float(r["mmr"]))
                for _, r in grp.sort_values("level").iterrows()
            ]
            self._tiers_by_symbol[sym] = ladder
        _LOG.info("loaded MMR ladders for %d Bitget symbols from %s",
                  len(self._tiers_by_symbol), path)

    def _load_fallback(self, path: Path) -> None:
        if not path.exists():
            _LOG.warning("MMR fallback YAML missing at %s", path)
            return
        cfg = load_yaml(path)
        tiers_cfg = (cfg.get("generic_altcoin") or {}).get("tiers") or []
        self._fallback = [
            Tier(int(t["level"]), float(t["start_unit"]), float(t["end_unit"]),
                 float(t["max_leverage"]), float(t["mmr"]))
            for t in tiers_cfg
        ]
        if not self._fallback:
            _LOG.warning("MMR fallback YAML at %s has no usable tiers", path)

    def _ladder(self, symbol: str) -> list[Tier]:
        ladder = self._tiers_by_symbol.get(symbol)
        if ladder:
            return ladder
        if symbol not in self._fallback_used:
            _LOG.warning("bitget MMR: no API ladder for %s — using fallback YAML (conservative)",
                         symbol)
            self._fallback_used.add(symbol)
        return self._fallback

    def _tier_for(self, symbol: str, notional_usdt: float) -> Optional[Tier]:
        ladder = self._ladder(symbol)
        if not ladder:
            return None
        for t in ladder:
            # Bitget tiers: notional in [start_unit, end_unit). Top tier's end is open.
            if notional_usdt >= t.start_unit and (t.end_unit <= 0 or notional_usdt < t.end_unit):
                return t
        return ladder[-1]  # above top tier — apply highest MMR

    def mmr_for(self, symbol: str, notional_usdt: float) -> float:
        t = self._tier_for(symbol, notional_usdt)
        if t is None:
            raise ValueError(f"no MMR tier resolved for {symbol} @ {notional_usdt}")
        return t.mmr

    def max_leverage(self, symbol: str, notional_usdt: float) -> float:
        t = self._tier_for(symbol, notional_usdt)
        if t is None:
            raise ValueError(f"no leverage tier resolved for {symbol} @ {notional_usdt}")
        return t.max_leverage

    def has_api_tiers(self, symbol: str) -> bool:
        return symbol in self._tiers_by_symbol
