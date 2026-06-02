"""Slippage estimator backed by recorder book calibration.

Calibration approach: ``scripts/calibrate_recorder_slippage.py`` samples book
snapshots across all available recorder days and computes mean slippage at a
notional grid. This calibrated curve is applied to all backtest trades
regardless of timestamp — using 7 days of real microstructure data as a proxy
for typical execution costs throughout the 540-day backtest window.

Venues with recorder data: bitget, binance.
Other venues fall through to fallback_bps without error.

Notes:
- Some coins (HYPE, PENGU, TAO) have <15 snapshots; calibration is
  statistically thin. Their values are used as-is and flagged in logs.
- Several small-cap coins (AAVE, TRUMP, PENGU) measured ABOVE the 5bp fallback.
  The engine will use the recorder estimate (which is honest) even if > fallback.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

_LOG = get_logger("funding_arb.backtest.recorder_slippage")
_MIN_SNAPSHOTS_WARN = 15
_DEFAULT_FALLBACK_BPS = 5.0


class RecorderSlippage:
    """Interpolate slippage from a pre-computed calibration table.

    Parameters
    ----------
    calibration_path:
        CSV produced by ``scripts/calibrate_recorder_slippage.py``.
    fallback_bps:
        Returned when no calibration data exists for a (venue, coin) pair.
    """

    def __init__(
        self,
        calibration_path: Path,
        fallback_bps: float = _DEFAULT_FALLBACK_BPS,
    ) -> None:
        self._fallback = fallback_bps
        self._curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._available: set[tuple[str, str]] = set()
        self._load(Path(calibration_path))

    # ------------------------------------------------------------------ #

    def _load(self, path: Path) -> None:
        if not path.exists():
            _LOG.warning("calibration file not found: %s — all trades use fallback", path)
            return
        df = pd.read_csv(path)
        required = {"venue", "coin", "notional_usd", "mean_slippage_bps", "n_snapshots"}
        if not required.issubset(df.columns):
            raise ValueError(f"calibration CSV missing columns {required - set(df.columns)}")

        thin: list[str] = []
        for (venue, coin), grp in df.groupby(["venue", "coin"]):
            grp = grp.sort_values("notional_usd")
            notionals = grp["notional_usd"].to_numpy(dtype=float)
            bps = grp["mean_slippage_bps"].to_numpy(dtype=float)
            self._curves[(venue, coin)] = (notionals, bps)
            self._available.add((venue, coin))
            min_n = int(grp["n_snapshots"].min())
            if min_n < _MIN_SNAPSHOTS_WARN:
                thin.append(f"{venue}/{coin}(n={min_n})")

        if thin:
            _LOG.warning(
                "thin calibration (<15 snapshots) for: %s — estimates less reliable",
                ", ".join(thin),
            )
        _LOG.info(
            "RecorderSlippage loaded: %d (venue,coin) pairs from %s",
            len(self._available), path,
        )

    # ------------------------------------------------------------------ #

    def get_slippage_bps(
        self,
        venue: str,
        coin: str,
        notional_usd: float,
    ) -> tuple[float, str]:
        """Return ``(slippage_bps, source)`` where source ∈ {"recorder","fallback"}.

        Uses mean slippage from the calibration table.  Falls back to
        ``fallback_bps`` when the (venue, coin) pair has no calibration data.
        """
        key = (venue, coin)
        if key not in self._available:
            return self._fallback, "fallback"
        notionals, bps_arr = self._curves[key]
        slippage = float(np.interp(notional_usd, notionals, bps_arr))
        slippage = max(0.0, slippage)
        return slippage, "recorder"

    def coverage_summary(self) -> dict:
        """Return dict summarising which venues/coins have calibration data."""
        venues: dict[str, list[str]] = {}
        for v, c in sorted(self._available):
            venues.setdefault(v, []).append(c)
        return venues

    @classmethod
    def from_project_root(
        cls,
        project_root: Optional[Path] = None,
        fallback_bps: float = _DEFAULT_FALLBACK_BPS,
    ) -> "RecorderSlippage":
        from ..utils.io import project_root as _project_root
        root = project_root or _project_root()
        return cls(
            calibration_path=root / "data" / "static" / "recorder_slippage_calibration.csv",
            fallback_bps=fallback_bps,
        )
