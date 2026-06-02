"""Per-asset calibration gate for paper-mode pre-trade filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from v1.contracts.event_types import DecisionIntentEvent

EXPECTED_OVERRIDES_SCHEMA_VERSION = "phase3_per_asset_entry_filter_overrides_v1"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AssetEntryFilter:
    enabled: bool
    min_score: float
    min_expected_edge_bps: float
    min_liq_notional_1s: float
    min_depth_collapse_ratio: float
    min_abs_flow_imbalance_1s: float
    max_spread_bps: float


class PerAssetCalibrationGate:
    def __init__(
        self,
        *,
        schema_version: str,
        source_path: Path,
        symbol_filters: Dict[str, AssetEntryFilter],
    ) -> None:
        self.schema_version = schema_version
        self.source_path = Path(source_path)
        self.symbol_filters = symbol_filters

    @classmethod
    def from_file(cls, path: Path) -> "PerAssetCalibrationGate":
        src = Path(path)
        data = json.loads(src.read_text(encoding="utf-8"))
        schema_version = str(data.get("schema_version") or "unknown")
        if schema_version != EXPECTED_OVERRIDES_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported calibration overrides schema_version={schema_version}; "
                f"expected={EXPECTED_OVERRIDES_SCHEMA_VERSION}"
            )
        symbols = data.get("symbols")
        if not isinstance(symbols, dict):
            raise ValueError("Invalid calibration overrides: symbols map missing")

        out: Dict[str, AssetEntryFilter] = {}
        for raw_symbol, payload in symbols.items():
            symbol = str(raw_symbol or "").upper()
            if not symbol:
                continue
            if not isinstance(payload, dict):
                continue
            entry = payload.get("entry_filter")
            if not isinstance(entry, dict):
                continue
            filt = AssetEntryFilter(
                enabled=bool(entry.get("enabled", False)),
                min_score=_float(entry.get("min_score"), 0.0),
                min_expected_edge_bps=_float(entry.get("min_expected_edge_bps"), 0.0),
                min_liq_notional_1s=_float(entry.get("min_liq_notional_1s"), 0.0),
                min_depth_collapse_ratio=_float(entry.get("min_depth_collapse_ratio"), 0.0),
                min_abs_flow_imbalance_1s=_float(entry.get("min_abs_flow_imbalance_1s"), 0.0),
                max_spread_bps=_float(entry.get("max_spread_bps"), 9999.0),
            )
            if filt.min_score < 0.0:
                raise ValueError(f"Invalid min_score for {symbol}: {filt.min_score}")
            if filt.min_expected_edge_bps < 0.0:
                raise ValueError(
                    f"Invalid min_expected_edge_bps for {symbol}: {filt.min_expected_edge_bps}"
                )
            if filt.min_liq_notional_1s < 0.0:
                raise ValueError(
                    f"Invalid min_liq_notional_1s for {symbol}: {filt.min_liq_notional_1s}"
                )
            if filt.min_depth_collapse_ratio < 0.0:
                raise ValueError(
                    f"Invalid min_depth_collapse_ratio for {symbol}: {filt.min_depth_collapse_ratio}"
                )
            if filt.min_abs_flow_imbalance_1s < 0.0:
                raise ValueError(
                    f"Invalid min_abs_flow_imbalance_1s for {symbol}: {filt.min_abs_flow_imbalance_1s}"
                )
            if filt.max_spread_bps <= 0.0:
                raise ValueError(f"Invalid max_spread_bps for {symbol}: {filt.max_spread_bps}")
            if filt.enabled:
                out[symbol] = filt

        return cls(
            schema_version=schema_version,
            source_path=src,
            symbol_filters=out,
        )

    def active_symbols(self) -> List[str]:
        return sorted(self.symbol_filters.keys())

    def evaluate(
        self,
        *,
        decision: DecisionIntentEvent,
        features: Mapping[str, object],
    ) -> List[str]:
        if decision.action not in {"ENTER_LONG", "ENTER_SHORT"}:
            return []

        filt = self.symbol_filters.get(str(decision.symbol).upper())
        if filt is None or not filt.enabled:
            return []

        reasons: List[str] = []

        score = _float(decision.score, 0.0)
        if score < filt.min_score:
            reasons.append("paper_calibration_score_below_min")

        expected_edge = _float(decision.expected_edge_bps, 0.0)
        if expected_edge < filt.min_expected_edge_bps:
            reasons.append("paper_calibration_expected_edge_below_min")

        liq_total = _float(features.get("liq_total_notional_1s"), 0.0)
        if liq_total < filt.min_liq_notional_1s:
            reasons.append("paper_calibration_liq_below_min")

        depth = _float(features.get("depth_collapse_ratio"), 0.0)
        if depth < filt.min_depth_collapse_ratio:
            reasons.append("paper_calibration_depth_below_min")

        flow_abs = abs(_float(features.get("flow_imbalance_1s"), 0.0))
        if flow_abs < filt.min_abs_flow_imbalance_1s:
            reasons.append("paper_calibration_flow_abs_below_min")

        spread = _float(features.get("spread_bps"), 9999.0)
        if spread > filt.max_spread_bps:
            reasons.append("paper_calibration_spread_above_max")

        return reasons
