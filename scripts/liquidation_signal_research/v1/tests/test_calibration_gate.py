import json

import pytest

from v1.contracts.event_types import DecisionIntentEvent
from v1.risk.calibration_gate import PerAssetCalibrationGate


def _decision(*, symbol: str = "BTCUSDT", action: str = "ENTER_LONG", score: float = 0.4, edge: float = 14.0) -> DecisionIntentEvent:
    return DecisionIntentEvent(
        event_id="d1",
        schema_version="v1",
        episode_id="ep1",
        decision_id="d1",
        symbol=symbol,
        decision_ts_ms=1000,
        action=action,  # type: ignore[arg-type]
        score=score,
        score_components={"x": 1.0},
        expected_edge_bps=edge,
        slippage_budget_bps=3.0,
        no_trade_reason=None,
    )


def test_calibration_gate_blocks_when_thresholds_not_met(tmp_path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                "symbols": {
                    "BTCUSDT": {
                        "entry_filter": {
                            "enabled": True,
                            "min_score": 0.5,
                            "min_expected_edge_bps": 15.0,
                            "min_liq_notional_1s": 100_000.0,
                            "min_depth_collapse_ratio": 0.12,
                            "min_abs_flow_imbalance_1s": 0.2,
                            "max_spread_bps": 4.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gate = PerAssetCalibrationGate.from_file(path)
    reasons = gate.evaluate(
        decision=_decision(score=0.4, edge=14.0),
        features={
            "liq_total_notional_1s": 80_000.0,
            "depth_collapse_ratio": 0.10,
            "flow_imbalance_1s": 0.15,
            "spread_bps": 4.5,
        },
    )
    assert "paper_calibration_score_below_min" in reasons
    assert "paper_calibration_expected_edge_below_min" in reasons
    assert "paper_calibration_liq_below_min" in reasons
    assert "paper_calibration_depth_below_min" in reasons
    assert "paper_calibration_flow_abs_below_min" in reasons
    assert "paper_calibration_spread_above_max" in reasons


def test_calibration_gate_skips_when_symbol_not_configured(tmp_path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                "symbols": {
                    "ETHUSDT": {
                        "entry_filter": {
                            "enabled": True,
                            "min_score": 0.45,
                            "min_expected_edge_bps": 12.0,
                            "min_liq_notional_1s": 20_000.0,
                            "min_depth_collapse_ratio": 0.05,
                            "min_abs_flow_imbalance_1s": 0.08,
                            "max_spread_bps": 6.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gate = PerAssetCalibrationGate.from_file(path)
    reasons = gate.evaluate(
        decision=_decision(symbol="BTCUSDT", score=0.6, edge=20.0),
        features={
            "liq_total_notional_1s": 200_000.0,
            "depth_collapse_ratio": 0.20,
            "flow_imbalance_1s": 0.25,
            "spread_bps": 2.0,
        },
    )
    assert reasons == []


def test_calibration_gate_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "unknown_schema_v999",
                "symbols": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PerAssetCalibrationGate.from_file(path)


def test_calibration_gate_rejects_invalid_thresholds(tmp_path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                "symbols": {
                    "BTCUSDT": {
                        "entry_filter": {
                            "enabled": True,
                            "min_score": -0.1,
                            "min_expected_edge_bps": 12.0,
                            "min_liq_notional_1s": 20_000.0,
                            "min_depth_collapse_ratio": 0.05,
                            "min_abs_flow_imbalance_1s": 0.08,
                            "max_spread_bps": 6.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PerAssetCalibrationGate.from_file(path)
