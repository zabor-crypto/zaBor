import json

from v1.analysis.daily_report_generator import Window, build_report


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")


def test_daily_report_includes_calibration_gate_attribution(tmp_path) -> None:
    root = tmp_path / "telemetry"
    _write_jsonl(
        root / "decision_log.jsonl",
        [
            {
                "decision_id": "d1",
                "episode_id": "ep1",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1000,
                "action": "ENTER_LONG",
                "risk_approved": False,
                "expected_edge_bps": 14.0,
                "risk_reasons": ["paper_calibration_score_below_min", "expected_edge_below_cost"],
            },
            {
                "decision_id": "d2",
                "episode_id": "ep2",
                "symbol": "ETHUSDT",
                "decision_ts_ms": 1005,
                "action": "ENTER_SHORT",
                "risk_approved": True,
                "expected_edge_bps": 25.0,
                "risk_reasons": [],
            },
        ],
    )
    _write_jsonl(
        root / "feature_log.jsonl",
        [
            {"episode_id": "ep1", "symbol": "BTCUSDT", "decision_ts_ms": 1000, "features": {}},
            {"episode_id": "ep2", "symbol": "ETHUSDT", "decision_ts_ms": 1005, "features": {}},
        ],
    )
    _write_jsonl(
        root / "order_lifecycle_log.jsonl",
        [
            {
                "type": "close_order",
                "status": "FILLED",
                "decision_id": "d1",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "exchange_ts_ms": 1010,
                "exit_reason": "hard_stop",
                "gross_pnl_bps": -3.0,
                "entry_price": 100.0,
                "fill_price": 99.7,
                "hold_ms": 1000,
                "mfe_bps": 0.0,
                "mae_bps": -4.0,
            }
        ],
    )
    _write_jsonl(
        root / "risk_health_log.jsonl",
        [
            {
                "event": "paper_calibration_gate_blocked",
                "symbol": "BTCUSDT",
                "decision_id": "d1",
                "exchange_ts_ms": 1001,
                "reasons": ["paper_calibration_score_below_min"],
            }
        ],
    )

    report = build_report(
        telemetry_root=root,
        window=Window(label="w", start_ts_ms=900, end_ts_ms=2000),
    )

    assert report["schema_version"] == "daily_report_v2"
    assert report["inputs"]["risk_rows"] == 1
    cal = report["calibration_gate"]
    assert cal["active_in_window"] is True
    assert cal["risk_event_blocks_total"] == 1
    assert cal["risk_event_unique_blocked_decisions"] == 1
    assert cal["decision_log_blocks_total"] == 1
    assert cal["risk_event_by_symbol"]["BTCUSDT"] == 1
    reasons = dict(cal["decision_log_reason_top10"])
    assert reasons["paper_calibration_score_below_min"] == 1


def test_daily_report_calibration_gate_defaults_when_no_blocks(tmp_path) -> None:
    root = tmp_path / "telemetry"
    _write_jsonl(
        root / "decision_log.jsonl",
        [
            {
                "decision_id": "d1",
                "episode_id": "ep1",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1000,
                "action": "NO_TRADE",
                "risk_approved": False,
                "expected_edge_bps": 0.0,
                "risk_reasons": ["decision_no_trade"],
            }
        ],
    )
    _write_jsonl(root / "feature_log.jsonl", [{"episode_id": "ep1", "symbol": "BTCUSDT", "decision_ts_ms": 1000, "features": {}}])
    _write_jsonl(root / "order_lifecycle_log.jsonl", [])
    _write_jsonl(root / "risk_health_log.jsonl", [{"event": "paper_heartbeat", "exchange_ts_ms": 1200}])

    report = build_report(
        telemetry_root=root,
        window=Window(label="w", start_ts_ms=900, end_ts_ms=2000),
    )
    cal = report["calibration_gate"]
    assert cal["active_in_window"] is False
    assert cal["risk_event_blocks_total"] == 0
    assert cal["decision_log_blocks_total"] == 0


def test_daily_report_includes_calibration_config_drift(tmp_path) -> None:
    project_root = tmp_path / "project"
    root = project_root / "v1_data" / "telemetry"
    _write_jsonl(
        root / "decision_log.jsonl",
        [
            {
                "decision_id": "d1",
                "episode_id": "ep1",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1000,
                "action": "NO_TRADE",
                "risk_approved": False,
                "expected_edge_bps": 0.0,
                "risk_reasons": ["decision_no_trade"],
            }
        ],
    )
    _write_jsonl(root / "feature_log.jsonl", [{"episode_id": "ep1", "symbol": "BTCUSDT", "decision_ts_ms": 1000, "features": {}}])
    _write_jsonl(root / "order_lifecycle_log.jsonl", [])
    _write_jsonl(root / "risk_health_log.jsonl", [{"event": "paper_heartbeat", "exchange_ts_ms": 1200}])

    runtime_overrides = project_root / "analysis" / "phase3_experiments" / "run_a" / "per_asset_calibration_overrides.json"
    runtime_overrides.parent.mkdir(parents=True, exist_ok=True)
    runtime_overrides.write_text(
        json.dumps(
            {
                "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                "source_run_id": "run_a",
                "symbols": {"BTCUSDT": {"entry_filter": {"enabled": True, "min_score": 0.3}}},
            }
        ),
        encoding="utf-8",
    )
    recommended = project_root / "analysis" / "phase3_experiments" / "run_b" / "per_asset_calibration_overrides.json"
    recommended.parent.mkdir(parents=True, exist_ok=True)
    recommended.write_text(
        json.dumps(
            {
                "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                "source_run_id": "run_b",
                "symbols": {"SOLUSDT": {"entry_filter": {"enabled": True, "min_score": 0.3}}},
            }
        ),
        encoding="utf-8",
    )

    (root / "paper_session_report_20260331_000000.json").write_text(
        json.dumps(
            {
                "paper_calibration_gate": {
                    "enabled": True,
                    "source_path": str(runtime_overrides),
                    "schema_version": "phase3_per_asset_entry_filter_overrides_v1",
                    "active_symbols": ["BTCUSDT"],
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        telemetry_root=root,
        window=Window(label="w", start_ts_ms=900, end_ts_ms=2000),
        project_root=project_root,
    )
    drift = report["calibration_config_drift"]
    assert drift["runtime_gate_enabled"] is True
    assert drift["runtime_overrides_path"] == str(runtime_overrides)
    assert drift["recommended_overrides_path"] == str(recommended)
    assert drift["status"] == "mismatch"
    assert drift["hash_match"] is False
