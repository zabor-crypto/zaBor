import json

from v1.telemetry.rollout_gates import RolloutGateThresholds, compute_rollout_metrics, evaluate_rollout_gates


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")


def test_rollout_gate_passes_for_complete_lineage(tmp_path) -> None:
    root = tmp_path / "telemetry"
    _write_jsonl(
        root / "feature_log.jsonl",
        [
            {"episode_id": "ep1", "symbol": "BTCUSDT", "decision_ts_ms": 1000},
            {"episode_id": "ep2", "symbol": "BTCUSDT", "decision_ts_ms": 1100},
        ],
    )
    _write_jsonl(
        root / "decision_log.jsonl",
        [
            {
                "decision_id": "d1",
                "episode_id": "ep1",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1000,
                "action": "ENTER_LONG",
                "risk_approved": True,
                "expected_edge_bps": 15.0,
            },
            {
                "decision_id": "d2",
                "episode_id": "ep2",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1100,
                "action": "NO_TRADE",
                "risk_approved": False,
                "expected_edge_bps": 0.0,
            },
            {
                # intentionally missing feature row: should not affect lineage
                # completeness because no order was emitted for this decision.
                "decision_id": "d3",
                "episode_id": "ep_missing",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 1150,
                "action": "NO_TRADE",
                "risk_approved": False,
                "expected_edge_bps": 0.0,
            }
        ],
    )
    _write_jsonl(
        root / "order_lifecycle_log.jsonl",
        [
            {
                "decision_id": "d1",
                "order_intent_id": "o1",
                "status": "FILLED",
                "symbol": "BTCUSDT",
                "exchange_ts_ms": 1001,
                "detail": {"realized_slippage_bps": 1.25, "state_side": "BUY"},
            }
        ],
    )
    _write_jsonl(root / "risk_health_log.jsonl", [])

    metrics = compute_rollout_metrics(
        telemetry_root=root,
        start_ts_ms=900,
        end_ts_ms=1200,
    )
    assert metrics.decision_count == 3
    assert metrics.fill_count == 1
    assert metrics.lineage_ratio == 1.0
    assert metrics.mean_realized_slippage_bps == 1.25

    result = evaluate_rollout_gates(
        telemetry_root=root,
        start_ts_ms=900,
        end_ts_ms=1200,
        thresholds=RolloutGateThresholds(
            min_decisions=3,
            min_fills=1,
            min_lineage_ratio=0.99,
            max_mean_realized_slippage_bps=2.0,
            max_p95_realized_slippage_bps=2.0,
            max_reject_ratio=0.2,
            max_residual_flatten_events=0,
        ),
    )
    assert result.passed is True
    assert result.failed_gates == []


def test_rollout_gate_fails_on_missing_lineage_and_rejects(tmp_path) -> None:
    root = tmp_path / "telemetry"
    _write_jsonl(root / "feature_log.jsonl", [])
    _write_jsonl(
        root / "decision_log.jsonl",
        [
            {
                "decision_id": "d1",
                "episode_id": "ep_missing",
                "symbol": "BTCUSDT",
                "decision_ts_ms": 2000,
                "action": "ENTER_SHORT",
                "risk_approved": True,
                "expected_edge_bps": 18.0,
            }
        ],
    )
    _write_jsonl(
        root / "order_lifecycle_log.jsonl",
        [
            {
                "decision_id": "d1",
                "order_intent_id": "o1",
                "status": "REJECTED",
                "symbol": "BTCUSDT",
                "exchange_ts_ms": 2001,
                "detail": {"reason": "ack_timeout"},
            }
        ],
    )
    _write_jsonl(
        root / "risk_health_log.jsonl",
        [{"event": "residual_flatten_required", "exchange_ts_ms": 2002}],
    )

    result = evaluate_rollout_gates(
        telemetry_root=root,
        start_ts_ms=1900,
        end_ts_ms=2100,
        thresholds=RolloutGateThresholds(
            min_decisions=1,
            min_fills=1,
            min_lineage_ratio=1.0,
            max_mean_realized_slippage_bps=2.0,
            max_p95_realized_slippage_bps=4.0,
            max_reject_ratio=0.0,
            max_residual_flatten_events=0,
        ),
    )

    assert result.passed is False
    assert "fill_count_below_min" in result.failed_gates
    assert "lineage_ratio_below_min" in result.failed_gates
    assert "reject_ratio_above_max" in result.failed_gates
    assert "residual_flatten_events_above_max" in result.failed_gates
