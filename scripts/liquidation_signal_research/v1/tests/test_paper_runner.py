import json

import pytest

from v1.app.paper_runner import run_paper_session
from v1.app.runtime import RuntimeConfig
from v1.telemetry.rollout_gates import RolloutGateThresholds


class _FakeEngine:
    def __init__(self, config):
        self.config = config
        self.decisions = [{"a": 1}]

        class _Telemetry:
            def __init__(self):
                self.logged = []

            def log_risk(self, payload):
                self.logged.append(payload)

        self.telemetry = _Telemetry()

        class _Supervisor:
            @staticmethod
            def health_snapshot():
                class H:
                    healthy = True
                    unhealthy_reasons = []

                return H()

        self.supervisor = _Supervisor()

        class _Metrics:
            @staticmethod
            def snapshot():
                return {"counters": {"raw_events_total": 10.0}, "gauges": {}}

        self.metrics = _Metrics()

    async def start_live(self):
        return None

    async def stop_live(self):
        return None

    @staticmethod
    def paper_calibration_gate_summary():
        return {"enabled": False, "source_path": None, "schema_version": None, "active_symbols": []}


class _FakeRollout:
    def to_dict(self):
        return {
            "passed": True,
            "failed_gates": [],
            "metrics": {
                "decision_count": 1,
                "lineage_ratio": 1.0,
                "mean_realized_slippage_bps": 0.0,
                "p95_realized_slippage_bps": 0.0,
                "fill_count": 1,
                "reject_ratio": 0.0,
                "residual_flatten_events": 0,
                "enter_decision_count": 1,
                "order_intent_count": 1,
                "reject_count": 0,
                "lineage_complete_decisions": 1,
                "mean_expected_edge_bps": 12.0,
                "mean_edge_after_slippage_bps": 12.0,
            },
            "thresholds": {
                "min_decisions": 1,
                "min_fills": 1,
                "min_lineage_ratio": 0.9,
                "max_mean_realized_slippage_bps": 2.0,
                "max_p95_realized_slippage_bps": 3.0,
                "max_reject_ratio": 0.2,
                "max_residual_flatten_events": 0,
            },
        }


@pytest.mark.asyncio
async def test_paper_runner_writes_report(tmp_path, monkeypatch) -> None:
    from v1.app import paper_runner as pr

    monkeypatch.setattr(pr, "ContinuationEngine", _FakeEngine)
    monkeypatch.setattr(pr, "evaluate_rollout_gates", lambda **kwargs: _FakeRollout())
    monkeypatch.setattr(
        pr,
        "run_parity",
        lambda **kwargs: {
            "replay_run_id": "paper_1",
            "parity": {"match_ratio": 1.0, "mismatch_count": 0},
        },
    )

    report_path = tmp_path / "out" / "paper_report.json"
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        shadow_mode=False,
    )

    out = await run_paper_session(
        config=cfg,
        duration_seconds=1,
        heartbeat_seconds=1,
        report_path=report_path,
        rollout_thresholds=RolloutGateThresholds(min_decisions=1, min_fills=1),
        run_parity_check=True,
    )

    assert out["mode"] == "paper"
    assert out["rollout_gates"]["passed"] is True
    assert out["parity"]["replay_run_id"] == "paper_1"
    assert out["paper_calibration_gate"]["enabled"] is False
    assert report_path.exists()

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "paper"
    assert saved["rollout_gates"]["passed"] is True
