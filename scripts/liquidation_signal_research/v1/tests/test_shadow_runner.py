import asyncio
import json
from pathlib import Path

import pytest

from v1.app.runtime import RuntimeConfig
from v1.app.shadow_runner import run_shadow_session


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

    def shadow_summary(self):
        return {
            "total_decisions": 1,
            "action_counts": {"NO_TRADE": 1},
            "no_trade_reason_counts": {"x": 1},
            "risk_reason_counts": {},
            "score_bucket_counts": {"[0.00,0.15)": 1},
            "desync_events": 0,
            "resync_attempts": 0,
            "resync_success": 0,
            "resync_failures": 0,
        }


@pytest.mark.asyncio
async def test_shadow_runner_writes_report(tmp_path, monkeypatch) -> None:
    from v1.app import shadow_runner as sr

    monkeypatch.setattr(sr, "ContinuationEngine", _FakeEngine)

    report_path = tmp_path / "out" / "shadow_report.json"
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        shadow_mode=True,
    )

    out = await run_shadow_session(
        config=cfg,
        duration_seconds=1,
        heartbeat_seconds=1,
        report_path=report_path,
    )

    assert out["mode"] == "shadow"
    assert out["shadow_summary"]["total_decisions"] == 1
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "shadow"
