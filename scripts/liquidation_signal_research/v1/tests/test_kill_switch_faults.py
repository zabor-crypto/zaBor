from v1.contracts.event_types import DecisionIntentEvent, FeatureSnapshotEvent, RegimeStateEvent
from v1.risk.risk_engine import RiskEngine, SessionRiskState


def _decision() -> DecisionIntentEvent:
    return DecisionIntentEvent(
        event_id="d",
        schema_version="v1",
        episode_id="e",
        decision_id="d",
        symbol="BTCUSDT",
        decision_ts_ms=1000,
        action="ENTER_LONG",
        score=1.0,
        score_components={"x": 1.0},
        expected_edge_bps=20.0,
        slippage_budget_bps=2.0,
        no_trade_reason=None,
    )


def _snapshot() -> FeatureSnapshotEvent:
    return FeatureSnapshotEvent(
        event_id="f",
        schema_version="v1",
        episode_id="e",
        decision_ts_ms=1000,
        symbol="BTCUSDT",
        features={"spread_bps": 1.0},
    )


def _regime() -> RegimeStateEvent:
    return RegimeStateEvent(
        event_id="r",
        schema_version="v1",
        decision_id="d",
        symbol="BTCUSDT",
        decision_ts_ms=1000,
        regime="stress_continuation",
        tradable=True,
        no_trade_reasons=[],
    )


def test_reject_burst_kill_switch_blocks_trading() -> None:
    risk = RiskEngine()
    gate = risk.evaluate(
        decision=_decision(),
        regime=_regime(),
        snapshot=_snapshot(),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(
            nav_usd=10_000.0,
            consecutive_rejects=5,
            telemetry_alive=True,
        ),
    )
    assert gate.approved is False
    assert "kill_switch:reject_cluster" in gate.reasons


def test_telemetry_blindness_kill_switch_blocks_trading() -> None:
    risk = RiskEngine()
    gate = risk.evaluate(
        decision=_decision(),
        regime=_regime(),
        snapshot=_snapshot(),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(
            nav_usd=10_000.0,
            telemetry_alive=False,
        ),
    )
    assert gate.approved is False
    assert "kill_switch:telemetry_blindness" in gate.reasons


def test_drawdown_unit_is_fraction_not_percent() -> None:
    # SessionRiskState.drawdown_pct and KillSwitchThresholds.max_drawdown_pct
    # must share the same unit (fraction, 0.08 = 8%). Past regression: engine.py
    # stored as percent (× 100), causing the documented 8% gate to fire at 0.08%.
    risk = RiskEngine()

    below = risk.evaluate(
        decision=_decision(),
        regime=_regime(),
        snapshot=_snapshot(),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(nav_usd=10_000.0, drawdown_pct=0.05, telemetry_alive=True),
    )
    assert "kill_switch:drawdown_breach" not in below.reasons

    above = risk.evaluate(
        decision=_decision(),
        regime=_regime(),
        snapshot=_snapshot(),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(nav_usd=10_000.0, drawdown_pct=0.09, telemetry_alive=True),
    )
    assert "kill_switch:drawdown_breach" in above.reasons
