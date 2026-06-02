from v1.contracts.event_types import DecisionIntentEvent, FeatureSnapshotEvent, RegimeStateEvent
from v1.risk.risk_engine import RiskEngine, RiskLimits, SessionRiskState


def _decision(action: str, edge: float) -> DecisionIntentEvent:
    return DecisionIntentEvent(
        event_id="d",
        schema_version="v1",
        episode_id="e",
        decision_id="d",
        symbol="BTCUSDT",
        decision_ts_ms=1000,
        action=action,
        score=1.0,
        score_components={"x": 1.0},
        expected_edge_bps=edge,
        slippage_budget_bps=2.0,
        no_trade_reason=None,
    )


def _snapshot(spread: float) -> FeatureSnapshotEvent:
    return FeatureSnapshotEvent(
        event_id="f",
        schema_version="v1",
        episode_id="e",
        decision_ts_ms=1000,
        symbol="BTCUSDT",
        features={"spread_bps": spread},
    )


def _regime(tradable: bool) -> RegimeStateEvent:
    return RegimeStateEvent(
        event_id="r",
        schema_version="v1",
        decision_id="d",
        symbol="BTCUSDT",
        decision_ts_ms=1000,
        regime="stress_continuation" if tradable else "blocked",
        tradable=tradable,
        no_trade_reasons=[] if tradable else ["blocked"],
    )


def test_risk_blocks_when_edge_below_cost() -> None:
    risk = RiskEngine()
    gate = risk.evaluate(
        decision=_decision("ENTER_LONG", edge=4.0),
        regime=_regime(True),
        snapshot=_snapshot(1.0),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(nav_usd=10_000.0),
    )
    assert gate.approved is False
    assert "expected_edge_below_cost" in gate.reasons


def test_risk_approves_valid_trade() -> None:
    risk = RiskEngine()
    gate = risk.evaluate(
        decision=_decision("ENTER_SHORT", edge=20.0),
        regime=_regime(True),
        snapshot=_snapshot(1.0),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(nav_usd=10_000.0),
    )
    assert gate.approved is True


def test_risk_blocks_on_extra_reasons() -> None:
    risk = RiskEngine()
    gate = risk.evaluate(
        decision=_decision("ENTER_LONG", edge=20.0),
        regime=_regime(True),
        snapshot=_snapshot(1.0),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=SessionRiskState(nav_usd=10_000.0),
        extra_reasons=["paper_calibration_score_below_min"],
    )
    assert gate.approved is False
    assert "paper_calibration_score_below_min" in gate.reasons


def test_risk_blocks_on_symbol_consecutive_losses_cap() -> None:
    risk = RiskEngine(limits=RiskLimits(max_symbol_consecutive_losses=2))
    state = SessionRiskState(
        nav_usd=10_000.0,
        symbol_consecutive_losses={"BTCUSDT": 2},
    )
    gate = risk.evaluate(
        decision=_decision("ENTER_LONG", edge=20.0),
        regime=_regime(True),
        snapshot=_snapshot(1.0),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=state,
    )
    assert gate.approved is False
    assert "symbol_consecutive_losses_cap" in gate.reasons


def test_risk_blocks_on_session_net_loss_cap() -> None:
    risk = RiskEngine(limits=RiskLimits(max_session_net_loss_fraction=0.02))
    state = SessionRiskState(
        nav_usd=10_000.0,
        session_pnl_usd=-250.0,
    )
    gate = risk.evaluate(
        decision=_decision("ENTER_SHORT", edge=20.0),
        regime=_regime(True),
        snapshot=_snapshot(1.0),
        stream_healthy=True,
        book_healthy=True,
        signal_age_ms=10,
        session=state,
    )
    assert gate.approved is False
    assert "session_net_loss_cap_reached" in gate.reasons
