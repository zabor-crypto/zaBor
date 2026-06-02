"""Tests for PositionManager exit logic, focusing on flow_reversal calibration."""

from v1.execution.position_manager import PositionManager


def _open(pm: PositionManager, *, symbol: str = "BTCUSDT", side: str = "LONG", liq: float = 10_000.0) -> None:
    pm.open_position(
        symbol=symbol,
        side=side,
        entry_price=100.0,
        entry_ts_ms=0,
        qty=0.001,
        decision_id="d1",
        episode_id="ep1",
        score=0.5,
        liq_notional_1s=liq,
    )


def _adverse_features(side: str) -> dict:
    # flow_imbalance_1s adverse for the given side, magnitude > FLOW_REVERSAL_THRESHOLD (0.15)
    return {"flow_imbalance_1s": -0.5 if side == "LONG" else 0.5}


def _favorable_features(side: str) -> dict:
    return {"flow_imbalance_1s": 0.5 if side == "LONG" else -0.5}


def test_flow_reversal_count_is_20() -> None:
    assert PositionManager.FLOW_REVERSAL_COUNT == 20


def test_flow_reversal_does_not_fire_before_count_reached() -> None:
    pm = PositionManager()
    _open(pm, side="LONG")
    features = _adverse_features("LONG")

    for tick in range(PositionManager.FLOW_REVERSAL_COUNT - 1):
        sig = pm.check_exits(
            symbol="BTCUSDT",
            ts_ms=tick * 250 + 250,
            features=features,
            mid_price=100.0,
        )
        assert sig is None or sig.reason != "flow_reversal", (
            f"flow_reversal fired too early at tick {tick + 1}"
        )


def test_flow_reversal_fires_at_count() -> None:
    pm = PositionManager()
    _open(pm, side="LONG")
    features = _adverse_features("LONG")

    sig = None
    for tick in range(PositionManager.FLOW_REVERSAL_COUNT):
        sig = pm.check_exits(
            symbol="BTCUSDT",
            ts_ms=(tick + 1) * 250,
            features=features,
            mid_price=100.0,
        )
    assert sig is not None and sig.reason == "flow_reversal"


def test_flow_reversal_counter_resets_on_favorable_flow() -> None:
    pm = PositionManager()
    _open(pm, side="LONG")

    # Accumulate 15 adverse ticks (not enough to fire)
    for tick in range(15):
        pm.check_exits(
            symbol="BTCUSDT",
            ts_ms=(tick + 1) * 250,
            features=_adverse_features("LONG"),
            mid_price=100.0,
        )
    pos = pm.get_position("BTCUSDT")
    assert pos is not None and pos.flow_reversal_count == 15

    # One favorable tick resets counter to 0
    pm.check_exits(
        symbol="BTCUSDT",
        ts_ms=16 * 250,
        features=_favorable_features("LONG"),
        mid_price=100.0,
    )
    assert pm.get_position("BTCUSDT").flow_reversal_count == 0


def test_flow_reversal_bonus_for_large_cascade() -> None:
    """Large cascade (>= LIQ_WIDE_FLOW_THRESHOLD) adds FLOW_REVERSAL_COUNT_BONUS ticks."""
    pm = PositionManager()
    large_liq = PositionManager.LIQ_WIDE_FLOW_THRESHOLD + 1.0
    _open(pm, side="SHORT", liq=large_liq)
    features = _adverse_features("SHORT")

    expected_count = PositionManager.FLOW_REVERSAL_COUNT + PositionManager.FLOW_REVERSAL_COUNT_BONUS

    # Should not fire until expected_count ticks
    for tick in range(expected_count - 1):
        sig = pm.check_exits(
            symbol="BTCUSDT",
            ts_ms=(tick + 1) * 250,
            features=features,
            mid_price=100.0,
        )
        assert sig is None or sig.reason != "flow_reversal"

    sig = pm.check_exits(
        symbol="BTCUSDT",
        ts_ms=expected_count * 250,
        features=features,
        mid_price=100.0,
    )
    assert sig is not None and sig.reason == "flow_reversal"


def test_price_only_mode_skips_flow_reversal() -> None:
    """price_only=True should not trigger flow_reversal regardless of adverse flow."""
    pm = PositionManager()
    _open(pm, side="LONG")
    features = _adverse_features("LONG")

    for tick in range(PositionManager.FLOW_REVERSAL_COUNT + 5):
        sig = pm.check_exits(
            symbol="BTCUSDT",
            ts_ms=(tick + 1) * 250,
            features=features,
            mid_price=100.0,
            price_only=True,
        )
        assert sig is None or sig.reason != "flow_reversal"
