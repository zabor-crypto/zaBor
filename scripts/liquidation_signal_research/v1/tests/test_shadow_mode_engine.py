from decimal import Decimal

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.book.snapshot_sync import OrderBookSnapshot
from v1.contracts.event_types import (
    DecisionIntentEvent,
    FeatureSnapshotEvent,
    RawMarketEvent,
    RegimeStateEvent,
    RiskGateEvent,
)


def _raw() -> RawMarketEvent:
    return RawMarketEvent(
        event_id="e",
        schema_version="v1",
        venue="binance_um_futures",
        stream="aggTrade",
        symbol="BTCUSDT",
        conn_id="c",
        exchange_ts_ms=2000,
        recv_wall_ts_ns=1,
        recv_mono_ts_ns=1,
        ingest_seq=1,
        stream_local_seq=1,
        payload_json='{"stream":"btcusdt@aggTrade","data":{"a":1,"p":"100","q":"1","m":false,"T":2000}}',
        payload_hash="h",
        parse_status="ok",
        gap_flag=False,
    )


class _RegimeAlways:
    def evaluate(self, *args, **kwargs):
        return RegimeStateEvent(
            event_id="r",
            schema_version="v1",
            decision_id="d",
            symbol="BTCUSDT",
            decision_ts_ms=2000,
            regime="stress_continuation",
            tradable=True,
            no_trade_reasons=[],
        )


class _SignalEnter:
    def evaluate(self, *args, **kwargs):
        return DecisionIntentEvent(
            event_id="d",
            schema_version="v1",
            episode_id="e",
            decision_id="d",
            symbol="BTCUSDT",
            decision_ts_ms=2000,
            action="ENTER_LONG",
            score=0.9,
            score_components={"x": 1.0},
            expected_edge_bps=20.0,
            slippage_budget_bps=2.0,
            no_trade_reason=None,
        )


class _RiskApprove:
    def evaluate(self, *args, **kwargs):
        return RiskGateEvent(
            event_id="g",
            schema_version="v1",
            decision_id="d",
            symbol="BTCUSDT",
            decision_ts_ms=2000,
            approved=True,
            reasons=[],
        )


class _FeatureStub:
    def on_agg_trade(self, event):
        return None

    def on_force_order(self, event):
        return None

    def on_mark_price(self, event):
        return None

    def on_order_book(self, symbol, ts_ms, book):
        return None

    def build_snapshot(self, symbol, decision_ts_ms):
        return FeatureSnapshotEvent(
            event_id="f",
            schema_version="v1",
            episode_id="e",
            decision_ts_ms=decision_ts_ms,
            symbol=symbol,
            features={"spread_bps": 1.0},
        )


def test_shadow_mode_blocks_execution_even_when_trade_is_approved(tmp_path) -> None:
    cfg = RuntimeConfig(
        symbols=["BTCUSDT"],
        streams=["forceOrder", "aggTrade", "depth", "markPrice"],
        data_root=tmp_path / "raw",
        telemetry_root=tmp_path / "telemetry",
        decision_interval_ms=1,
        shadow_mode=True,
    )
    engine = ContinuationEngine(cfg)

    engine.features = _FeatureStub()
    engine.regime_gate = _RegimeAlways()
    engine.signal_engine = _SignalEnter()
    engine.risk_engine = _RiskApprove()

    class _Health:
        healthy = True

    engine.supervisor.health_snapshot = lambda: _Health()

    engine.book.apply_snapshot(
        OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=100,
            bids=[(Decimal("100.0"), Decimal("2.0"))],
            asks=[(Decimal("100.1"), Decimal("2.0"))],
            exchange_ts_ms=1_000,
        )
    )
    engine._sync_pending["BTCUSDT"] = False

    async def _fail_submit(*args, **kwargs):
        raise AssertionError("submit_order should not be called in shadow mode")

    engine.executor.submit_order = _fail_submit

    decision = engine._process_raw_event_common(_raw(), assume_stream_healthy=False)
    assert decision is not None
    assert decision["risk_approved"] is True
    assert decision.get("shadow_blocked_execution") is True
