from decimal import Decimal

from v1.execution.binance_executor import BinanceUmExecutor, ExecutorConfig
from v1.execution.order_state_machine import OrderState, OrderStatus


def test_ack_timeout_marks_rejected() -> None:
    executor = BinanceUmExecutor(ExecutorConfig(paper_mode=True, ack_timeout_ms=100, ttl_ms=200))
    state = OrderState(
        order_intent_id="intent-1",
        symbol="BTCUSDT",
        side="BUY",
        qty=Decimal("1"),
        limit_price=Decimal("100"),
        status=OrderStatus.SUBMITTED,
        submitted_ts_ms=1_000,
    )
    executor.orders[state.order_intent_id] = state

    updates = executor.collect_timeout_updates(now_ts_ms=1_200)
    assert len(updates) == 1
    assert updates[0].status == "REJECTED"
    assert updates[0].detail["reason"] == "ack_timeout"


def test_ttl_timeout_partial_fill_requires_residual_flatten() -> None:
    executor = BinanceUmExecutor(ExecutorConfig(paper_mode=True, ack_timeout_ms=100, ttl_ms=200))
    state = OrderState(
        order_intent_id="intent-2",
        symbol="BTCUSDT",
        side="BUY",
        qty=Decimal("10"),
        limit_price=Decimal("100"),
        status=OrderStatus.PARTIALLY_FILLED,
        submitted_ts_ms=1_000,
        ack_ts_ms=1_050,
        order_id="paper-1",
        filled_qty=Decimal("4"),
        avg_fill_price=Decimal("100"),
    )
    executor.orders[state.order_intent_id] = state

    updates = executor.collect_timeout_updates(now_ts_ms=1_300)
    assert len(updates) == 1
    assert updates[0].status == "CANCELED"
    assert updates[0].detail["residual_flatten_required"] is True
    assert updates[0].detail["residual_qty"] == "6"
    assert updates[0].detail["flatten_side"] == "SELL"


def test_user_stream_trade_updates_order_state() -> None:
    executor = BinanceUmExecutor(ExecutorConfig(paper_mode=False, user_stream_enabled=False))
    state = OrderState(
        order_intent_id="intent-3",
        symbol="BTCUSDT",
        side="BUY",
        qty=Decimal("5"),
        limit_price=Decimal("100"),
        status=OrderStatus.SUBMITTED,
        submitted_ts_ms=1_000,
    )
    executor.orders[state.order_intent_id] = state
    executor.client_order_id_to_intent_id["intent-3"] = "intent-3"

    partial = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_100,
        "o": {
            "c": "intent-3",
            "i": 123,
            "x": "TRADE",
            "X": "PARTIALLY_FILLED",
            "l": "2",
            "L": "100",
            "z": "2",
        },
    }
    partial_update = executor.apply_user_stream_event(partial)
    assert partial_update is not None
    assert partial_update.status == "PARTIALLY_FILLED"

    final_fill = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_200,
        "o": {
            "c": "intent-3",
            "i": 123,
            "x": "TRADE",
            "X": "FILLED",
            "l": "3",
            "L": "100",
            "z": "5",
        },
    }
    final_update = executor.apply_user_stream_event(final_fill)
    assert final_update is not None
    assert final_update.status == "FILLED"
