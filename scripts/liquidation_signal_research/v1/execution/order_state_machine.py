"""Order lifecycle state machine for aggressive-limit execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional


class OrderStatus(str, Enum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    ACKED = "ACKED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderState:
    order_intent_id: str
    symbol: str
    side: str
    qty: Decimal
    limit_price: Decimal
    status: OrderStatus = OrderStatus.NEW
    order_id: Optional[str] = None
    submitted_ts_ms: int = 0
    ack_ts_ms: int = 0
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    terminal_reason: str = ""
    updates: Dict[str, object] = field(default_factory=dict)


class OrderStateMachine:
    def __init__(self, ack_timeout_ms: int = 1200, ttl_ms: int = 800) -> None:
        self.ack_timeout_ms = int(ack_timeout_ms)
        self.ttl_ms = int(ttl_ms)

    def on_submit(self, state: OrderState, ts_ms: int) -> OrderState:
        state.status = OrderStatus.SUBMITTED
        state.submitted_ts_ms = int(ts_ms)
        return state

    def on_ack(self, state: OrderState, *, order_id: str, ts_ms: int) -> OrderState:
        state.status = OrderStatus.ACKED
        state.order_id = order_id
        state.ack_ts_ms = int(ts_ms)
        return state

    def on_fill(self, state: OrderState, *, fill_qty: Decimal, fill_price: Decimal) -> OrderState:
        filled_total = state.filled_qty + fill_qty
        if filled_total > 0:
            weighted = (state.avg_fill_price * state.filled_qty) + (fill_price * fill_qty)
            state.avg_fill_price = weighted / filled_total
        state.filled_qty = filled_total

        if state.filled_qty >= state.qty:
            state.status = OrderStatus.FILLED
        else:
            state.status = OrderStatus.PARTIALLY_FILLED
        return state

    def on_cancel(self, state: OrderState, reason: str) -> OrderState:
        state.status = OrderStatus.CANCELED
        state.terminal_reason = reason
        return state

    def on_reject(self, state: OrderState, reason: str) -> OrderState:
        state.status = OrderStatus.REJECTED
        state.terminal_reason = reason
        return state

    def on_expire(self, state: OrderState) -> OrderState:
        state.status = OrderStatus.EXPIRED
        state.terminal_reason = "ttl_expired"
        return state

    def should_cancel_for_timeout(self, state: OrderState, now_ts_ms: int) -> bool:
        now = int(now_ts_ms)
        if state.status == OrderStatus.SUBMITTED:
            return (now - state.submitted_ts_ms) > self.ack_timeout_ms
        if state.status in (OrderStatus.ACKED, OrderStatus.PARTIALLY_FILLED):
            base = state.ack_ts_ms if state.ack_ts_ms > 0 else state.submitted_ts_ms
            return (now - base) > self.ttl_ms
        return False

    @staticmethod
    def is_terminal(state: OrderState) -> bool:
        return state.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
