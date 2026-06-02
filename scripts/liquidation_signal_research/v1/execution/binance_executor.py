"""Binance UM futures aggressive-limit executor with reconciliation hooks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from v1.contracts.event_types import (
    DecisionIntentEvent,
    OrderIntentEvent,
    OrderUpdateEvent,
    SCHEMA_VERSION_V1,
)
from v1.contracts.ids import EVENT_NAMESPACE, make_order_intent_id, make_stable_id
from v1._fapi import next_base, safe_urlopen
from v1.execution.order_state_machine import OrderState, OrderStateMachine, OrderStatus
from v1.execution.slippage_guard import SlippageGuard
from v1.execution.user_data_stream import BinanceUserDataStream


@dataclass(frozen=True)
class ExecutorConfig:
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://fapi.binance.com"
    user_stream_ws_base_url: str = "wss://fstream.binance.com/ws"
    recv_window_ms: int = 5000
    ttl_ms: int = 800
    ack_timeout_ms: int = 1200
    max_replace_count: int = 2
    paper_mode: bool = True
    user_stream_enabled: bool = True
    user_stream_keepalive_seconds: int = 1800


class BinanceUmExecutor:
    def __init__(
        self,
        config: ExecutorConfig | None = None,
        *,
        user_stream: BinanceUserDataStream | None = None,
    ) -> None:
        self.config = config or ExecutorConfig()
        self.state_machine = OrderStateMachine(
            ack_timeout_ms=self.config.ack_timeout_ms,
            ttl_ms=self.config.ttl_ms,
        )
        self.slippage_guard = SlippageGuard()
        self.orders: Dict[str, OrderState] = {}
        self.order_id_to_intent_id: Dict[str, str] = {}
        self.client_order_id_to_intent_id: Dict[str, str] = {}
        self._pending_stream_updates: List[OrderUpdateEvent] = []

        self._user_stream = user_stream
        if self._user_stream is None and not self.config.paper_mode and self.config.user_stream_enabled:
            self._user_stream = BinanceUserDataStream(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                ws_base_url=self.config.user_stream_ws_base_url,
                keepalive_interval_seconds=self.config.user_stream_keepalive_seconds,
            )

    @staticmethod
    def _residual_flatten_fields(state: OrderState) -> Dict[str, object]:
        residual_qty = state.qty - state.filled_qty
        flatten_required = residual_qty > Decimal("0") and state.filled_qty > Decimal("0")
        if not flatten_required:
            return {
                "residual_flatten_required": False,
                "residual_qty": "0",
                "flatten_side": None,
            }
        flatten_side = "SELL" if state.side == "BUY" else "BUY"
        return {
            "residual_flatten_required": True,
            "residual_qty": str(residual_qty),
            "flatten_side": flatten_side,
        }

    @staticmethod
    def _realized_slippage_bps(*, side: str, arrival_price: Decimal, fill_price: Decimal) -> float:
        if arrival_price <= Decimal("0"):
            return 0.0
        if side == "BUY":
            return float(((fill_price - arrival_price) / arrival_price) * Decimal("10000"))
        return float(((arrival_price - fill_price) / arrival_price) * Decimal("10000"))

    def build_order_intent(
        self,
        *,
        decision: DecisionIntentEvent,
        best_bid: Decimal,
        best_ask: Decimal,
        qty: Decimal,
    ) -> OrderIntentEvent:
        side = "BUY" if decision.action == "ENTER_LONG" else "SELL"
        if side == "BUY":
            limit_price = best_ask * (Decimal("1") + Decimal(str(decision.slippage_budget_bps / 10000.0)))
        else:
            limit_price = best_bid * (Decimal("1") - Decimal(str(decision.slippage_budget_bps / 10000.0)))

        order_intent_id = make_order_intent_id(
            decision_id=decision.decision_id,
            side=side,
            qty=str(qty),
            limit_price=str(limit_price),
        )
        return OrderIntentEvent(
            event_id=order_intent_id,
            schema_version=SCHEMA_VERSION_V1,
            decision_id=decision.decision_id,
            order_intent_id=order_intent_id,
            symbol=decision.symbol,
            side=side,
            qty=qty,
            limit_price=limit_price,
            ttl_ms=self.config.ttl_ms,
            max_slippage_bps=float(decision.slippage_budget_bps),
        )

    def _build_signed_query(self, params: Dict[str, object]) -> str:
        encoded = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            encoded.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{encoded}&signature={signature}"

    def _blocking_request(self, method: str, path: str, params: Dict[str, object]) -> Dict[str, object]:
        if self.config.paper_mode:
            return {}

        query = self._build_signed_query(params)
        url = f"{next_base()}{path}?{query}"
        req = urllib.request.Request(url=url, method=method)
        req.add_header("X-MBX-APIKEY", self.config.api_key)
        with safe_urlopen(req, 10) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    async def start(self) -> None:
        if self._user_stream is not None:
            await self._user_stream.start(self._on_user_stream_payload)

    async def stop(self) -> None:
        if self._user_stream is not None:
            await self._user_stream.stop()

    async def _on_user_stream_payload(self, payload: Dict[str, object]) -> None:
        update = self.apply_user_stream_event(payload)
        if update is not None:
            self._pending_stream_updates.append(update)

    def drain_user_stream_updates(self) -> List[OrderUpdateEvent]:
        items = list(self._pending_stream_updates)
        self._pending_stream_updates.clear()
        return items

    async def submit_order(
        self,
        *,
        intent: OrderIntentEvent,
        best_bid: Decimal,
        best_ask: Decimal,
        now_ts_ms: int,
    ) -> Tuple[OrderState, OrderUpdateEvent]:
        slippage = self.slippage_guard.check(
            side=intent.side,
            limit_price=intent.limit_price,
            best_bid=best_bid,
            best_ask=best_ask,
            max_slippage_bps=intent.max_slippage_bps,
        )

        state = self.orders.get(intent.order_intent_id)
        if state is None:
            state = OrderState(
                order_intent_id=intent.order_intent_id,
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                limit_price=intent.limit_price,
            )
            self.orders[state.order_intent_id] = state

        self.state_machine.on_submit(state, now_ts_ms)

        if not slippage.allowed:
            self.state_machine.on_reject(state, slippage.reason)
            update = self._make_update_event(state, now_ts_ms, detail={"reason": slippage.reason})
            return state, update

        client_order_id = intent.order_intent_id[:36]
        self.client_order_id_to_intent_id[client_order_id] = intent.order_intent_id

        if self.config.paper_mode:
            order_id = f"paper-{intent.order_intent_id[:12]}"
            self.order_id_to_intent_id[order_id] = intent.order_intent_id
            self.state_machine.on_ack(state, order_id=order_id, ts_ms=now_ts_ms)
            arrival_price = best_ask if intent.side == "BUY" else best_bid
            fill_price = arrival_price
            self.state_machine.on_fill(state, fill_qty=intent.qty, fill_price=fill_price)
            realized_slippage_bps = self._realized_slippage_bps(
                side=intent.side,
                arrival_price=arrival_price,
                fill_price=fill_price,
            )
            update = self._make_update_event(
                state,
                now_ts_ms,
                detail={
                    "paper_mode": True,
                    "client_order_id": client_order_id,
                    "arrival_price": str(arrival_price),
                    "fill_price": str(fill_price),
                    "fill_qty": str(intent.qty),
                    "realized_slippage_bps": realized_slippage_bps,
                },
            )
            return state, update

        params = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": str(intent.qty),
            "price": str(intent.limit_price),
            "newClientOrderId": client_order_id,
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }

        payload = await asyncio.to_thread(self._blocking_request, "POST", "/fapi/v1/order", params)
        order_id = str(payload.get("orderId", ""))
        if len(order_id) == 0:
            self.state_machine.on_reject(state, "missing_order_id")
        else:
            self.order_id_to_intent_id[order_id] = intent.order_intent_id
            self.state_machine.on_ack(state, order_id=order_id, ts_ms=now_ts_ms)

        update = self._make_update_event(
            state,
            now_ts_ms,
            detail={"exchange_payload": payload, "client_order_id": client_order_id},
        )
        return state, update

    def apply_user_stream_event(
        self,
        payload: Dict[str, object],
        *,
        recv_ts_ms: Optional[int] = None,
    ) -> Optional[OrderUpdateEvent]:
        if payload.get("e") != "ORDER_TRADE_UPDATE":
            return None

        order_data = payload.get("o")
        if not isinstance(order_data, dict):
            return None

        client_order_id = str(order_data.get("c", ""))
        order_id = str(order_data.get("i", ""))

        order_intent_id = self.client_order_id_to_intent_id.get(client_order_id)
        if order_intent_id is None and order_id:
            order_intent_id = self.order_id_to_intent_id.get(order_id)
        if order_intent_id is None:
            return None

        state = self.orders.get(order_intent_id)
        if state is None:
            return None

        event_ts_ms = int(payload.get("E") or recv_ts_ms or int(time.time() * 1000))
        execution_type = str(order_data.get("x", ""))
        order_status = str(order_data.get("X", ""))

        if state.status == OrderStatus.SUBMITTED:
            if order_id:
                self.order_id_to_intent_id[order_id] = order_intent_id
            self.state_machine.on_ack(state, order_id=order_id or (state.order_id or ""), ts_ms=event_ts_ms)

        fill_qty = Decimal(str(order_data.get("l", "0")))
        fill_price = Decimal(str(order_data.get("L", "0")))
        if execution_type == "TRADE" and fill_qty > 0 and fill_price > 0:
            self.state_machine.on_fill(state, fill_qty=fill_qty, fill_price=fill_price)

        if order_status == "CANCELED":
            self.state_machine.on_cancel(state, "exchange_canceled")
        elif order_status in {"EXPIRED", "EXPIRED_IN_MATCH"}:
            self.state_machine.on_expire(state)
        elif order_status == "REJECTED":
            reject_reason = str(order_data.get("r", "exchange_rejected"))
            self.state_machine.on_reject(state, reject_reason)

        if order_status == "FILLED":
            state.status = OrderStatus.FILLED

        flatten_fields = self._residual_flatten_fields(state)

        return self._make_update_event(
            state,
            event_ts_ms,
            detail={
                "source": "user_stream",
                "execution_type": execution_type,
                "order_status": order_status,
                "client_order_id": client_order_id,
                "exchange_order_id": order_id,
                "last_fill_qty": str(fill_qty),
                "last_fill_price": str(fill_price),
                "cum_fill_qty": str(order_data.get("z", "0")),
                **flatten_fields,
            },
        )

    def cancel_order_intent(
        self,
        *,
        order_intent_id: str,
        reason: str,
        now_ts_ms: int,
    ) -> Optional[OrderUpdateEvent]:
        state = self.orders.get(order_intent_id)
        if state is None:
            return None
        if self.state_machine.is_terminal(state):
            return None

        if not self.config.paper_mode and state.order_id:
            params = {
                "symbol": state.symbol,
                "orderId": state.order_id,
                "recvWindow": self.config.recv_window_ms,
                "timestamp": int(time.time() * 1000),
            }
            self._blocking_request("DELETE", "/fapi/v1/order", params)

        self.state_machine.on_cancel(state, reason)
        flatten_fields = self._residual_flatten_fields(state)
        return self._make_update_event(
            state,
            now_ts_ms,
            detail={
                "reason": reason,
                **flatten_fields,
            },
        )

    def _prune_terminal_orders(self) -> None:
        terminal_ids = [
            oid for oid, state in self.orders.items()
            if self.state_machine.is_terminal(state)
        ]
        for oid in terminal_ids:
            state = self.orders.pop(oid, None)
            if state is not None:
                self.client_order_id_to_intent_id.pop(oid[:36], None)
                if state.order_id:
                    self.order_id_to_intent_id.pop(state.order_id, None)

    def collect_timeout_updates(self, now_ts_ms: int) -> List[OrderUpdateEvent]:
        updates: List[OrderUpdateEvent] = []
        for state in list(self.orders.values()):
            if self.state_machine.is_terminal(state):
                continue
            if not self.state_machine.should_cancel_for_timeout(state, now_ts_ms):
                continue

            if state.status == OrderStatus.SUBMITTED:
                self.state_machine.on_reject(state, "ack_timeout")
                updates.append(
                    self._make_update_event(
                        state,
                        now_ts_ms,
                        detail={"reason": "ack_timeout"},
                    )
                )
                continue

            reason = "ttl_timeout_cancel"
            update = self.cancel_order_intent(
                order_intent_id=state.order_intent_id,
                reason=reason,
                now_ts_ms=now_ts_ms,
            )
            if update is not None:
                updates.append(update)
        self._prune_terminal_orders()
        return updates

    def _make_update_event(self, state: OrderState, ts_ms: int, detail: Dict[str, object]) -> OrderUpdateEvent:
        detail_out = {
            **detail,
            "state_side": state.side,
            "state_qty": str(state.qty),
            "state_limit_price": str(state.limit_price),
            "state_filled_qty": str(state.filled_qty),
            "state_avg_fill_price": str(state.avg_fill_price),
            "state_terminal_reason": state.terminal_reason or None,
        }
        event_id = make_stable_id(
            EVENT_NAMESPACE,
            "order_update",
            state.order_intent_id,
            state.status.value,
            ts_ms,
        )
        return OrderUpdateEvent(
            event_id=event_id,
            schema_version=SCHEMA_VERSION_V1,
            order_intent_id=state.order_intent_id,
            order_id=state.order_id,
            symbol=state.symbol,
            status=state.status.value,
            exchange_ts_ms=int(ts_ms),
            recv_mono_ts_ns=time.monotonic_ns(),
            detail=detail_out,
        )
