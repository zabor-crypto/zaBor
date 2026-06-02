"""Second-level microstructure feature engine for continuation-under-stress v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from v1.book.local_book import LocalOrderBookEngine
from v1.contracts.event_types import (
    AggTradeEvent,
    FeatureSnapshotEvent,
    ForceOrderEvent,
    MarkPriceEvent,
    SCHEMA_VERSION_V1,
)
from v1.contracts.ids import EVENT_NAMESPACE, make_episode_id, make_stable_id
from v1.features.event_windows import EventWindow, MidPriceWindow


@dataclass
class SymbolFeatureState:
    buy_notional: EventWindow
    sell_notional: EventWindow
    liq_buy_notional: EventWindow
    liq_sell_notional: EventWindow
    spread_bps: EventWindow
    depth_notional: EventWindow
    mid_prices: MidPriceWindow
    last_mark_price: float = 0.0


class MicrostructureFeatureEngine:
    def __init__(self, symbols: Iterable[str]) -> None:
        self._states: Dict[str, SymbolFeatureState] = {
            s: SymbolFeatureState(
                buy_notional=EventWindow(max_window_ms=6_000),
                sell_notional=EventWindow(max_window_ms=6_000),
                liq_buy_notional=EventWindow(max_window_ms=6_000),
                liq_sell_notional=EventWindow(max_window_ms=6_000),
                spread_bps=EventWindow(max_window_ms=6_000),
                depth_notional=EventWindow(max_window_ms=6_000),
                mid_prices=MidPriceWindow(max_window_ms=6_000),
            )
            for s in symbols
        }

    def on_agg_trade(self, event: AggTradeEvent) -> None:
        state = self._states[event.symbol]
        notional = float(event.price * event.qty)
        if event.taker_side == "BUY":
            state.buy_notional.add(event.exchange_ts_ms, notional)
        else:
            state.sell_notional.add(event.exchange_ts_ms, notional)

    def on_force_order(self, event: ForceOrderEvent) -> None:
        state = self._states.get(event.symbol)
        if state is None:
            return  # !forceOrder@arr delivers all symbols; skip non-trading ones
        notional = float(event.derived_notional)
        if event.side == "BUY":
            state.liq_buy_notional.add(event.exchange_ts_ms, notional)
        else:
            state.liq_sell_notional.add(event.exchange_ts_ms, notional)

    def on_mark_price(self, event: MarkPriceEvent) -> None:
        state = self._states[event.symbol]
        state.last_mark_price = float(event.mark_price)

    def on_order_book(self, symbol: str, ts_ms: int, book: LocalOrderBookEngine) -> None:
        state = self._states[symbol]
        spread = book.spread_bps(symbol)
        if spread is not None:
            state.spread_bps.add(ts_ms, spread)
        tob = book.top_of_book(symbol)
        if tob is None:
            return

        best_bid_px, best_bid_qty, best_ask_px, best_ask_qty = tob
        _bid = float(best_bid_px)
        _ask = float(best_ask_px)
        mid = (_bid + _ask) * 0.5
        top_depth_notional = _bid * float(best_bid_qty) + _ask * float(best_ask_qty)

        state.depth_notional.add(ts_ms, top_depth_notional)
        state.mid_prices.add(ts_ms, mid)

    def build_snapshot(self, symbol: str, decision_ts_ms: int) -> FeatureSnapshotEvent:
        state = self._states[symbol]

        # ── 1-second flow ────────────────────────────────────────────────────
        buy_1s = state.buy_notional.sum(1000, now_ts_ms=decision_ts_ms)
        sell_1s = state.sell_notional.sum(1000, now_ts_ms=decision_ts_ms)
        flow_total = buy_1s + sell_1s
        flow_imbalance = 0.0 if flow_total <= 0 else (buy_1s - sell_1s) / flow_total

        # ── 3-second flow (direction stability) ─────────────────────────────
        buy_3s = state.buy_notional.sum(3000, now_ts_ms=decision_ts_ms)
        sell_3s = state.sell_notional.sum(3000, now_ts_ms=decision_ts_ms)
        flow_total_3s = buy_3s + sell_3s
        flow_imbalance_3s = 0.0 if flow_total_3s <= 0 else (buy_3s - sell_3s) / flow_total_3s

        # Flow acceleration: positive = flow is building, negative = decaying
        flow_acceleration = flow_imbalance - flow_imbalance_3s

        # ── 1-second liquidation (directional) ──────────────────────────────
        liq_buy_1s = state.liq_buy_notional.sum(1000, now_ts_ms=decision_ts_ms)
        liq_sell_1s = state.liq_sell_notional.sum(1000, now_ts_ms=decision_ts_ms)
        liq_total_1s = liq_buy_1s + liq_sell_1s

        # Liquidation imbalance: +1 = all shorts liquidated (bullish), -1 = all longs (bearish)
        liq_imbalance_1s = 0.0
        if liq_total_1s > 0:
            liq_imbalance_1s = (liq_buy_1s - liq_sell_1s) / liq_total_1s

        # Liquidation-flow alignment: 1.0 when liq direction confirms flow direction
        liq_flow_aligned = 0.0
        if liq_total_1s > 0 and abs(flow_imbalance) > 0.01:
            liq_flow_aligned = 1.0 if (liq_imbalance_1s * flow_imbalance > 0) else 0.0

        # ── 3-second liquidation (cascade context) ───────────────────────────
        liq_buy_3s = state.liq_buy_notional.sum(3000, now_ts_ms=decision_ts_ms)
        liq_sell_3s = state.liq_sell_notional.sum(3000, now_ts_ms=decision_ts_ms)
        liq_total_3s = liq_buy_3s + liq_sell_3s

        # Cascade momentum: >0 means liq is accelerating (recent 1s > average rate)
        liq_momentum = 0.0
        if liq_total_3s > 0:
            liq_momentum = (liq_total_1s / liq_total_3s) - (1.0 / 3.0)

        # ── spread / depth / impact ──────────────────────────────────────────
        spread_now = 0.0
        spread_latest = state.spread_bps.latest()
        if spread_latest is not None:
            spread_now = spread_latest.value

        depth_now = 0.0
        depth_latest = state.depth_notional.latest()
        if depth_latest is not None:
            depth_now = depth_latest.value
        depth_mean_5s = state.depth_notional.mean(5000, now_ts_ms=decision_ts_ms)
        depth_collapse = 0.0
        if depth_mean_5s > 0:
            depth_collapse = max(0.0, (depth_mean_5s - depth_now) / depth_mean_5s)

        ret_500_bps = state.mid_prices.return_bps(500, now_ts_ms=decision_ts_ms)
        ret_1500_bps = state.mid_prices.return_bps(1500, now_ts_ms=decision_ts_ms)
        recovery_bps = ret_1500_bps - ret_500_bps
        failed_recovery = 1.0 if abs(ret_1500_bps) > abs(ret_500_bps) else 0.0

        features = {
            "taker_buy_notional_1s": buy_1s,
            "taker_sell_notional_1s": sell_1s,
            "flow_imbalance_1s": flow_imbalance,
            "flow_imbalance_3s": flow_imbalance_3s,
            "flow_acceleration": flow_acceleration,
            "liq_buy_notional_1s": liq_buy_1s,
            "liq_sell_notional_1s": liq_sell_1s,
            "liq_total_notional_1s": liq_total_1s,
            "liq_imbalance_1s": liq_imbalance_1s,
            "liq_flow_aligned": liq_flow_aligned,
            "liq_total_notional_3s": liq_total_3s,
            "liq_momentum": liq_momentum,
            "spread_bps": spread_now,
            "depth_collapse_ratio": depth_collapse,
            "ret_500_bps": ret_500_bps,
            "ret_1500_bps": ret_1500_bps,
            "recovery_bps": recovery_bps,
            "failed_recovery": failed_recovery,
            "mark_price": state.last_mark_price,
        }

        feature_hash = make_stable_id(
            EVENT_NAMESPACE,
            symbol,
            decision_ts_ms,
            round(features["flow_imbalance_1s"], 6),
            round(features["depth_collapse_ratio"], 6),
            round(features["spread_bps"], 6),
            round(features["ret_500_bps"], 6),
            round(features["ret_1500_bps"], 6),
        )
        episode_id = make_episode_id(symbol, decision_ts_ms, feature_hash)

        return FeatureSnapshotEvent(
            event_id=feature_hash,
            schema_version=SCHEMA_VERSION_V1,
            episode_id=episode_id,
            decision_ts_ms=decision_ts_ms,
            symbol=symbol,
            features=features,
        )
