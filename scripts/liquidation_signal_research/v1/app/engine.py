"""Main continuation v1 runtime engine and replay-compatible pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from v1.app.runtime import RuntimeConfig
from v1.book.local_book import LocalOrderBookEngine
from v1.book.snapshot_sync import BinanceSnapshotClient
from v1.contracts.event_types import (
    AggTradeEvent,
    DepthDeltaEvent,
    ForceOrderEvent,
    MarkPriceEvent,
    RawMarketEvent,
)
from v1.contracts.event_types import OrderIntentEvent, SCHEMA_VERSION_V1
from v1.contracts.ids import make_order_intent_id
from v1.execution.binance_executor import BinanceUmExecutor, ExecutorConfig
from v1.execution.position_manager import ExitSignal, OpenPosition, PositionManager
from v1.features.microstructure_features import MicrostructureFeatureEngine
from v1.ingest.binance_ws_client import BinanceWsClient
from v1.ingest.raw_capture_writer import RawCaptureWriter
from v1.ingest.stream_supervisor import StreamSupervisor
from v1.normalize.normalizer import EventNormalizer
from v1.regime.regime_gate import RegimeGate, RegimeThresholds
from v1.risk.calibration_gate import PerAssetCalibrationGate
from v1.risk.kill_switch import KillSwitchThresholds
from v1.risk.ml_quality_gate import MLQualityGate
from v1.risk.risk_engine import RiskEngine, RiskLimits, SessionRiskState
from v1.signal.continuation_score import ContinuationScoreEngine
from v1.store.jsonl_partition_store import JsonlPartitionStore
from v1.telemetry.metrics import MetricsRegistry
from v1.telemetry.shadow_monitors import ShadowMonitor
from v1.telemetry.structured_logs import TelemetryRouter

logger = logging.getLogger(__name__)

# ── Fee rates ──────────────────────────────────────────────────────────────────
# Applied per notional side (entry or exit individually, not combined).
# Paper mode mirrors Binance UM futures standard rates:
#   Taker: 0.0004 = 4 bps per side
#   Maker: 0.0002 = 2 bps per side
# The legacy formula used (entry_px + exit_px) * qty * 0.0006 which over-estimated
# fees (~6 bps/side).  New formula: entry_fee + exit_fee separately so maker savings
# are precisely measurable.
TAKER_FEE_RATE: float = 0.0004   # 4 bps per side (was 0.0006 in the old combined formula)
MAKER_FEE_RATE: float = 0.0002   # 2 bps per side — maker entry / passive trailing exit


@dataclass
class PendingMakerEntry:
    """Tracks an open (unfilled) maker limit entry order in paper simulation."""
    symbol: str
    side: str                # "BUY" or "SELL"
    limit_price: float       # passive limit: bid for BUY, ask for SELL
    qty: float
    deadline_ms: int         # cancel if no fill by this timestamp
    signal_ts_ms: int        # when the signal fired
    decision_id: str
    episode_id: str
    score: float
    liq_notional_1s: float
    decision: Any            # DecisionIntentEvent (typed Any to avoid circular import)


class ContinuationEngine:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        snapshot_client: BinanceSnapshotClient | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()

        self.store = JsonlPartitionStore(
            self.config.data_root,
            flush_interval_events=self.config.store_flush_interval_events,
        )
        self.capture_writer = RawCaptureWriter(
            store=self.store,
            queue_size=self.config.capture_queue_size,
        )
        self.normalizer = EventNormalizer()
        self.book = LocalOrderBookEngine(
            self.config.symbols,
            stale_after_ms=self.config.book_stale_after_ms,
            max_depth_levels=self.config.max_book_depth_levels,
        )
        self.features = MicrostructureFeatureEngine(self.config.symbols)
        _regime_thresholds = RegimeThresholds(
            enable_session_filter=not self.config.regime_session_optional,
            enable_liq_dead_zone_filter=not self.config.regime_liq_optional,
            enable_liq_stress_filter=not self.config.regime_liq_optional,
            enable_liq_flow_alignment_filter=not self.config.regime_liq_optional,
        )
        self.regime_gate = RegimeGate(_regime_thresholds)
        self.signal_engine = ContinuationScoreEngine()
        self.risk_engine = RiskEngine(
            RiskLimits(
                max_consecutive_losses=self.config.risk_max_consecutive_losses,
                max_symbol_consecutive_losses=self.config.risk_max_symbol_consecutive_losses,
            ),
            kill_switch_thresholds=KillSwitchThresholds(
                max_drawdown_pct=self.config.kill_switch_max_drawdown_pct,
            ),
        )
        self.executor = BinanceUmExecutor(
            config=ExecutorConfig(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
                base_url=self.config.execution_base_url,
                user_stream_ws_base_url=self.config.execution_user_stream_ws_base_url,
                paper_mode=self.config.execution_paper_mode,
            )
        )
        self.telemetry = TelemetryRouter(
            self.config.telemetry_root,
            feature_flush_interval=self.config.telemetry_feature_flush_interval,
            decision_flush_interval=self.config.telemetry_decision_flush_interval,
        )
        self.metrics = MetricsRegistry()
        self.supervisor = StreamSupervisor(
            symbols=self.config.symbols,
            # HYPEUSDT aggTrade is sparse during low-vol periods; avoid spurious health failures
            stale_overrides={("HYPEUSDT", "aggTrade"): 8_000},
        )
        self.position_manager = PositionManager()
        self.shadow_monitor = ShadowMonitor()
        self.snapshot_client = snapshot_client or BinanceSnapshotClient()
        self.paper_calibration_gate: Optional[PerAssetCalibrationGate] = None
        self._paper_calibration_gate_path: Optional[Path] = None

        if (
            self.config.execution_paper_mode
            and (not self.config.shadow_mode)
            and bool(self.config.enable_paper_calibration_gate)
        ):
            if self.config.calibration_overrides_path is None:
                raise RuntimeError(
                    "Paper calibration gate enabled but calibration_overrides_path is not set"
                )
            else:
                try:
                    gate_path = Path(self.config.calibration_overrides_path)
                    self.paper_calibration_gate = PerAssetCalibrationGate.from_file(gate_path)
                    self._paper_calibration_gate_path = gate_path
                    logger.info(
                        "Paper calibration gate enabled from %s (symbols=%s)",
                        str(gate_path),
                        ",".join(self.paper_calibration_gate.active_symbols()),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load paper calibration overrides from "
                        f"{self.config.calibration_overrides_path}: {exc}"
                    )

        # ML quality gate: hot-loaded from ml_auto_calibrator output
        # Looks in analysis/ml_calibration/ relative to project root (data_root's grandparent)
        self.ml_quality_gate: Optional[MLQualityGate] = None
        if not self.config.disable_ml_quality_gate:
            _proj_root = self.config.data_root.parent.parent
            ml_cal_path = _proj_root / "analysis" / "ml_calibration" / "ml_calibration_latest.json"
            if ml_cal_path.exists():
                try:
                    self.ml_quality_gate = MLQualityGate.from_file(ml_cal_path)
                    logger.info("ML quality gate loaded from %s", ml_cal_path)
                except Exception as exc:
                    logger.warning("ML quality gate load failed: %s", exc)
        else:
            logger.info("ML quality gate disabled via config.disable_ml_quality_gate")

        self._client: Optional[BinanceWsClient] = None
        self._telemetry_flush_task: Optional[asyncio.Task] = None
        self._last_decision_ts_ms: Dict[str, int] = {symbol: 0 for symbol in self.config.symbols}
        self._sync_pending: Dict[str, bool] = {symbol: True for symbol in self.config.symbols}
        self._last_resync_attempt_ms: Dict[str, int] = {symbol: 0 for symbol in self.config.symbols}
        self._resync_tasks: Dict[str, asyncio.Task] = {}
        self._depth_buffers = {
            symbol: deque(maxlen=self.config.depth_buffer_max_events)
            for symbol in self.config.symbols
        }
        self.session_state = SessionRiskState(nav_usd=10_000.0, peak_nav_usd=10_000.0)
        self.decisions: deque[Dict[str, object]] = deque(maxlen=10_000)
        # throttle telemetry writes for pure-blocked decisions (storage fix)
        self._last_feature_log_ms: Dict[str, int] = {symbol: 0 for symbol in self.config.symbols}
        self._last_decision_log_ms: Dict[str, int] = {symbol: 0 for symbol in self.config.symbols}
        # prevent concurrent order submissions per symbol (double-fill guard)
        self._symbol_pending_submit: Dict[str, bool] = {symbol: False for symbol in self.config.symbols}
        # fast-path exit: last mid price used for 0.5 bps change threshold
        # 0.0 forces a check on the first event after position open.
        self._last_fast_mid: Dict[str, float] = {symbol: 0.0 for symbol in self.config.symbols}
        # Phase 1 maker entry: one pending passive entry per symbol
        self._pending_maker_entries: Dict[str, PendingMakerEntry] = {}
        # Maker investigation counters (reset each session, reported in risk log)
        self._maker_fills: int = 0
        self._maker_misses: int = 0

    async def start_live(self) -> None:
        await self.capture_writer.start()
        if self.config.auto_snapshot_sync:
            await self._bootstrap_snapshots()
        await self.executor.start()
        self._client = BinanceWsClient(
            symbols=self.config.symbols,
            streams=self.config.streams,
            on_raw_event=self.on_raw_event,
            local_addr=self.config.ws_local_addr or None,
            socks5_proxy=self.config.ws_socks5_proxy or None,
        )
        self._telemetry_flush_task = asyncio.create_task(
            self._periodic_telemetry_flush(), name="telemetry-flush"
        )
        await self.supervisor.start(self._client)

    async def stop_live(self) -> None:
        for task in list(self._resync_tasks.values()):
            task.cancel()
        if self._resync_tasks:
            await asyncio.gather(*self._resync_tasks.values(), return_exceptions=True)
        self._resync_tasks.clear()
        if self._telemetry_flush_task is not None:
            self._telemetry_flush_task.cancel()
            try:
                await self._telemetry_flush_task
            except asyncio.CancelledError:
                pass
            self._telemetry_flush_task = None
        self.telemetry.flush_all()
        await self.executor.stop()
        await self.supervisor.stop()
        await self.capture_writer.stop()

    async def _periodic_telemetry_flush(self) -> None:
        """Flush telemetry write buffers to OS every 30 s."""
        while True:
            await asyncio.sleep(30)
            try:
                self.telemetry.flush_all()
            except Exception:
                logger.exception("periodic telemetry flush failed")

    async def _bootstrap_snapshots(self) -> None:
        await asyncio.gather(
            *[
                self._refresh_snapshot(symbol=symbol, reason="startup_bootstrap")
                for symbol in self.config.symbols
            ],
            return_exceptions=True,
        )

    def _mark_symbol_in_sync_if_ready(self, symbol: str) -> None:
        state = self.book.states[symbol]
        if state.last_update_id is not None and self.book.is_healthy(symbol):
            self._sync_pending[symbol] = False

    def _book_ready(self, symbol: str) -> bool:
        self._mark_symbol_in_sync_if_ready(symbol)
        return self.book.is_healthy(symbol) and (not self._sync_pending[symbol])

    async def _refresh_snapshot(self, *, symbol: str, reason: str) -> None:
        self._sync_pending[symbol] = True
        now_ms = int(time.time() * 1000)
        self._last_resync_attempt_ms[symbol] = now_ms
        self._log_risk_event(
            {
                "symbol": symbol,
                "event": "book_resync_start",
                "reason": reason,
                "exchange_ts_ms": now_ms,
            }
        )
        self.metrics.inc("book_resync_attempt_total")

        try:
            snapshot = await asyncio.wait_for(
                self.snapshot_client.fetch_snapshot(
                    symbol,
                    limit=self.config.snapshot_depth_limit,
                ),
                timeout=self.config.snapshot_fetch_timeout_seconds,
            )
            if snapshot.exchange_ts_ms <= 0:
                snapshot = type(snapshot)(
                    symbol=snapshot.symbol,
                    last_update_id=snapshot.last_update_id,
                    bids=snapshot.bids,
                    asks=snapshot.asks,
                    exchange_ts_ms=now_ms,
                )

            self.book.apply_snapshot(snapshot)
            replay_ok = self._replay_buffered_depth(symbol)
            self.features.on_order_book(symbol, snapshot.exchange_ts_ms, self.book)
            if not replay_ok:
                self._sync_pending[symbol] = True
                self.metrics.inc("book_resync_replay_retry_total")
                self._log_risk_event(
                    {
                        "symbol": symbol,
                        "event": "book_resync_replay_retry",
                        "exchange_ts_ms": snapshot.exchange_ts_ms,
                    }
                )
                return
            self._sync_pending[symbol] = False
            self.metrics.inc("book_resync_success_total")
            self._log_risk_event(
                {
                    "symbol": symbol,
                    "event": "book_resync_ok",
                    "exchange_ts_ms": snapshot.exchange_ts_ms,
                    "last_update_id": snapshot.last_update_id,
                }
            )
        except Exception as exc:
            self.metrics.inc("book_resync_failure_total")
            self._sync_pending[symbol] = True
            self._log_risk_event(
                {
                    "symbol": symbol,
                    "event": "book_resync_failed",
                    "reason": str(exc),
                    "exchange_ts_ms": now_ms,
                }
            )
            logger.warning("Snapshot refresh failed for %s: %s", symbol, exc)

    def _schedule_snapshot_resync(self, *, symbol: str, reason: str) -> None:
        if not self.config.auto_snapshot_sync:
            return
        now_ms = int(time.time() * 1000)
        last_attempt = self._last_resync_attempt_ms.get(symbol, 0)
        if (now_ms - last_attempt) < self.config.snapshot_resync_cooldown_ms:
            return

        existing = self._resync_tasks.get(symbol)
        if existing is not None and not existing.done():
            return

        async def _runner() -> None:
            await self._refresh_snapshot(symbol=symbol, reason=reason)

        task = asyncio.create_task(_runner(), name=f"book-resync-{symbol}")
        self._resync_tasks[symbol] = task

        def _done(t: asyncio.Task) -> None:
            current = self._resync_tasks.get(symbol)
            if current is task:
                self._resync_tasks.pop(symbol, None)
            if not t.cancelled() and t.exception() is not None:
                logger.error(
                    "book_resync_task_unhandled_error symbol=%s: %s",
                    symbol,
                    t.exception(),
                    exc_info=t.exception(),
                )

        task.add_done_callback(_done)

    def _buffer_depth_delta(self, event: DepthDeltaEvent) -> None:
        buf = self._depth_buffers[event.symbol]
        if len(buf) >= buf.maxlen:
            self.metrics.inc("depth_buffer_overflow_total")
            self._log_risk_event(
                {
                    "symbol": event.symbol,
                    "event": "depth_buffer_overflow",
                    "exchange_ts_ms": event.exchange_ts_ms,
                    "maxlen": buf.maxlen,
                }
            )
        buf.append(event)
        self.metrics.inc("depth_buffered_events_total")

    def _replay_buffered_depth(self, symbol: str) -> bool:
        buf = self._depth_buffers[symbol]
        if len(buf) == 0:
            return self.book.is_healthy(symbol)

        state = self.book.states[symbol]
        if state.last_update_id is None:
            return False

        events = list(buf)
        has_new_event = any(event.final_update_id > state.last_update_id for event in events)
        if not has_new_event:
            buf.clear()
            return True

        start_idx = None
        for idx, event in enumerate(events):
            if event.final_update_id <= state.last_update_id:
                continue
            if event.first_update_id <= (state.last_update_id + 1) <= event.final_update_id:
                start_idx = idx
                break

        if start_idx is None:
            self.metrics.inc("book_resync_replay_bridge_miss_total")
            self._log_risk_event(
                {
                    "symbol": symbol,
                    "event": "book_resync_replay_bridge_miss",
                    "exchange_ts_ms": int(time.time() * 1000),
                    "buffered_events": len(events),
                    "last_update_id": state.last_update_id,
                }
            )
            buf.clear()
            return False

        applied = 0
        for event in events[start_idx:]:
            if not self.book.apply_depth_delta(event):
                self.metrics.inc("book_resync_replay_failed_total")
                self._log_risk_event(
                    {
                        "symbol": symbol,
                        "event": "book_resync_replay_failed",
                        "reason": self.book.desync_reason(symbol),
                        "exchange_ts_ms": event.exchange_ts_ms,
                        "buffered_events": len(events),
                    }
                )
                buf.clear()
                return False
            applied += 1

        if applied > 0:
            self.metrics.inc("book_resync_replay_applied_total", applied)
            self._log_risk_event(
                {
                    "symbol": symbol,
                    "event": "book_resync_replay_applied",
                    "exchange_ts_ms": int(time.time() * 1000),
                    "applied_events": applied,
                }
            )
        buf.clear()
        return self.book.is_healthy(symbol)

    async def on_raw_event(self, raw_event: RawMarketEvent) -> None:
        self.supervisor.observe(raw_event)
        await self.capture_writer.enqueue(raw_event)
        # only log errors/gaps — not every event (storage: ~1.1GB/day → <1MB/day)
        if raw_event.parse_status != "ok" or raw_event.gap_flag:
            self.telemetry.log_raw_capture(
                {
                    "event_id": raw_event.event_id,
                    "symbol": raw_event.symbol,
                    "stream": raw_event.stream,
                    "ingest_seq": raw_event.ingest_seq,
                    "parse_status": raw_event.parse_status,
                    "gap_flag": raw_event.gap_flag,
                }
            )
        self.metrics.inc("raw_events_total")
        self._process_raw_event_common(raw_event, assume_stream_healthy=False)

    def _compute_entry_qty(self, symbol: str, mid_price: float) -> Decimal:
        """Target ~$100 notional, rounded to lot step."""
        from v1.app.runtime import DEFAULT_SYMBOL_SPECS
        spec = DEFAULT_SYMBOL_SPECS.get(symbol)
        if mid_price > 0 and spec is not None:
            target_notional = Decimal("100")
            raw_qty = target_notional / Decimal(str(mid_price))
            lot = Decimal(str(spec.lot_step))
            qty = (raw_qty / lot).to_integral_value() * lot
            min_q = Decimal(str(spec.min_qty))
            return qty if qty >= min_q else min_q
        return Decimal("0.001")

    def _open_position_from_fill(
        self,
        *,
        symbol: str,
        side: str,                  # "BUY" or "SELL" (intent side)
        fill_price: float,
        fill_ts_ms: int,
        qty: float,
        decision_id: str,
        episode_id: str,
        score: float,
        liq_notional_1s: float,
        entry_maker: bool = False,
    ) -> None:
        """Open position and update session state after a confirmed fill."""
        self.session_state.exposure_usd += qty * fill_price
        self.session_state.consecutive_rejects = 0
        self.position_manager.open_position(
            symbol=symbol,
            side="LONG" if side == "BUY" else "SHORT",
            entry_price=fill_price,
            entry_ts_ms=fill_ts_ms,
            qty=qty,
            decision_id=decision_id,
            episode_id=episode_id,
            score=score,
            liq_notional_1s=liq_notional_1s,
            entry_maker=entry_maker,
        )
        self._last_fast_mid[symbol] = 0.0

    async def _submit_and_log_order(
        self,
        *,
        decision_id: str,
        symbol: str,
        ts_ms: int,
        best_bid: Decimal,
        best_ask: Decimal,
        decision,
        score: float = 0.0,
        liq_notional_1s: float = 0.0,
    ) -> None:
        mid_price = float((best_bid + best_ask) / 2)
        qty = self._compute_entry_qty(symbol, mid_price)

        # ── Phase 1: Maker entry (paper simulation) ────────────────────────────
        # Post at passive side (bid for LONG, ask for SHORT).  Fill simulated
        # when an aggTrade occurs at/through the limit within maker_entry_timeout_ms.
        if self.config.maker_entry_mode and self.config.execution_paper_mode:
            side = "BUY" if decision.action == "ENTER_LONG" else "SELL"
            # Passive limit: best_bid for BUY (we want sellers to hit us),
            # best_ask for SELL (we want buyers to lift us).
            limit_price = float(best_bid) if side == "BUY" else float(best_ask)
            entry = PendingMakerEntry(
                symbol=symbol,
                side=side,
                limit_price=limit_price,
                qty=float(qty),
                deadline_ms=ts_ms + self.config.maker_entry_timeout_ms,
                signal_ts_ms=ts_ms,
                decision_id=decision_id,
                episode_id=decision.episode_id,
                score=score,
                liq_notional_1s=liq_notional_1s,
                decision=decision,
            )
            self._pending_maker_entries[symbol] = entry
            # Keep _symbol_pending_submit=True until fill or miss (prevents double entry).
            self._log_risk_event({
                "symbol": symbol,
                "event": "maker_entry_pending",
                "exchange_ts_ms": ts_ms,
                "side": side,
                "limit_price": limit_price,
                "deadline_ms": entry.deadline_ms,
                "score": score,
            })
            return   # _symbol_pending_submit stays True; cleared on fill or miss

        # ── Standard taker path ───────────────────────────────────────────────
        intent = self.executor.build_order_intent(
            decision=decision,
            best_bid=best_bid,
            best_ask=best_ask,
            qty=qty,
        )
        state, update = await self.executor.submit_order(
            intent=intent,
            best_bid=best_bid,
            best_ask=best_ask,
            now_ts_ms=ts_ms,
        )
        self.telemetry.log_order(
            {
                "decision_id": decision_id,
                "order_intent_id": intent.order_intent_id,
                "order_id": state.order_id,
                "status": state.status.value,
                "symbol": symbol,
                "side": intent.side,
                "qty": str(intent.qty),
                "limit_price": str(intent.limit_price),
                "exchange_ts_ms": ts_ms,
                "score": score,
                "liq_notional_1s": liq_notional_1s,
                "detail": update.detail,
            }
        )
        if state.status.value == "FILLED":
            fill_price = float(update.detail.get("fill_price") or intent.limit_price)
            self._open_position_from_fill(
                symbol=symbol,
                side=intent.side,
                fill_price=fill_price,
                fill_ts_ms=ts_ms,
                qty=float(qty),
                decision_id=decision_id,
                episode_id=decision.episode_id,
                score=score,
                liq_notional_1s=liq_notional_1s,
                entry_maker=False,
            )
        elif state.status.value == "REJECTED":
            self.session_state.consecutive_rejects += 1
        # clear pending flag so next decision cycle can submit for this symbol
        self._symbol_pending_submit[symbol] = False
        if bool(update.detail.get("residual_flatten_required", False)):
            self._log_risk_event(
                {
                    "symbol": symbol,
                    "event": "residual_flatten_required",
                    "order_intent_id": intent.order_intent_id,
                    "exchange_ts_ms": ts_ms,
                }
            )

    def _check_and_fill_maker_entry(self, symbol: str, trade_price: float, ts_ms: int) -> None:
        """
        Called on every AggTrade event.  Checks if a pending maker entry should
        be filled (trade_price crosses our passive limit) or expired (deadline passed).

        This runs on the hot path and must be synchronous and fast.
        """
        entry = self._pending_maker_entries.get(symbol)
        if entry is None:
            return

        # ── Expiry check ──────────────────────────────────────────────────────
        if ts_ms > entry.deadline_ms:
            self._maker_misses += 1
            lag_ms = ts_ms - entry.signal_ts_ms
            self._log_risk_event({
                "symbol": symbol,
                "event": "maker_entry_miss",
                "exchange_ts_ms": ts_ms,
                "signal_ts_ms": entry.signal_ts_ms,
                "fill_lag_ms": lag_ms,
                "limit_price": entry.limit_price,
                "side": entry.side,
                "score": entry.score,
                "maker_fills_total": self._maker_fills,
                "maker_misses_total": self._maker_misses,
            })
            del self._pending_maker_entries[symbol]
            self._symbol_pending_submit[symbol] = False
            return

        # ── Fill check ────────────────────────────────────────────────────────
        # BUY: filled when a seller trades at or below our bid (price ≤ limit).
        # SELL: filled when a buyer trades at or above our ask (price ≥ limit).
        filled = (
            (entry.side == "BUY" and trade_price <= entry.limit_price)
            or (entry.side == "SELL" and trade_price >= entry.limit_price)
        )
        if not filled:
            return

        fill_lag_ms = ts_ms - entry.signal_ts_ms
        self._maker_fills += 1
        fill_rate = self._maker_fills / max(1, self._maker_fills + self._maker_misses) * 100.0

        # Price improvement vs taker (taker would cross spread; maker stays passive)
        taker_price = entry.limit_price  # already at bid/ask, no additional improvement
        price_improvement_bps = 0.0      # spread was near-zero; improvement is fee-only

        self._log_risk_event({
            "symbol": symbol,
            "event": "maker_entry_fill",
            "exchange_ts_ms": ts_ms,
            "signal_ts_ms": entry.signal_ts_ms,
            "fill_lag_ms": fill_lag_ms,
            "fill_price": trade_price,
            "limit_price": entry.limit_price,
            "side": entry.side,
            "score": entry.score,
            "price_improvement_bps": price_improvement_bps,
            "maker_fills_total": self._maker_fills,
            "maker_misses_total": self._maker_misses,
            "maker_fill_rate_pct": round(fill_rate, 1),
        })

        # Open position at the aggTrade fill price (at or better than our limit)
        self._open_position_from_fill(
            symbol=symbol,
            side=entry.side,
            fill_price=trade_price,
            fill_ts_ms=ts_ms,
            qty=entry.qty,
            decision_id=entry.decision_id,
            episode_id=entry.episode_id,
            score=entry.score,
            liq_notional_1s=entry.liq_notional_1s,
            entry_maker=True,
        )
        del self._pending_maker_entries[symbol]
        self._symbol_pending_submit[symbol] = False

    async def _submit_close_order(
        self,
        *,
        exit_signal: ExitSignal,
        ts_ms: int,
        best_bid: Decimal,
        best_ask: Decimal,
        pos: Optional[OpenPosition],
    ) -> None:
        symbol = exit_signal.symbol
        close_side = exit_signal.close_side
        qty = Decimal(str(pos.qty)) if pos is not None else Decimal("0.001")
        slippage_budget_bps = self.signal_engine.config.slippage_budget_bps

        if close_side == "BUY":
            limit_price = best_ask * (Decimal("1") + Decimal(str(slippage_budget_bps / 10_000.0)))
        else:
            limit_price = best_bid * (Decimal("1") - Decimal(str(slippage_budget_bps / 10_000.0)))

        close_intent_id = make_order_intent_id(
            decision_id=pos.decision_id if pos else "unknown",
            side=close_side,
            qty=str(qty),
            limit_price=str(limit_price),
        )
        intent = OrderIntentEvent(
            event_id=close_intent_id,
            schema_version=SCHEMA_VERSION_V1,
            decision_id=pos.decision_id if pos else "unknown",
            order_intent_id=close_intent_id,
            symbol=symbol,
            side=close_side,
            qty=qty,
            limit_price=limit_price,
            ttl_ms=self.executor.config.ttl_ms,
            max_slippage_bps=slippage_budget_bps,
        )
        state, update = await self.executor.submit_order(
            intent=intent,
            best_bid=best_bid,
            best_ask=best_ask,
            now_ts_ms=ts_ms,
        )

        fill_price_raw = update.detail.get("fill_price")
        fill_price = float(fill_price_raw) if fill_price_raw else float(limit_price)

        gross_pnl_usd = 0.0
        gross_pnl_bps = 0.0
        fee_usd = 0.0
        net_pnl_usd = 0.0
        entry_mode = "taker"
        exit_mode = "taker"
        if pos is not None:
            direction = 1.0 if pos.side == "LONG" else -1.0
            gross_pnl_usd = direction * (fill_price - pos.entry_price) * pos.qty
            gross_pnl_bps = direction * (fill_price - pos.entry_price) / pos.entry_price * 10_000.0
            # ── Per-side fee model (Phase 1 / Phase 3 maker investigation) ────────
            # Entry: maker rate if filled as a passive limit order (Phase 1),
            #        taker rate for standard cross fills.
            # Exit: maker rate if trailing_stop + maker_exit_trailing enabled (Phase 3),
            #       taker rate for all other exits (hard_stop, hard_tp, flow_reversal, etc.)
            entry_fee_rate = MAKER_FEE_RATE if pos.entry_maker else TAKER_FEE_RATE
            is_maker_exit = self.config.maker_exit_trailing and exit_signal.exit_maker
            exit_fee_rate = MAKER_FEE_RATE if is_maker_exit else TAKER_FEE_RATE
            entry_mode = "maker" if pos.entry_maker else "taker"
            exit_mode = "maker" if is_maker_exit else "taker"
            fee_usd = (pos.entry_price * pos.qty * entry_fee_rate) + (fill_price * pos.qty * exit_fee_rate)
            net_pnl_usd = gross_pnl_usd - fee_usd
            self.session_state.daily_pnl_usd += net_pnl_usd
            self.session_state.session_pnl_usd += net_pnl_usd
            if self.session_state.nav_usd > 0.0:
                self.session_state.nav_usd += net_pnl_usd
                if self.session_state.peak_nav_usd <= 0.0:
                    self.session_state.peak_nav_usd = self.session_state.nav_usd
                if self.session_state.nav_usd > self.session_state.peak_nav_usd:
                    self.session_state.peak_nav_usd = self.session_state.nav_usd
                if self.session_state.peak_nav_usd > 0.0:
                    # Stored as fraction (0.08 = 8%), matching the convention of
                    # other risk fields (max_exposure_fraction, max_daily_loss_fraction)
                    # and the threshold this is compared to in KillSwitchThresholds.
                    self.session_state.drawdown_pct = max(
                        0.0,
                        (self.session_state.peak_nav_usd - self.session_state.nav_usd)
                        / self.session_state.peak_nav_usd,
                    )

            symbol_key = str(symbol or "").upper()
            if net_pnl_usd < 0.0:
                self.session_state.consecutive_losses += 1
                self.session_state.symbol_consecutive_losses[symbol_key] = (
                    self.session_state.symbol_consecutive_losses.get(symbol_key, 0) + 1
                )
            else:
                self.session_state.consecutive_losses = 0
                self.session_state.symbol_consecutive_losses[symbol_key] = 0

        self.telemetry.log_order(
            {
                "type": "close_order",
                "decision_id": pos.decision_id if pos else "unknown",
                "episode_id": pos.episode_id if pos else "unknown",
                "order_intent_id": intent.order_intent_id,
                "order_id": state.order_id,
                "status": state.status.value,
                "symbol": symbol,
                "side": close_side,
                "qty": str(qty),
                "limit_price": str(limit_price),
                "exchange_ts_ms": ts_ms,
                "fill_price": fill_price,
                "exit_reason": exit_signal.reason,
                "gross_pnl_bps": gross_pnl_bps,
                "gross_pnl_usd": gross_pnl_usd,
                "fee_usd": fee_usd,
                "net_pnl_usd": net_pnl_usd,
                "mfe_bps": exit_signal.mfe_bps,
                "mae_bps": exit_signal.mae_bps,
                "hold_ms": exit_signal.hold_ms,
                "entry_price": pos.entry_price if pos else None,
                # entry context — for score stratification and calibration analysis
                "score": pos.score if pos else None,
                "liq_notional_1s": pos.liq_notional_1s if pos else None,
                "dynamic_tp_bps": pos.dynamic_tp_bps if pos else None,
                # maker investigation — track per-trade fee regime for Phase 1 / Phase 3 analysis
                "entry_mode": entry_mode,
                "exit_mode": exit_mode,
                # ── TOB snapshot at close (May 20 2026 — enables maker-mode fillrate
                # modelling and adverse-selection analysis without needing to re-read
                # raw depth files post-hoc). best_bid / best_ask are the values passed
                # into _submit_close_order from the caller; mid is their midpoint.
                "tob_best_bid": float(best_bid),
                "tob_best_ask": float(best_ask),
                "tob_mid": float((best_bid + best_ask) / Decimal("2")),
                "tob_spread_bps": float((best_ask - best_bid) / ((best_bid + best_ask) / Decimal("2")) * Decimal("10000")) if best_bid > 0 else None,
                "detail": update.detail,
            }
        )

    def _log_risk_event(self, payload: Dict[str, object]) -> None:
        self.telemetry.log_risk(payload)
        self.shadow_monitor.observe_risk_event(payload)

    def _flush_executor_updates(self, ts_ms: int) -> None:
        updates = self.executor.drain_user_stream_updates()
        updates.extend(self.executor.collect_timeout_updates(ts_ms))
        for update in updates:
            self.telemetry.log_order(
                {
                    "order_intent_id": update.order_intent_id,
                    "order_id": update.order_id,
                    "status": update.status,
                    "symbol": update.symbol,
                    "exchange_ts_ms": update.exchange_ts_ms,
                    "detail": update.detail,
                }
            )
            if bool(update.detail.get("residual_flatten_required", False)):
                self._log_risk_event(
                    {
                        "symbol": update.symbol,
                        "event": "residual_flatten_required",
                        "order_intent_id": update.order_intent_id,
                        "exchange_ts_ms": update.exchange_ts_ms,
                    }
                )

    def process_raw_event_for_replay(self, raw_event: RawMarketEvent) -> Optional[Dict[str, object]]:
        self.metrics.inc("replay_raw_events_total")
        return self._process_raw_event_common(raw_event, assume_stream_healthy=True)

    def _process_raw_event_common(
        self,
        raw_event: RawMarketEvent,
        *,
        assume_stream_healthy: bool,
    ) -> Optional[Dict[str, object]]:
        normalized = self.normalizer.normalize(raw_event)
        if normalized is None:
            self.metrics.inc("normalized_drop_total")
            return None

        symbol = raw_event.symbol
        ts_ms = raw_event.exchange_ts_ms

        if isinstance(normalized, AggTradeEvent):
            self.features.on_agg_trade(normalized)
            self.metrics.inc("agg_trade_events_total")
            # Phase 1: check pending maker entries on every aggTrade tick.
            # Synchronous and fast — no I/O, just dict lookup + price comparison.
            self._check_and_fill_maker_entry(symbol, float(normalized.price), ts_ms)
        elif isinstance(normalized, ForceOrderEvent):
            self.features.on_force_order(normalized)
            self.metrics.inc("force_order_events_total")
        elif isinstance(normalized, MarkPriceEvent):
            self.features.on_mark_price(normalized)
            self.metrics.inc("mark_price_events_total")
        elif isinstance(normalized, DepthDeltaEvent):
            if self._sync_pending[symbol]:
                self._buffer_depth_delta(normalized)
                self.metrics.inc("depth_buffered_while_sync_total")
                if not assume_stream_healthy:
                    self._schedule_snapshot_resync(
                        symbol=symbol,
                        reason="depth_buffered_while_sync_pending",
                    )
            else:
                applied = self.book.apply_depth_delta(normalized)
                if not applied:
                    self.metrics.inc("depth_desync_events_total")
                    self._log_risk_event(
                        {
                            "symbol": symbol,
                            "event": "book_desync",
                            "reason": self.book.desync_reason(symbol),
                            "exchange_ts_ms": ts_ms,
                        }
                    )
                    if not assume_stream_healthy:
                        self._sync_pending[symbol] = True
                        self._buffer_depth_delta(normalized)
                        self._schedule_snapshot_resync(
                            symbol=symbol,
                            reason=f"depth_delta_rejected:{self.book.desync_reason(symbol)}",
                        )
                self.features.on_order_book(symbol, ts_ms, self.book)
            self.metrics.inc("depth_events_total")

        self.book.refresh_staleness(symbol, ts_ms)
        if not self.book.is_healthy(symbol):
            self._sync_pending[symbol] = True
            if not assume_stream_healthy:
                self._schedule_snapshot_resync(
                    symbol=symbol,
                    reason=f"book_not_healthy:{self.book.desync_reason(symbol)}",
                )
        else:
            self._mark_symbol_in_sync_if_ready(symbol)
        self.metrics.set_gauge(
            f"book_sync_pending_{symbol}",
            1.0 if self._sync_pending[symbol] else 0.0,
        )

        # Fast-path exit check: runs on every market event that moves mid ≥ 0.5 bps.
        # Evaluates price-based exits only (hard_stop, TP, trailing_stop, time_limit)
        # without touching the flow_reversal counter.  This prevents price gapping
        # through the trailing stop floor between decision-cycle evaluations.
        # The 0.5 bps threshold is safe: all exit levels (HARD_STOP=12, TRAIL=2,
        # TP≥21 bps) require a move well above 0.5 bps to fire.  Time-limit exits
        # are caught by the 250ms decision cycle (TIME_LIMIT_MS=60,000ms).
        if not assume_stream_healthy and self.position_manager.has_position(symbol):
            _tob_fast = self.book.top_of_book(symbol)
            if _tob_fast is not None:
                _bid_fast, _, _ask_fast, _ = _tob_fast
                _mid_fast = float((_bid_fast + _ask_fast) / 2)
                _lmid = self._last_fast_mid[symbol]
                # Skip when price is essentially unchanged — no exit level crossed.
                if _lmid > 0.0 and abs(_mid_fast - _lmid) * 10_000.0 < _lmid * 0.5:
                    return None
                self._last_fast_mid[symbol] = _mid_fast
                _fast_signal = self.position_manager.check_exits(
                    symbol=symbol,
                    ts_ms=ts_ms,
                    features={},
                    mid_price=_mid_fast,
                    price_only=True,
                )
                if _fast_signal is not None and not self._symbol_pending_submit.get(symbol, False):
                    _pos_fast = self.position_manager.close_position(symbol)
                    self.session_state.exposure_usd = max(
                        0.0,
                        self.session_state.exposure_usd - (
                            _pos_fast.qty * _pos_fast.entry_price if _pos_fast else 0.0
                        ),
                    )
                    self._symbol_pending_submit[symbol] = True
                    _fast_task = asyncio.create_task(
                        self._submit_close_order(
                            exit_signal=_fast_signal,
                            ts_ms=ts_ms,
                            best_bid=_bid_fast,
                            best_ask=_ask_fast,
                            pos=_pos_fast,
                        )
                    )

                    def _on_fast_done(t: asyncio.Task, _sym: str = symbol) -> None:
                        self._symbol_pending_submit[_sym] = False
                        if not t.cancelled() and t.exception() is not None:
                            logger.error(
                                "close_order_task_error symbol=%s: %s",
                                _sym,
                                t.exception(),
                                exc_info=t.exception(),
                            )

                    _fast_task.add_done_callback(_on_fast_done)

        last_decision = self._last_decision_ts_ms[symbol]
        if (ts_ms - last_decision) < self.config.decision_interval_ms:
            return None
        self._last_decision_ts_ms[symbol] = ts_ms

        # Drain executor order updates once per decision cycle, not per raw event.
        # Previously this was called on every WebSocket message (~100/s), adding
        # unnecessary per-event overhead on the event loop hot path.
        if not assume_stream_healthy:
            self._flush_executor_updates(ts_ms)

        stream_healthy = True
        if not assume_stream_healthy:
            stream_healthy = self.supervisor.health_snapshot().healthy

        book_healthy = self._book_ready(symbol)
        book_reason = self.book.desync_reason(symbol)
        if self._sync_pending[symbol] and book_reason == "ok":
            book_reason = "sync_pending"
        feature_snapshot = self.features.build_snapshot(symbol, ts_ms)
        # log features when liq is active, or once per 60s for baseline sampling.
        # suppressed at telemetry_log_level="summary" or "off".
        _log_level = self.config.telemetry_log_level
        if _log_level not in ("summary", "off"):
            _liq_active = float(feature_snapshot.features.get("liq_total_notional_1s") or 0.0) > 0.0
            _feature_due = (ts_ms - self._last_feature_log_ms.get(symbol, 0)) >= 60_000
            if _liq_active or _feature_due:
                self._last_feature_log_ms[symbol] = ts_ms
                self.telemetry.log_feature(
                    {
                        "episode_id": feature_snapshot.episode_id,
                        "symbol": symbol,
                        "decision_ts_ms": ts_ms,
                        "features": feature_snapshot.features,
                    }
                )

        # Full exit check: includes flow_reversal and depth_recovered.
        # Runs at the 250ms decision cadence.  If the fast-path already closed
        # the position above, has_position() returns False and this is skipped.
        if not assume_stream_healthy and self.position_manager.has_position(symbol):
            tob_exit = self.book.top_of_book(symbol)
            if tob_exit is not None:
                _bid_exit, _, _ask_exit, _ = tob_exit
                _mid_price = float((_bid_exit + _ask_exit) / 2)
                _exit_signal = self.position_manager.check_exits(
                    symbol=symbol,
                    ts_ms=ts_ms,
                    features=feature_snapshot.features,
                    mid_price=_mid_price,
                )
                if _exit_signal is not None and not self._symbol_pending_submit.get(symbol, False):
                    _pos = self.position_manager.close_position(symbol)
                    self.session_state.exposure_usd = max(
                        0.0,
                        self.session_state.exposure_usd - (_pos.qty * _pos.entry_price if _pos else 0.0),
                    )
                    self._symbol_pending_submit[symbol] = True
                    _close_task = asyncio.create_task(
                        self._submit_close_order(
                            exit_signal=_exit_signal,
                            ts_ms=ts_ms,
                            best_bid=_bid_exit,
                            best_ask=_ask_exit,
                            pos=_pos,
                        )
                    )

                    def _on_close_done(t: asyncio.Task, _sym: str = symbol) -> None:
                        self._symbol_pending_submit[_sym] = False
                        if not t.cancelled() and t.exception() is not None:
                            logger.error(
                                "close_order_task_error symbol=%s: %s",
                                _sym,
                                t.exception(),
                                exc_info=t.exception(),
                            )

                    _close_task.add_done_callback(_on_close_done)

        regime = self.regime_gate.evaluate(
            feature_snapshot,
            stream_healthy=stream_healthy,
            book_healthy=book_healthy,
            book_reason=book_reason,
        )
        decision = self.signal_engine.evaluate(feature_snapshot, regime)
        calibration_reasons = []
        if self.paper_calibration_gate is not None:
            calibration_reasons = self.paper_calibration_gate.evaluate(
                decision=decision,
                features=feature_snapshot.features,
            )
            if calibration_reasons:
                self.metrics.inc("paper_calibration_gate_block_total")
                self.metrics.inc(f"paper_calibration_gate_block_{symbol}_total")
                for reason in calibration_reasons:
                    reason_key = str(reason).replace("-", "_")
                    self.metrics.inc(f"paper_calibration_gate_block_reason_{reason_key}_total")
                    self.metrics.inc(f"paper_calibration_gate_block_{symbol}_{reason_key}_total")
                self._log_risk_event(
                    {
                        "symbol": symbol,
                        "event": "paper_calibration_gate_blocked",
                        "decision_id": decision.decision_id,
                        "exchange_ts_ms": ts_ms,
                        "reasons": calibration_reasons,
                        "score": float(decision.score or 0.0),
                        "expected_edge_bps": float(decision.expected_edge_bps or 0.0),
                    }
                )

        # ML quality gate: learned pre-trade filter (hot-reloaded from calibrator output)
        if self.ml_quality_gate is not None:
            ml_reasons = self.ml_quality_gate.evaluate(
                score=float(decision.score or 0.0),
                features=feature_snapshot.features,
                action=decision.action,
            )
            if ml_reasons:
                calibration_reasons.extend(ml_reasons)
                self.metrics.inc("ml_quality_gate_block_total")
                self._log_risk_event(
                    {
                        "symbol": symbol,
                        "event": "ml_quality_gate_blocked",
                        "decision_id": decision.decision_id,
                        "exchange_ts_ms": ts_ms,
                        "reasons": ml_reasons,
                        "score": float(decision.score or 0.0),
                    }
                )

        # Liq-notional entry filter: removed May 18 2026 to collect signal data
        # during low-liq regime. Was $200K (lowered from $300K on May 14).
        # Re-enable once forceOrder data accumulates and regime is confirmed active.

        if assume_stream_healthy:
            signal_age_ms = 0
        else:
            signal_age_ms = int(time.time() * 1000) - ts_ms
        risk = self.risk_engine.evaluate(
            decision=decision,
            regime=regime,
            snapshot=feature_snapshot,
            stream_healthy=stream_healthy,
            book_healthy=book_healthy,
            signal_age_ms=max(0, signal_age_ms),
            session=self.session_state,
            extra_reasons=calibration_reasons,
        )

        decision_payload = decision.to_dict()
        decision_payload["risk_approved"] = risk.approved
        decision_payload["risk_reasons"] = risk.reasons
        self.decisions.append(decision_payload)

        # log decisions: always for ENTER_*/partial scores; sample every 60s for pure-blocked.
        # At "summary" level: only log ENTER_* decisions.
        # At "off" level: suppress decision_log entirely.
        _pure_blocked = (
            decision.action == "NO_TRADE"
            and list(decision.score_components.keys()) == ["blocked"]
        )
        _decision_due = (ts_ms - self._last_decision_log_ms.get(symbol, 0)) >= 60_000
        _should_log_decision = False
        if _log_level == "off":
            _should_log_decision = False
        elif _log_level == "summary":
            _should_log_decision = decision.action in {"ENTER_LONG", "ENTER_SHORT"}
        else:
            _should_log_decision = not _pure_blocked or _decision_due
        if _should_log_decision:
            self._last_decision_log_ms[symbol] = ts_ms
            self.telemetry.log_decision(
                {
                    "episode_id": decision.episode_id,
                    "decision_id": decision.decision_id,
                    "symbol": decision.symbol,
                    "decision_ts_ms": decision.decision_ts_ms,
                    "action": decision.action,
                    "score": decision.score,
                    "score_components": decision.score_components,
                    "expected_edge_bps": decision.expected_edge_bps,
                    "risk_approved": risk.approved,
                    "no_trade_reason": decision.no_trade_reason,
                    "risk_reasons": risk.reasons,
                }
            )
        self.shadow_monitor.observe_decision(decision_payload)

        if risk.approved and decision.action in {"ENTER_LONG", "ENTER_SHORT"}:
            tob = self.book.top_of_book(symbol)
            if tob is not None:
                best_bid, _, best_ask, _ = tob
                if self.config.shadow_mode:
                    self.metrics.inc("shadow_execution_blocked_total")
                    decision_payload["shadow_blocked_execution"] = True
                elif assume_stream_healthy:
                    decision_payload["execution_deferred"] = True
                elif self._symbol_pending_submit.get(symbol, False):
                    # double-fill guard: block concurrent submissions per symbol
                    decision_payload["execution_deferred_pending"] = True
                elif self.position_manager.has_position(symbol):
                    # position already open — skip new entry until it is closed
                    decision_payload["execution_deferred_position_open"] = True
                else:
                    self._symbol_pending_submit[symbol] = True
                    _task = asyncio.create_task(
                        self._submit_and_log_order(
                            decision_id=decision.decision_id,
                            symbol=symbol,
                            ts_ms=ts_ms,
                            best_bid=best_bid,
                            best_ask=best_ask,
                            decision=decision,
                            score=float(decision.score or 0.0),
                            liq_notional_1s=float(
                                feature_snapshot.features.get("liq_total_notional_1s") or 0.0
                            ),
                        )
                    )

                    def _on_submit_done(t: asyncio.Task) -> None:
                        if not t.cancelled() and t.exception() is not None:
                            logger.error(
                                "order_submit_task_error symbol=%s: %s",
                                symbol,
                                t.exception(),
                                exc_info=t.exception(),
                            )

                    _task.add_done_callback(_on_submit_done)
        return decision_payload

    def shadow_summary(self) -> Dict[str, object]:
        summary = self.shadow_monitor.summary()
        return {
            "total_decisions": summary.total_decisions,
            "action_counts": summary.action_counts,
            "no_trade_reason_counts": summary.no_trade_reason_counts,
            "risk_reason_counts": summary.risk_reason_counts,
            "score_bucket_counts": summary.score_bucket_counts,
            "desync_events": summary.desync_events,
            "resync_attempts": summary.resync_attempts,
            "resync_success": summary.resync_success,
            "resync_failures": summary.resync_failures,
        }

    def paper_calibration_gate_summary(self) -> Dict[str, object]:
        gate = self.paper_calibration_gate
        if gate is None:
            return {
                "enabled": False,
                "source_path": None,
                "schema_version": None,
                "active_symbols": [],
            }
        return {
            "enabled": True,
            "source_path": str(self._paper_calibration_gate_path) if self._paper_calibration_gate_path else None,
            "schema_version": gate.schema_version,
            "active_symbols": gate.active_symbols(),
        }
