"""Paper-mode runner for persistent v1 execution-path validation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.replay.parity_runner import run_parity
from v1.telemetry.rollout_gates import RolloutGateThresholds, evaluate_rollout_gates


def _parse_symbols(value: str) -> Optional[List[str]]:
    symbols = [item.strip().upper() for item in value.split(",") if len(item.strip()) > 0]
    if len(symbols) == 0:
        return None
    return symbols


def _make_replay_run_id() -> str:
    return f"paper_{int(time.time() * 1000)}"


async def run_paper_session(
    *,
    config: RuntimeConfig,
    duration_seconds: int,
    heartbeat_seconds: int,
    report_path: Path,
    rollout_thresholds: RolloutGateThresholds,
    run_parity_check: bool,
    checkpoint_every_heartbeats: int = 10,
    max_session_hours: float = 0.0,
) -> dict:
    engine = ContinuationEngine(config)
    start_wall = int(time.time() * 1000)
    stop_event = asyncio.Event()
    calibration_gate = engine.paper_calibration_gate_summary()
    runtime_options = {
        "telemetry_log_level": str(config.telemetry_log_level),
        "capture_queue_size": int(config.capture_queue_size),
        "store_flush_interval_events": int(config.store_flush_interval_events),
        "max_book_depth_levels": int(config.max_book_depth_levels),
    }

    # Compute effective hard deadline in seconds (whichever limit fires first).
    # --max-session-hours is the "clean session" boundary used by the restart
    # wrapper to produce regular session reports; --duration-seconds is the
    # legacy limit.  0 means no limit for either.
    _max_seconds = 0
    if max_session_hours > 0:
        _max_seconds = int(max_session_hours * 3600)
    if duration_seconds > 0:
        _max_seconds = min(_max_seconds, duration_seconds) if _max_seconds > 0 else duration_seconds

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    await engine.start_live()
    heartbeat_count = 0
    try:
        while not stop_event.is_set():
            now = time.time()
            elapsed = now - (start_wall / 1000.0)
            if _max_seconds > 0 and elapsed >= _max_seconds:
                logger.info(
                    "paper_session: session limit reached (%.0f s); shutting down cleanly",
                    elapsed,
                )
                break

            await asyncio.sleep(max(1, heartbeat_seconds))
            heartbeat_count += 1
            health = engine.supervisor.health_snapshot()
            engine.telemetry.log_risk(
                {
                    "event": "paper_heartbeat",
                    "healthy": health.healthy,
                    "unhealthy_reasons": health.unhealthy_reasons,
                    "exchange_ts_ms": int(time.time() * 1000),
                    "decisions_seen": len(engine.decisions),
                }
            )

            if checkpoint_every_heartbeats > 0 and (heartbeat_count % checkpoint_every_heartbeats == 0):
                _cp_end_ts = int(time.time() * 1000)
                # evaluate_rollout_gates scans large JSONL files — run in a
                # thread so the asyncio event loop (and WS ping handler) is not
                # blocked during the scan.
                checkpoint_rollout = await asyncio.to_thread(
                    evaluate_rollout_gates,
                    telemetry_root=config.telemetry_root,
                    start_ts_ms=start_wall,
                    end_ts_ms=_cp_end_ts,
                    thresholds=rollout_thresholds,
                )
                checkpoint_out = {
                    "mode": "paper",
                    "is_final": False,
                    "start_ts_ms": start_wall,
                    "end_ts_ms": _cp_end_ts,
                    "duration_seconds": (_cp_end_ts - start_wall) / 1000.0,
                    "symbols": config.symbols,
                    "streams": config.streams,
                    "decision_interval_ms": config.decision_interval_ms,
                    "paper_calibration_gate": calibration_gate,
                    "runtime_options": runtime_options,
                    "metrics": engine.metrics.snapshot(),
                    "rollout_gates": checkpoint_rollout.to_dict(),
                    "parity": None,
                }
                report_path.parent.mkdir(parents=True, exist_ok=True)
                _payload = json.dumps(checkpoint_out, indent=2)
                await asyncio.to_thread(report_path.write_text, _payload, "utf-8")
    finally:
        await engine.stop_live()

    end_wall = int(time.time() * 1000)

    rollout = await asyncio.to_thread(
        evaluate_rollout_gates,
        telemetry_root=config.telemetry_root,
        start_ts_ms=start_wall,
        end_ts_ms=end_wall,
        thresholds=rollout_thresholds,
    )

    parity_report = None
    if run_parity_check:
        parity_report = await asyncio.to_thread(
            run_parity,
            data_root=config.data_root,
            telemetry_root=config.telemetry_root,
            symbols=config.symbols,
            start_ts_ms=start_wall,
            end_ts_ms=end_wall,
            replay_run_id=_make_replay_run_id(),
        )

    out = {
        "mode": "paper",
        "is_final": True,
        "start_ts_ms": start_wall,
        "end_ts_ms": end_wall,
        "duration_seconds": (end_wall - start_wall) / 1000.0,
        "symbols": config.symbols,
        "streams": config.streams,
        "decision_interval_ms": config.decision_interval_ms,
        "paper_calibration_gate": calibration_gate,
        "runtime_options": runtime_options,
        "metrics": engine.metrics.snapshot(),
        "rollout_gates": rollout.to_dict(),
        "parity": parity_report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _final_payload = json.dumps(out, indent=2)
    await asyncio.to_thread(report_path.write_text, _final_payload, "utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance v1 paper mode with rollout gates")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: MVP symbols)")
    parser.add_argument("--duration-seconds", type=int, default=3600, help="0 for no time limit")
    parser.add_argument(
        "--max-session-hours",
        type=float,
        default=0.0,
        help=(
            "Gracefully stop after N hours and write a final session report. "
            "Designed for use with a bash while-true restart wrapper so each "
            "restart produces a clean report boundary.  0 = no limit (use "
            "--duration-seconds instead)."
        ),
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--decision-interval-ms", type=int, default=250)
    parser.add_argument("--data-root", default="v1_data/raw")
    parser.add_argument("--telemetry-root", default="v1_data/telemetry")
    parser.add_argument("--report-path", default="v1_data/telemetry/paper_session_report.json")
    parser.add_argument("--no-parity", action="store_true", help="Skip replay parity check in session report")
    parser.add_argument("--require-gates-pass", action="store_true", help="Exit non-zero if rollout gates fail")
    parser.add_argument("--checkpoint-every-heartbeats", type=int, default=10)
    parser.add_argument(
        "--enable-calibration-gate",
        action="store_true",
        help="Enable paper-only per-asset calibration pre-trade gate",
    )
    parser.add_argument(
        "--calibration-overrides-path",
        default="",
        help="Path to per_asset_calibration_overrides.json produced by phase3 experiment runner",
    )

    # ── Performance tuning flags ───────────────────────────────────────────────
    parser.add_argument(
        "--telemetry-log-level",
        default="full",
        choices=["full", "summary", "off"],
        help=(
            "Telemetry verbosity. 'full' logs every feature snapshot and "
            "NO_TRADE decision (default). 'summary' logs only ENTER_* "
            "decisions, reducing feature_log and decision_log volume by ~95%%. "
            "'off' suppresses feature_log and decision_log entirely."
        ),
    )
    parser.add_argument(
        "--store-flush-interval-events",
        type=int,
        default=500,
        help=(
            "Flush raw partition data to OS every N appended records. "
            "Lower = more durable but more OS writes. Default 500 ≈ every 5 s."
        ),
    )
    parser.add_argument(
        "--capture-queue-size",
        type=int,
        default=2000,
        help="In-memory raw capture queue depth. Default 2000 (~20 s at 100 ev/s).",
    )
    parser.add_argument(
        "--max-book-depth-levels",
        type=int,
        default=0,
        help=(
            "Cap the local order book to N price levels per side. "
            "0 = unlimited (default). 200 is sufficient for all feature computation."
        ),
    )

    parser.add_argument(
        "--ws-local-addr",
        type=str,
        default="",
        help="Bind WebSocket connections to this source IP. Empty = OS default.",
    )
    parser.add_argument(
        "--ws-socks5-proxy",
        type=str,
        default="",
        help="Route WebSocket connections through this SOCKS5 proxy (e.g. socks5://127.0.0.1:1080). Empty = direct.",
    )

    parser.add_argument(
        "--regime-liq-optional",
        action="store_true",
        help=(
            "Regime gate soft mode: skip insufficient_liquidation_stress, "
            "liq_flow_misaligned, and liq_in_dead_zone reasons so the signal "
            "model runs on flow+depth+price alone (signal-without-liq validation)."
        ),
    )
    parser.add_argument(
        "--regime-session-optional",
        action="store_true",
        help="Regime gate soft mode: skip outside_us_session reason (allow trading any hour).",
    )
    parser.add_argument(
        "--disable-ml-quality-gate",
        action="store_true",
        help=(
            "Skip the ML quality gate entirely. Use during regime-soft validation "
            "when the loaded calibration is stale relative to the new feature regime."
        ),
    )
    parser.add_argument(
        "--risk-max-consecutive-losses",
        type=int,
        default=6,
        help=(
            "Override RiskLimits.max_consecutive_losses (default 6). Bump to a "
            "large value during paper-mode data gathering so a transient loss "
            "cluster does not lock the bot out of trading."
        ),
    )
    parser.add_argument(
        "--risk-max-symbol-consecutive-losses",
        type=int,
        default=4,
        help=(
            "Override RiskLimits.max_symbol_consecutive_losses (default 4). "
            "Same paper-mode rationale as --risk-max-consecutive-losses."
        ),
    )
    parser.add_argument(
        "--kill-switch-max-drawdown-pct",
        type=float,
        default=0.08,
        help=(
            "Override KillSwitchThresholds.max_drawdown_pct (default 0.08, i.e. 8 percent). "
            "In paper mode the gate self-locks once tripped (no closes -> drawdown "
            "cannot recover). Bump to e.g. 10.0 during paper validation to keep "
            "entries flowing for exit-calibration data."
        ),
    )

    parser.add_argument("--min-decisions", type=int, default=10)
    parser.add_argument("--min-fills", type=int, default=1)
    parser.add_argument("--min-lineage-ratio", type=float, default=0.95)
    parser.add_argument("--max-mean-realized-slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-p95-realized-slippage-bps", type=float, default=4.0)
    parser.add_argument("--max-reject-ratio", type=float, default=0.15)
    parser.add_argument("--max-residual-flatten-events", type=int, default=0)

    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    cfg = RuntimeConfig(
        symbols=symbols or RuntimeConfig().symbols,
        streams=RuntimeConfig().streams,
        data_root=Path(args.data_root),
        telemetry_root=Path(args.telemetry_root),
        decision_interval_ms=int(args.decision_interval_ms),
        execution_paper_mode=True,
        shadow_mode=False,
        enable_paper_calibration_gate=bool(args.enable_calibration_gate),
        calibration_overrides_path=(
            Path(args.calibration_overrides_path) if str(args.calibration_overrides_path).strip() else None
        ),
        telemetry_log_level=str(args.telemetry_log_level),
        store_flush_interval_events=int(args.store_flush_interval_events),
        capture_queue_size=int(args.capture_queue_size),
        max_book_depth_levels=int(args.max_book_depth_levels),
        ws_local_addr=str(args.ws_local_addr),
        ws_socks5_proxy=str(args.ws_socks5_proxy),
        regime_liq_optional=bool(args.regime_liq_optional),
        regime_session_optional=bool(args.regime_session_optional),
        disable_ml_quality_gate=bool(args.disable_ml_quality_gate),
        risk_max_consecutive_losses=int(args.risk_max_consecutive_losses),
        risk_max_symbol_consecutive_losses=int(args.risk_max_symbol_consecutive_losses),
        kill_switch_max_drawdown_pct=float(args.kill_switch_max_drawdown_pct),
    )

    thresholds = RolloutGateThresholds(
        min_decisions=int(args.min_decisions),
        min_fills=int(args.min_fills),
        min_lineage_ratio=float(args.min_lineage_ratio),
        max_mean_realized_slippage_bps=float(args.max_mean_realized_slippage_bps),
        max_p95_realized_slippage_bps=float(args.max_p95_realized_slippage_bps),
        max_reject_ratio=float(args.max_reject_ratio),
        max_residual_flatten_events=int(args.max_residual_flatten_events),
    )

    # Lock all current and future pages in RAM to prevent swap eviction.
    # Swap I/O stalls the asyncio event loop, causing WS ping timeouts and
    # gap-slippage through exit levels.  Silently skips on non-Linux or
    # when the process lacks CAP_IPC_LOCK (logged as a warning only).
    try:
        import ctypes
        import ctypes.util
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        MCL_CURRENT, MCL_FUTURE = 1, 2
        if _libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
            import errno as _errno
            logger.warning(
                "mlockall failed (errno=%d %s) — process may be paged to swap",
                ctypes.get_errno(),
                _errno.errorcode.get(ctypes.get_errno(), "?"),
            )
        else:
            logger.info("mlockall: process pages locked in RAM")
    except Exception as _exc:
        logger.info("mlockall unavailable: %s", _exc)

    # Use uvloop if installed (2–4× faster asyncio for I/O-bound WS workloads).
    # Install with: pip install uvloop
    try:
        import uvloop  # type: ignore[import]
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("uvloop event loop policy active")
    except ImportError:
        pass  # fall back to default asyncio event loop

    report = asyncio.run(
        run_paper_session(
            config=cfg,
            duration_seconds=int(args.duration_seconds),
            max_session_hours=float(args.max_session_hours),
            heartbeat_seconds=int(args.heartbeat_seconds),
            report_path=Path(args.report_path),
            rollout_thresholds=thresholds,
            run_parity_check=not bool(args.no_parity),
            checkpoint_every_heartbeats=int(args.checkpoint_every_heartbeats),
        )
    )

    print(json.dumps(report, indent=2))
    if bool(args.require_gates_pass) and (not bool(report["rollout_gates"]["passed"])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
