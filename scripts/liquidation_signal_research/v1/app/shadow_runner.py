"""Shadow-mode runner for live decision-path monitoring without execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig


def _parse_symbols(value: str) -> Optional[List[str]]:
    symbols = [item.strip().upper() for item in value.split(",") if len(item.strip()) > 0]
    if len(symbols) == 0:
        return None
    return symbols


async def run_shadow_session(
    *,
    config: RuntimeConfig,
    duration_seconds: int,
    heartbeat_seconds: int,
    report_path: Path,
) -> dict:
    engine = ContinuationEngine(config)
    start_wall = int(time.time() * 1000)
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    await engine.start_live()
    try:
        while not stop_event.is_set():
            now = time.time()
            elapsed = now - (start_wall / 1000.0)
            if duration_seconds > 0 and elapsed >= duration_seconds:
                break

            await asyncio.sleep(max(1, heartbeat_seconds))
            health = engine.supervisor.health_snapshot()
            engine.telemetry.log_risk(
                {
                    "event": "shadow_heartbeat",
                    "healthy": health.healthy,
                    "unhealthy_reasons": health.unhealthy_reasons,
                    "exchange_ts_ms": int(time.time() * 1000),
                    "decisions_seen": len(engine.decisions),
                }
            )
    finally:
        await engine.stop_live()

    end_wall = int(time.time() * 1000)
    summary = engine.shadow_summary()
    out = {
        "mode": "shadow",
        "start_ts_ms": start_wall,
        "end_ts_ms": end_wall,
        "duration_seconds": (end_wall - start_wall) / 1000.0,
        "symbols": config.symbols,
        "streams": config.streams,
        "decision_interval_ms": config.decision_interval_ms,
        "metrics": engine.metrics.snapshot(),
        "shadow_summary": summary,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance v1 shadow mode")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: MVP symbols)")
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--decision-interval-ms", type=int, default=250)
    parser.add_argument("--data-root", default="v1_data/raw")
    parser.add_argument("--telemetry-root", default="v1_data/telemetry")
    parser.add_argument("--report-path", default="v1_data/telemetry/shadow_session_report.json")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    cfg = RuntimeConfig(
        symbols=symbols or RuntimeConfig().symbols,
        streams=RuntimeConfig().streams,
        data_root=Path(args.data_root),
        telemetry_root=Path(args.telemetry_root),
        decision_interval_ms=int(args.decision_interval_ms),
        execution_paper_mode=True,
        shadow_mode=True,
    )

    report = asyncio.run(
        run_shadow_session(
            config=cfg,
            duration_seconds=int(args.duration_seconds),
            heartbeat_seconds=int(args.heartbeat_seconds),
            report_path=Path(args.report_path),
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
