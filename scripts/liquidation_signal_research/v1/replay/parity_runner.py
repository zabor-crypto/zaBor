"""CLI utility to run replay/live parity checks for decision streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from v1.app.engine import ContinuationEngine
from v1.app.runtime import RuntimeConfig
from v1.replay.parity_checker import compare_decision_streams
from v1.replay.replayer import DeterministicReplayer


def _load_live_decisions(path: Path, start_ts_ms: Optional[int], end_ts_ms: Optional[int]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if len(line) == 0:
                continue
            payload = json.loads(line)
            ts = payload.get("decision_ts_ms")
            if ts is not None:
                ts_int = int(ts)
                if start_ts_ms is not None and ts_int < start_ts_ms:
                    continue
                if end_ts_ms is not None and ts_int > end_ts_ms:
                    continue
            out.append(payload)
    return out


def run_parity(
    *,
    data_root: Path,
    telemetry_root: Path,
    symbols: Optional[List[str]],
    start_ts_ms: Optional[int],
    end_ts_ms: Optional[int],
    replay_run_id: str,
) -> Dict[str, Any]:
    replay_telemetry_root = telemetry_root / "_replay_runs" / replay_run_id
    config = RuntimeConfig(
        symbols=symbols or RuntimeConfig().symbols,
        data_root=data_root,
        telemetry_root=replay_telemetry_root,
        execution_paper_mode=True,
    )
    engine = ContinuationEngine(config)

    replayer = DeterministicReplayer(data_root)
    replay_result = replayer.run(
        replay_run_id=replay_run_id,
        process_raw_event=engine.process_raw_event_for_replay,
        symbols=symbols,
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
    )

    live_decisions = _load_live_decisions(
        telemetry_root / "decision_log.jsonl",
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
    )
    parity = compare_decision_streams(live_decisions, replay_result.decisions)

    return {
        "replay_run_id": replay_run_id,
        "data_root": str(data_root),
        "telemetry_root": str(telemetry_root),
        "replay_telemetry_root": str(replay_telemetry_root),
        "symbols": symbols,
        "start_ts_ms": start_ts_ms,
        "end_ts_ms": end_ts_ms,
        "raw_count": replay_result.raw_count,
        "replay_decision_count": replay_result.decision_count,
        "live_decision_count": len(live_decisions),
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decision-stream replay parity check")
    parser.add_argument("--data-root", required=True, help="Raw store root path")
    parser.add_argument("--telemetry-root", required=True, help="Telemetry root path")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list")
    parser.add_argument("--start-ts-ms", type=int, default=None)
    parser.add_argument("--end-ts-ms", type=int, default=None)
    parser.add_argument("--replay-run-id", required=True)
    parser.add_argument("--out", required=True, help="Output JSON report path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if len(s.strip()) > 0]
    if len(symbols) == 0:
        symbols = None

    report = run_parity(
        data_root=Path(args.data_root),
        telemetry_root=Path(args.telemetry_root),
        symbols=symbols,
        start_ts_ms=args.start_ts_ms,
        end_ts_ms=args.end_ts_ms,
        replay_run_id=args.replay_run_id,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
