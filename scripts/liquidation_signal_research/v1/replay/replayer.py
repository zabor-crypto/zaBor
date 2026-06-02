"""Deterministic replay runner from raw JSONL partitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from v1.contracts.event_types import RawMarketEvent
from v1.store.reader import JsonlPartitionReader


@dataclass
class ReplayResult:
    replay_run_id: str
    raw_count: int
    decision_count: int
    decisions: List[Dict[str, object]]


class DeterministicReplayer:
    def __init__(self, store_root: Path) -> None:
        self.reader = JsonlPartitionReader(store_root)

    def run(
        self,
        *,
        replay_run_id: str,
        process_raw_event: Callable[[RawMarketEvent], Optional[Dict[str, object]]],
        symbols: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[str]] = None,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
    ) -> ReplayResult:
        raw_records = self.reader.iter_records(
            schema_name="raw_market_event_v1",
            symbols=symbols,
            streams=streams,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
        )

        decision_events: List[Dict[str, object]] = []
        raw_count = 0
        for record in raw_records:
            raw_count += 1
            event = RawMarketEvent(
                event_id=str(record["event_id"]),
                schema_version=str(record["schema_version"]),
                venue=str(record["venue"]),
                stream=str(record["stream"]),
                symbol=str(record["symbol"]),
                conn_id=str(record["conn_id"]),
                exchange_ts_ms=int(record["exchange_ts_ms"]),
                recv_wall_ts_ns=int(record["recv_wall_ts_ns"]),
                recv_mono_ts_ns=int(record["recv_mono_ts_ns"]),
                ingest_seq=int(record["ingest_seq"]),
                stream_local_seq=int(record.get("stream_local_seq", 0)),
                payload_json=str(record["payload_json"]),
                payload_hash=str(record["payload_hash"]),
                parse_status=str(record["parse_status"]),
                gap_flag=bool(record["gap_flag"]),
            )
            outcome = process_raw_event(event)
            if outcome is not None:
                decision_events.append(outcome)

        return ReplayResult(
            replay_run_id=replay_run_id,
            raw_count=raw_count,
            decision_count=len(decision_events),
            decisions=decision_events,
        )
