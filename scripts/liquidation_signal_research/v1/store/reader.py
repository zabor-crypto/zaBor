"""Readers for JSONL partition store with deterministic merge ordering."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import zstandard as zstd


@dataclass(frozen=True)
class RecordRef:
    path: Path
    record: Dict[str, object]


class JsonlPartitionReader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _parse_partition_ts(path: Path) -> datetime:
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        date_str = parts.get("date", "1970-01-01")
        hour_str = parts.get("hour", "00")
        return datetime.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)

    def _partition_files(
        self,
        schema_name: str,
        symbols: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[str]] = None,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
    ) -> List[Path]:
        pattern = f"date=*/hour=*/symbol=*/stream=*/{schema_name}.jsonl.zst"
        all_files = sorted(self.root.glob(pattern))

        symbol_set = set(symbols or [])
        stream_set = set(streams or [])
        filtered: List[Path] = []
        for path in all_files:
            parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
            symbol = parts.get("symbol")
            stream = parts.get("stream")
            if symbol_set and symbol not in symbol_set:
                continue
            if stream_set and stream not in stream_set:
                continue

            partition_start = self._parse_partition_ts(path)
            partition_start_ms = int(partition_start.timestamp() * 1000)
            partition_end_ms = partition_start_ms + (60 * 60 * 1000)
            if start_ts_ms is not None and partition_end_ms < start_ts_ms:
                continue
            if end_ts_ms is not None and partition_start_ms > end_ts_ms:
                continue
            filtered.append(path)
        return filtered

    @staticmethod
    def _iter_file(path: Path) -> Iterator[Dict[str, object]]:
        dctx = zstd.ZstdDecompressor()
        with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _sort_key(record: Dict[str, object]) -> Tuple[int, int, int]:
        return (
            int(record.get("recv_mono_ts_ns", 0) or 0),
            int(record.get("ingest_seq", 0) or 0),
            int(record.get("stream_local_seq", 0) or 0),
        )

    def read_records(
        self,
        *,
        schema_name: str,
        symbols: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[str]] = None,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
    ) -> List[RecordRef]:
        files = self._partition_files(
            schema_name=schema_name,
            symbols=symbols,
            streams=streams,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
        )
        out: List[RecordRef] = []
        for path in files:
            for record in self._iter_file(path):
                exchange_ts_ms = int(record.get("exchange_ts_ms", 0) or 0)
                if start_ts_ms is not None and exchange_ts_ms < start_ts_ms:
                    continue
                if end_ts_ms is not None and exchange_ts_ms > end_ts_ms:
                    continue
                out.append(RecordRef(path=path, record=record))

        out.sort(key=lambda item: self._sort_key(item.record))
        return out

    def iter_records(
        self,
        *,
        schema_name: str,
        symbols: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[str]] = None,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
    ) -> Iterable[Dict[str, object]]:
        for item in self.read_records(
            schema_name=schema_name,
            symbols=symbols,
            streams=streams,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
        ):
            yield item.record
