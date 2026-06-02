"""Partition manifest helpers for immutable JSONL event store."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PartitionManifestRecord:
    created_at: str
    schema_name: str
    schema_version: str
    symbol: str
    stream: str
    partition_date: str
    partition_hour: str
    file_path: str
    metadata: Dict[str, object]


class StoreManifest:
    """Append-only manifest log for created partitions and schema metadata."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.jsonl"
        self._lock = threading.Lock()

    def append(
        self,
        *,
        schema_name: str,
        schema_version: str,
        symbol: str,
        stream: str,
        partition_date: str,
        partition_hour: str,
        file_path: Path,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        record = PartitionManifestRecord(
            created_at=datetime.now(timezone.utc).isoformat(),
            schema_name=schema_name,
            schema_version=schema_version,
            symbol=symbol,
            stream=stream,
            partition_date=partition_date,
            partition_hour=partition_hour,
            file_path=str(file_path),
            metadata=metadata or {},
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), separators=(",", ":"), ensure_ascii=True) + "\n")
