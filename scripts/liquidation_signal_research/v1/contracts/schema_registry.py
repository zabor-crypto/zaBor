"""Schema registry for raw and normalized event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set

from .event_types import SCHEMA_VERSION_V1


@dataclass(frozen=True)
class SchemaDefinition:
    name: str
    version: str
    required_fields: Set[str]


class SchemaRegistry:
    """Simple schema registry with strict required-field validation."""

    def __init__(self) -> None:
        self._schemas: Dict[str, SchemaDefinition] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        self.register(
            SchemaDefinition(
                name="raw_market_event_v1",
                version=SCHEMA_VERSION_V1,
                required_fields={
                    "event_id",
                    "schema_version",
                    "venue",
                    "stream",
                    "symbol",
                    "conn_id",
                    "exchange_ts_ms",
                    "recv_wall_ts_ns",
                    "recv_mono_ts_ns",
                    "ingest_seq",
                    "stream_local_seq",
                    "payload_json",
                    "payload_hash",
                    "parse_status",
                    "gap_flag",
                },
            )
        )
        self.register(
            SchemaDefinition(
                name="feature_snapshot_v1",
                version=SCHEMA_VERSION_V1,
                required_fields={
                    "event_id",
                    "schema_version",
                    "episode_id",
                    "decision_ts_ms",
                    "symbol",
                    "features",
                },
            )
        )
        self.register(
            SchemaDefinition(
                name="decision_intent_v1",
                version=SCHEMA_VERSION_V1,
                required_fields={
                    "event_id",
                    "schema_version",
                    "episode_id",
                    "decision_id",
                    "symbol",
                    "decision_ts_ms",
                    "action",
                    "score",
                    "score_components",
                    "expected_edge_bps",
                    "slippage_budget_bps",
                    "no_trade_reason",
                },
            )
        )

    def register(self, schema: SchemaDefinition) -> None:
        self._schemas[schema.name] = schema

    def get(self, name: str) -> SchemaDefinition:
        return self._schemas[name]

    def list_names(self) -> Iterable[str]:
        return self._schemas.keys()

    def validate(self, schema_name: str, payload: Mapping[str, object]) -> None:
        schema = self.get(schema_name)
        missing = schema.required_fields.difference(payload.keys())
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Schema {schema_name} missing required fields: {fields}")
        version = payload.get("schema_version")
        if version != schema.version:
            raise ValueError(
                f"Schema {schema_name} version mismatch: expected={schema.version}, got={version}"
            )


DEFAULT_SCHEMA_REGISTRY = SchemaRegistry()
