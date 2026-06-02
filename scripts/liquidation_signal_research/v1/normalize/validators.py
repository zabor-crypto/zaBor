"""Raw event validators for stream-specific required fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from v1.contracts.event_types import (
    STREAM_AGG_TRADE,
    STREAM_DEPTH,
    STREAM_FORCE_ORDER,
    STREAM_MARK_PRICE,
    RawMarketEvent,
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: Tuple[str, ...] = ()


REQUIRED_FIELDS = {
    STREAM_FORCE_ORDER: (("o",),),
    STREAM_AGG_TRADE: (("t",), ("p",), ("q",), ("m",), ("T",)),
    STREAM_DEPTH: (("U",), ("u",), ("b",), ("a",)),
    STREAM_MARK_PRICE: (("p",), ("E",)),
}


def validate_raw_event(event: RawMarketEvent) -> ValidationResult:
    if event.parse_status != "ok":
        return ValidationResult(valid=False, errors=("parse_status_not_ok",))

    try:
        payload = json.loads(event.payload_json)
    except json.JSONDecodeError:
        return ValidationResult(valid=False, errors=("payload_json_decode_error",))

    data = payload.get("data", payload)
    requirements = REQUIRED_FIELDS.get(event.stream)
    if not requirements:
        return ValidationResult(valid=False, errors=("unsupported_stream", event.stream))

    errors: List[str] = []
    for path in requirements:
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                errors.append(f"missing:{'.'.join(path)}")
                break
            current = current[key]

    if errors:
        return ValidationResult(valid=False, errors=tuple(errors))
    return ValidationResult(valid=True)
