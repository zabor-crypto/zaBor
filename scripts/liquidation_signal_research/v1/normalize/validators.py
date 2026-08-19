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
    STREAM_AGG_TRADE: (("p",), ("q",), ("m",), ("T",)),
    STREAM_DEPTH: (("U",), ("u",), ("b",), ("a",)),
    STREAM_MARK_PRICE: (("p",), ("E",)),
}

# Requirements satisfied by any one of several alternative keys.
#
# Binance names the aggregate-trade identifier `a` on the @aggTrade stream; `t`
# is the per-trade id on the @trade stream, which this pipeline does not consume.
# Requiring `t` here rejected every real aggTrade event: validation failed, the
# normalizer returned None, and the engine counted the event under
# `normalized_drop_total` and returned before producing a decision. Both keys are
# accepted so that captures recorded under the earlier assumption still replay.
ANY_OF_FIELDS = {
    STREAM_AGG_TRADE: (("a", "t"),),
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

    for alternatives in ANY_OF_FIELDS.get(event.stream, ()):
        if not isinstance(data, dict) or not any(key in data for key in alternatives):
            errors.append(f"missing_any:{'|'.join(alternatives)}")

    if errors:
        return ValidationResult(valid=False, errors=tuple(errors))
    return ValidationResult(valid=True)
