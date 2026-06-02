"""Rollout gate evaluation from session telemetry logs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class RolloutGateThresholds:
    min_decisions: int = 10
    min_fills: int = 1
    # Order-lineage completeness target. Decision logs intentionally sample some
    # pure NO_TRADE paths, so 0.999 is unrealistically strict in production.
    min_lineage_ratio: float = 0.95
    max_mean_realized_slippage_bps: float = 2.0
    max_p95_realized_slippage_bps: float = 4.0
    max_reject_ratio: float = 0.15
    max_residual_flatten_events: int = 0


@dataclass(frozen=True)
class RolloutMetrics:
    decision_count: int
    enter_decision_count: int
    order_intent_count: int
    fill_count: int
    reject_count: int
    reject_ratio: float
    residual_flatten_events: int
    lineage_complete_decisions: int
    lineage_ratio: float
    mean_realized_slippage_bps: float
    p95_realized_slippage_bps: float
    mean_expected_edge_bps: float
    mean_edge_after_slippage_bps: float


@dataclass(frozen=True)
class RolloutGateResult:
    passed: bool
    failed_gates: List[str]
    metrics: RolloutMetrics
    thresholds: RolloutGateThresholds

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "failed_gates": self.failed_gates,
            "metrics": asdict(self.metrics),
            "thresholds": asdict(self.thresholds),
        }


@dataclass
class _OrderAggregate:
    decision_id: Optional[str] = None
    statuses: List[str] = None  # type: ignore[assignment]
    realized_slippage_bps: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.statuses is None:
            self.statuses = []
        if self.realized_slippage_bps is None:
            self.realized_slippage_bps = []


def _jsonl_rows(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if len(line) == 0:
                continue
            rows.append(json.loads(line))
    return rows


def _ts_in_window(payload: Mapping[str, object], ts_field: str, start_ts_ms: int, end_ts_ms: int) -> bool:
    ts = payload.get(ts_field)
    if ts is None:
        return False
    ts_ms = int(ts)
    return start_ts_ms <= ts_ms <= end_ts_ms


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return float(num) / float(den)


def _percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil(q * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _extract_realized_slippage_bps(detail: Mapping[str, object]) -> Optional[float]:
    raw = detail.get("realized_slippage_bps")
    if raw is not None:
        val = float(raw)
        return max(0.0, val)

    try:
        side = str(detail.get("state_side", ""))
        avg_fill = float(detail.get("state_avg_fill_price", "0") or 0.0)
        arrival = float(detail.get("arrival_price", "0") or 0.0)
        limit = float(detail.get("state_limit_price", "0") or 0.0)
    except (TypeError, ValueError):
        return None

    baseline = arrival if arrival > 0 else limit
    if baseline <= 0 or avg_fill <= 0:
        return None

    if side == "BUY":
        return max(0.0, ((avg_fill - baseline) / baseline) * 10000.0)
    if side == "SELL":
        return max(0.0, ((baseline - avg_fill) / baseline) * 10000.0)
    return None


def compute_rollout_metrics(
    *,
    telemetry_root: Path,
    start_ts_ms: int,
    end_ts_ms: int,
) -> RolloutMetrics:
    telemetry_root = Path(telemetry_root)

    decision_rows = [
        row
        for row in _jsonl_rows(telemetry_root / "decision_log.jsonl")
        if _ts_in_window(row, "decision_ts_ms", start_ts_ms, end_ts_ms)
    ]
    feature_rows = [
        row
        for row in _jsonl_rows(telemetry_root / "feature_log.jsonl")
        if _ts_in_window(row, "decision_ts_ms", start_ts_ms, end_ts_ms)
    ]
    order_rows = [
        row
        for row in _jsonl_rows(telemetry_root / "order_lifecycle_log.jsonl")
        if _ts_in_window(row, "exchange_ts_ms", start_ts_ms, end_ts_ms)
    ]
    risk_rows = [
        row
        for row in _jsonl_rows(telemetry_root / "risk_health_log.jsonl")
        if _ts_in_window(row, "exchange_ts_ms", start_ts_ms, end_ts_ms)
    ]

    feature_episodes = {
        str(row.get("episode_id", ""))
        for row in feature_rows
        if len(str(row.get("episode_id", ""))) > 0
    }

    decisions_by_id: Dict[str, Dict[str, object]] = {}
    for row in decision_rows:
        decision_id = str(row.get("decision_id", ""))
        if len(decision_id) == 0:
            continue
        decisions_by_id[decision_id] = row

    orders_by_intent: Dict[str, _OrderAggregate] = {}
    order_intents_by_decision: Dict[str, List[str]] = {}

    for row in order_rows:
        order_intent_id = str(row.get("order_intent_id", ""))
        if len(order_intent_id) == 0:
            continue

        aggregate = orders_by_intent.get(order_intent_id)
        if aggregate is None:
            aggregate = _OrderAggregate()
            orders_by_intent[order_intent_id] = aggregate

        decision_id = str(row.get("decision_id", "") or "")
        if len(decision_id) > 0:
            aggregate.decision_id = decision_id
            order_intents_by_decision.setdefault(decision_id, []).append(order_intent_id)

        status = str(row.get("status", ""))
        if len(status) > 0:
            aggregate.statuses.append(status)

        detail = row.get("detail")
        if isinstance(detail, dict):
            slippage = _extract_realized_slippage_bps(detail)
            if slippage is not None:
                aggregate.realized_slippage_bps.append(slippage)

    filled_intents = [
        oid
        for oid, agg in orders_by_intent.items()
        if any(status == "FILLED" for status in agg.statuses)
    ]
    rejected_intents = [
        oid
        for oid, agg in orders_by_intent.items()
        if any(status == "REJECTED" for status in agg.statuses)
    ]

    residual_flatten_events = sum(
        1
        for row in risk_rows
        if str(row.get("event", "")) == "residual_flatten_required"
    )

    # Lineage completeness is measured on order intents (execution lineage),
    # not total decisions. Decision logging intentionally samples pure blocked
    # paths, so using decision_count as denominator creates false gate failures.
    lineage_complete = 0
    enter_count = 0
    for decision in decisions_by_id.values():
        action = str(decision.get("action", ""))
        if action in {"ENTER_LONG", "ENTER_SHORT"}:
            enter_count += 1

    for aggregate in orders_by_intent.values():
        decision_id = aggregate.decision_id
        if decision_id is None:
            continue
        decision = decisions_by_id.get(decision_id)
        if decision is None:
            continue
        episode_id = str(decision.get("episode_id", ""))
        if episode_id in feature_episodes:
            lineage_complete += 1

    realized_slippages: List[float] = []
    expected_edges: List[float] = []
    net_edges: List[float] = []

    for order_intent_id in filled_intents:
        aggregate = orders_by_intent[order_intent_id]
        if len(aggregate.realized_slippage_bps) == 0:
            continue
        realized = float(aggregate.realized_slippage_bps[-1])
        realized_slippages.append(realized)

        decision_id = aggregate.decision_id
        if decision_id and decision_id in decisions_by_id:
            expected = float(decisions_by_id[decision_id].get("expected_edge_bps", 0.0) or 0.0)
            expected_edges.append(expected)
            net_edges.append(expected - realized)

    mean_realized_slippage = fmean(realized_slippages) if len(realized_slippages) > 0 else 0.0
    p95_realized_slippage = _percentile(realized_slippages, 0.95)
    mean_expected_edge = fmean(expected_edges) if len(expected_edges) > 0 else 0.0
    mean_net_edge = fmean(net_edges) if len(net_edges) > 0 else 0.0

    decision_count = len(decisions_by_id)
    order_intent_count = len(orders_by_intent)
    fill_count = len(filled_intents)
    reject_count = len(rejected_intents)

    return RolloutMetrics(
        decision_count=decision_count,
        enter_decision_count=enter_count,
        order_intent_count=order_intent_count,
        fill_count=fill_count,
        reject_count=reject_count,
        reject_ratio=_safe_ratio(reject_count, order_intent_count),
        residual_flatten_events=residual_flatten_events,
        lineage_complete_decisions=lineage_complete,
        lineage_ratio=_safe_ratio(lineage_complete, order_intent_count),
        mean_realized_slippage_bps=mean_realized_slippage,
        p95_realized_slippage_bps=p95_realized_slippage,
        mean_expected_edge_bps=mean_expected_edge,
        mean_edge_after_slippage_bps=mean_net_edge,
    )


def evaluate_rollout_gates(
    *,
    telemetry_root: Path,
    start_ts_ms: int,
    end_ts_ms: int,
    thresholds: RolloutGateThresholds | None = None,
) -> RolloutGateResult:
    thresholds = thresholds or RolloutGateThresholds()
    metrics = compute_rollout_metrics(
        telemetry_root=Path(telemetry_root),
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
    )

    failed: List[str] = []

    if metrics.decision_count < thresholds.min_decisions:
        failed.append("decision_count_below_min")
    if metrics.fill_count < thresholds.min_fills:
        failed.append("fill_count_below_min")
    if metrics.lineage_ratio < thresholds.min_lineage_ratio:
        failed.append("lineage_ratio_below_min")
    if metrics.mean_realized_slippage_bps > thresholds.max_mean_realized_slippage_bps:
        failed.append("mean_realized_slippage_above_max")
    if metrics.p95_realized_slippage_bps > thresholds.max_p95_realized_slippage_bps:
        failed.append("p95_realized_slippage_above_max")
    if metrics.reject_ratio > thresholds.max_reject_ratio:
        failed.append("reject_ratio_above_max")
    if metrics.residual_flatten_events > thresholds.max_residual_flatten_events:
        failed.append("residual_flatten_events_above_max")

    return RolloutGateResult(
        passed=len(failed) == 0,
        failed_gates=failed,
        metrics=metrics,
        thresholds=thresholds,
    )
