"""Continuation score engine (liquidity-vacuum stress) for v1."""

from __future__ import annotations

from dataclasses import dataclass

from v1.contracts.event_types import (
    DecisionIntentEvent,
    FeatureSnapshotEvent,
    RegimeStateEvent,
    SCHEMA_VERSION_V1,
)
from v1.contracts.ids import EVENT_NAMESPACE, make_decision_id, make_stable_id


@dataclass(frozen=True)
class ContinuationScoreConfig:
    config_version: str = "v1.0.0"
    # ── Entry threshold (score floor) ──────────────────────────────────────────
    # Raised 0.35 → 0.42: grid-search on 451 trades showed 0.42+ combinations
    # outperform by ~2 bps gross.  Score bucket analysis (calibration replay, 586
    # trades) also showed 0.35-0.45 best, but 0.42 floor trims the weakest tail
    # while keeping sufficient trade volume.
    entry_threshold: float = 0.42
    # ── Score ceiling ─────────────────────────────────────────────────────────
    # High-score trades (>0.55) under-perform: calibration replay showed
    # 0.55-0.65 bucket → -1.57 bps, 0.65+ → -4.04 bps at 180s.
    # V3 grid search confirmed score_band_0.42_0.55 as best scoring subset
    # (33 V3 trades, 85% WR, +5.1 bps gross with best exits).
    score_ceiling: float = 0.55
    # ── ETH LONG gate (Phase 2) ────────────────────────────────────────────────
    # V3 live data: ETH LONG 93% WR, +6.65 bps gross (14 trades).
    # ETH SHORT blocks based on structural asymmetry: SHORT liquidation cascades
    # on ETH reliably bounce; LONG cascades do not have the same recovery pattern.
    # Set to a frozenset of symbols where SHORT direction is blocked.
    long_only_symbols: frozenset = frozenset({"ETHUSDT"})
    max_spread_bps: float = 8.0
    slippage_budget_bps: float = 3.0


class ContinuationScoreEngine:
    def __init__(self, config: ContinuationScoreConfig | None = None) -> None:
        self.config = config or ContinuationScoreConfig()

    def evaluate(self, snapshot: FeatureSnapshotEvent, regime: RegimeStateEvent) -> DecisionIntentEvent:
        features = snapshot.features

        feature_hash = make_stable_id(
            EVENT_NAMESPACE,
            "feature-hash",
            snapshot.symbol,
            snapshot.decision_ts_ms,
            round(features.get("flow_imbalance_1s", 0.0), 6),
            round(features.get("depth_collapse_ratio", 0.0), 6),
            round(features.get("spread_bps", 0.0), 6),
            round(features.get("ret_500_bps", 0.0), 6),
        )
        decision_id = make_decision_id(
            snapshot.symbol,
            snapshot.decision_ts_ms,
            feature_hash,
            self.config.config_version,
        )

        if not regime.tradable:
            return DecisionIntentEvent(
                event_id=decision_id,
                schema_version=SCHEMA_VERSION_V1,
                episode_id=snapshot.episode_id,
                decision_id=decision_id,
                symbol=snapshot.symbol,
                decision_ts_ms=snapshot.decision_ts_ms,
                action="NO_TRADE",
                score=0.0,
                score_components={"blocked": 1.0},
                expected_edge_bps=0.0,
                slippage_budget_bps=self.config.slippage_budget_bps,
                no_trade_reason=";".join(regime.no_trade_reasons),
            )

        flow = float(features.get("flow_imbalance_1s", 0.0))
        flow_3s = float(features.get("flow_imbalance_3s", 0.0))
        liq_imbalance = float(features.get("liq_imbalance_1s", 0.0))
        liq_flow_aligned = float(features.get("liq_flow_aligned", 0.0))
        flow_accel = float(features.get("flow_acceleration", 0.0))
        liq_momentum = float(features.get("liq_momentum", 0.0))

        # Direction: prefer liq_imbalance (strongest signal), fall back to 3s flow, then 1s flow
        if abs(liq_imbalance) > 0.3:
            direction_sign = 1.0 if liq_imbalance > 0 else -1.0
        elif abs(flow_3s) > 0.05:
            direction_sign = 1.0 if flow_3s >= 0 else -1.0
        else:
            direction_sign = 1.0 if flow >= 0 else -1.0

        depth = float(features.get("depth_collapse_ratio", 0.0))
        spread_bps = float(features.get("spread_bps", 0.0))
        impact = direction_sign * float(features.get("ret_500_bps", 0.0)) / 10.0
        failed_recovery = float(features.get("failed_recovery", 0.0))
        liq_total = float(features.get("liq_total_notional_1s", 0.0))

        import math as _math
        liq_norm = min(1.0, _math.log1p(liq_total / 20_000.0) / _math.log1p(7.5))
        spread_penalty = min(1.0, spread_bps / max(1.0, self.config.max_spread_bps))

        # Cascade exhaustion penalty: negative momentum means liq is decelerating
        # (the cascade may be ending — entering now catches the reversal, not continuation)
        exhaustion_penalty = max(0.0, -liq_momentum) if liq_momentum < 0 else 0.0

        components = {
            "flow": abs(flow) * 0.20,                        # reduced from 0.35 — less 1s noise
            "flow_3s": abs(flow_3s) * 0.10,                  # NEW: multi-tf direction stability
            "liq_direction": liq_flow_aligned * 0.15,         # NEW: liquidation confirms flow
            "depth_collapse": depth * 0.15,                   # reduced from 0.20
            "impact": impact * 0.15,                          # reduced from 0.20
            "failed_recovery": failed_recovery * 0.10,        # reduced from 0.15
            "flow_acceleration": flow_accel * 0.05,           # positive = building, negative = exhausting
            "liq_context": liq_norm * 0.10,                   # unchanged
            "spread_penalty": -spread_penalty * 0.15,         # reduced from -0.20 to make room
            "exhaustion_penalty": -exhaustion_penalty * 0.05, # NEW: penalize decelerating cascades
        }

        score = sum(components.values())
        if score < self.config.entry_threshold:
            return DecisionIntentEvent(
                event_id=decision_id,
                schema_version=SCHEMA_VERSION_V1,
                episode_id=snapshot.episode_id,
                decision_id=decision_id,
                symbol=snapshot.symbol,
                decision_ts_ms=snapshot.decision_ts_ms,
                action="NO_TRADE",
                score=score,
                score_components=components,
                expected_edge_bps=max(0.0, score * 6.0),
                slippage_budget_bps=self.config.slippage_budget_bps,
                no_trade_reason="score_below_threshold",
            )

        # Score ceiling: high-score trades consistently under-perform (calibration
        # replay: 0.55-0.65 → -1.57 bps, 0.65+ → -4.04 bps).  Block them.
        if score > self.config.score_ceiling:
            return DecisionIntentEvent(
                event_id=decision_id,
                schema_version=SCHEMA_VERSION_V1,
                episode_id=snapshot.episode_id,
                decision_id=decision_id,
                symbol=snapshot.symbol,
                decision_ts_ms=snapshot.decision_ts_ms,
                action="NO_TRADE",
                score=score,
                score_components=components,
                expected_edge_bps=0.0,
                slippage_budget_bps=self.config.slippage_budget_bps,
                no_trade_reason="score_above_ceiling",
            )

        action = "ENTER_LONG" if direction_sign > 0 else "ENTER_SHORT"

        # Phase 2 — ETH LONG gate: block SHORT entries on symbols where SHORT
        # liquidation cascades historically do not produce reliable continuation
        # (V3 live: ETH LONG 93% WR / +6.65 bps; ETH SHORT sample too small and
        # directionally inconsistent).
        if action == "ENTER_SHORT" and snapshot.symbol in self.config.long_only_symbols:
            return DecisionIntentEvent(
                event_id=decision_id,
                schema_version=SCHEMA_VERSION_V1,
                episode_id=snapshot.episode_id,
                decision_id=decision_id,
                symbol=snapshot.symbol,
                decision_ts_ms=snapshot.decision_ts_ms,
                action="NO_TRADE",
                score=score,
                score_components=components,
                expected_edge_bps=0.0,
                slippage_budget_bps=self.config.slippage_budget_bps,
                no_trade_reason="long_only_gate",
            )
        # Calibrated from MFE data: score 0.4 → 12bps, score 0.6 → 18bps, score 0.8 → 24bps
        expected_edge_bps = max(0.0, score * 30.0)
        return DecisionIntentEvent(
            event_id=decision_id,
            schema_version=SCHEMA_VERSION_V1,
            episode_id=snapshot.episode_id,
            decision_id=decision_id,
            symbol=snapshot.symbol,
            decision_ts_ms=snapshot.decision_ts_ms,
            action=action,
            score=score,
            score_components=components,
            expected_edge_bps=expected_edge_bps,
            slippage_budget_bps=self.config.slippage_budget_bps,
            no_trade_reason=None,
        )
