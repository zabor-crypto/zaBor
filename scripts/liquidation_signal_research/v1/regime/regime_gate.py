"""Regime gate for continuation-under-liquidity-vacuum stress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from v1.contracts.event_types import FeatureSnapshotEvent, RegimeStateEvent, SCHEMA_VERSION_V1
from v1.contracts.ids import make_stable_id, EVENT_NAMESPACE


@dataclass(frozen=True)
class RegimeThresholds:
    min_liq_notional_1s: float = 20_000.0
    min_flow_imbalance_abs: float = 0.08
    max_spread_bps: float = 6.0
    min_depth_collapse_ratio: float = 0.05
    # ── V3 backtest-validated filters (Apr 15, 2026 — n=92 subset, +0.95bps) ──
    # US trading session (15-20 UTC) — tightened from 13-20 after Apr 21 grid search:
    # session_15_20UTC showed 34 V3 trades / +4.66 bps gross vs 13-20 / +4.00 bps.
    # Hours 13-15 UTC (EU close) carry more noise than peak US session (15-20 UTC).
    enable_session_filter: bool = True
    session_start_hour_utc: int = 15
    session_end_hour_utc: int = 20
    # Liquidation "dead zone" — 50K-200K shows -2.30 to -4.03 bps avg
    # Either trade <50K (small fast moves) or >200K (real cascades)
    enable_liq_dead_zone_filter: bool = True
    liq_dead_zone_lo: float = 50_000.0
    liq_dead_zone_hi: float = 200_000.0
    # Override: if 3s liq is >$300K, allow even within dead zone (true cascade)
    liq_dead_zone_override_3s: float = 300_000.0
    # ── May 20 2026: regime-soft mode for signal-without-liq validation ──
    # When enable_liq_stress_filter=False, the "insufficient_liquidation_stress"
    # reason is suppressed. Combined with enable_liq_dead_zone_filter=False and
    # enable_liq_flow_alignment_filter=False, the regime gate becomes liq-agnostic
    # and forwards decisions to the signal model on flow+depth+price alone.
    # Used to test whether the signal model produces scores ≥ 0.42 without liq input
    # during forceOrder-dormant market regimes. Default True preserves prior behaviour.
    enable_liq_stress_filter: bool = True
    enable_liq_flow_alignment_filter: bool = True


class RegimeGate:
    def __init__(self, thresholds: RegimeThresholds | None = None) -> None:
        self.thresholds = thresholds or RegimeThresholds()

    def evaluate(
        self,
        snapshot: FeatureSnapshotEvent,
        *,
        stream_healthy: bool,
        book_healthy: bool,
        book_reason: str,
    ) -> RegimeStateEvent:
        f = snapshot.features
        reasons: List[str] = []

        if not stream_healthy:
            reasons.append("stream_unhealthy")
        if not book_healthy:
            reasons.append(f"book_unhealthy:{book_reason}")

        if self.thresholds.enable_liq_stress_filter:
            if f.get("liq_total_notional_1s", 0.0) < self.thresholds.min_liq_notional_1s:
                reasons.append("insufficient_liquidation_stress")

        if abs(f.get("flow_imbalance_1s", 0.0)) < self.thresholds.min_flow_imbalance_abs:
            reasons.append("weak_flow_imbalance")

        if f.get("spread_bps", 9999.0) > self.thresholds.max_spread_bps:
            reasons.append("spread_too_wide")

        if f.get("depth_collapse_ratio", 0.0) < self.thresholds.min_depth_collapse_ratio:
            reasons.append("no_depth_collapse")

        # Liquidation must confirm flow direction (core thesis: trade WITH forced liquidation)
        liq_total = f.get("liq_total_notional_1s", 0.0)
        liq_flow_aligned = f.get("liq_flow_aligned", 0.0)
        if self.thresholds.enable_liq_flow_alignment_filter:
            if liq_total >= self.thresholds.min_liq_notional_1s and liq_flow_aligned < 0.5:
                reasons.append("liq_flow_misaligned")

        # ── V3 filters (backtest-validated on 396 trades, +0.95bps subset) ──
        # Session filter: only trade during US session (13-20 UTC)
        if self.thresholds.enable_session_filter:
            hour = (snapshot.decision_ts_ms // 3_600_000) % 24
            if not (self.thresholds.session_start_hour_utc <= hour <= self.thresholds.session_end_hour_utc):
                reasons.append("outside_us_session")

        # Liquidation dead-zone filter: 50K-200K is the noise zone
        if self.thresholds.enable_liq_dead_zone_filter:
            in_dead_zone = (
                self.thresholds.liq_dead_zone_lo <= liq_total <= self.thresholds.liq_dead_zone_hi
            )
            if in_dead_zone:
                # Allow if 3s cascade is large (real momentum despite small 1s window)
                liq_3s = f.get("liq_total_notional_3s", 0.0)
                if liq_3s < self.thresholds.liq_dead_zone_override_3s:
                    reasons.append("liq_in_dead_zone")

        tradable = len(reasons) == 0
        regime = "stress_continuation" if tradable else "blocked"

        event_id = make_stable_id(
            EVENT_NAMESPACE,
            "regime",
            snapshot.symbol,
            snapshot.decision_ts_ms,
            regime,
            ",".join(reasons),
        )
        return RegimeStateEvent(
            event_id=event_id,
            schema_version=SCHEMA_VERSION_V1,
            decision_id=event_id,
            symbol=snapshot.symbol,
            decision_ts_ms=snapshot.decision_ts_ms,
            regime=regime,
            tradable=tradable,
            no_trade_reasons=reasons,
        )
