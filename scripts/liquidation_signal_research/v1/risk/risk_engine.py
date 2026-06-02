"""Pre-trade and session risk gates for continuation v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from v1.contracts.event_types import (
    DecisionIntentEvent,
    FeatureSnapshotEvent,
    RegimeStateEvent,
    RiskGateEvent,
    SCHEMA_VERSION_V1,
)
from v1.contracts.ids import EVENT_NAMESPACE, make_stable_id
from v1.risk.kill_switch import KillSwitch, KillSwitchThresholds


@dataclass
class SessionRiskState:
    exposure_usd: float = 0.0
    nav_usd: float = 0.0
    daily_pnl_usd: float = 0.0
    session_pnl_usd: float = 0.0
    peak_nav_usd: float = 0.0
    consecutive_losses: int = 0
    symbol_consecutive_losses: Dict[str, int] = field(default_factory=dict)
    consecutive_rejects: int = 0
    rolling_error_rate: float = 0.0
    drawdown_pct: float = 0.0
    telemetry_alive: bool = True


@dataclass(frozen=True)
class RiskLimits:
    max_spread_bps: float = 8.0
    max_exposure_fraction: float = 0.5
    max_daily_loss_fraction: float = 0.03
    max_session_net_loss_fraction: float = 0.03
    max_consecutive_losses: int = 6
    max_symbol_consecutive_losses: int = 4
    min_signal_freshness_ms: int = 1200
    # Binance UM taker-taker round-trip = ~8 bps (0.04% × 2); 3 bps safety buffer.
    fee_plus_buffer_bps: float = 8.0


class RiskEngine:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        kill_switch_thresholds: KillSwitchThresholds | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch = KillSwitch(kill_switch_thresholds)

    def evaluate(
        self,
        *,
        decision: DecisionIntentEvent,
        regime: RegimeStateEvent,
        snapshot: FeatureSnapshotEvent,
        stream_healthy: bool,
        book_healthy: bool,
        signal_age_ms: int,
        session: SessionRiskState,
        extra_reasons: Optional[Sequence[str]] = None,
    ) -> RiskGateEvent:
        reasons: List[str] = []

        kill = self.kill_switch.evaluate(
            consecutive_rejects=session.consecutive_rejects,
            rolling_error_rate=session.rolling_error_rate,
            drawdown_pct=session.drawdown_pct,
            telemetry_alive=session.telemetry_alive,
        )
        if kill.active:
            reasons.append(f"kill_switch:{kill.reason}")

        if not stream_healthy:
            reasons.append("mandatory_stream_unhealthy")

        if not book_healthy:
            reasons.append("book_desynced_or_stale")

        if not regime.tradable:
            reasons.append("regime_blocked")

        if decision.action == "NO_TRADE":
            reasons.append("decision_no_trade")

        spread_bps = float(snapshot.features.get("spread_bps", 9999.0))
        if spread_bps > self.limits.max_spread_bps:
            reasons.append("spread_over_cap")

        if signal_age_ms > self.limits.min_signal_freshness_ms:
            reasons.append("signal_stale")

        min_required_edge = self.limits.fee_plus_buffer_bps + float(decision.slippage_budget_bps)
        if decision.expected_edge_bps < min_required_edge:
            reasons.append("expected_edge_below_cost")

        if session.nav_usd > 0:
            if (session.exposure_usd / session.nav_usd) > self.limits.max_exposure_fraction:
                reasons.append("exposure_cap_reached")

            if (-session.daily_pnl_usd / session.nav_usd) > self.limits.max_daily_loss_fraction:
                reasons.append("daily_loss_cap_reached")

            if (-session.session_pnl_usd / session.nav_usd) > self.limits.max_session_net_loss_fraction:
                reasons.append("session_net_loss_cap_reached")

        if session.consecutive_losses >= self.limits.max_consecutive_losses:
            reasons.append("consecutive_losses_cap")

        symbol = str(decision.symbol or "").upper()
        symbol_losses = int(session.symbol_consecutive_losses.get(symbol, 0))
        if symbol_losses >= self.limits.max_symbol_consecutive_losses:
            reasons.append("symbol_consecutive_losses_cap")

        if extra_reasons:
            for reason in extra_reasons:
                text = str(reason or "").strip()
                if text:
                    reasons.append(text)

        if reasons:
            deduped: List[str] = []
            seen = set()
            for reason in reasons:
                if reason in seen:
                    continue
                seen.add(reason)
                deduped.append(reason)
            reasons = deduped

        approved = len(reasons) == 0
        event_id = make_stable_id(
            EVENT_NAMESPACE,
            "risk_gate",
            decision.decision_id,
            approved,
            ",".join(reasons),
        )

        return RiskGateEvent(
            event_id=event_id,
            schema_version=SCHEMA_VERSION_V1,
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            decision_ts_ms=decision.decision_ts_ms,
            approved=approved,
            reasons=reasons,
        )
