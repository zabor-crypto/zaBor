"""Session-level kill-switch policy for v1 runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KillSwitchState:
    active: bool = False
    reason: str = ""


@dataclass(frozen=True)
class KillSwitchThresholds:
    max_consecutive_rejects: int = 5
    max_error_rate: float = 0.2
    max_drawdown_pct: float = 0.08


class KillSwitch:
    def __init__(self, thresholds: KillSwitchThresholds | None = None) -> None:
        self.thresholds = thresholds or KillSwitchThresholds()
        self.state = KillSwitchState()

    def evaluate(
        self,
        *,
        consecutive_rejects: int,
        rolling_error_rate: float,
        drawdown_pct: float,
        telemetry_alive: bool,
    ) -> KillSwitchState:
        if not telemetry_alive:
            self.state = KillSwitchState(active=True, reason="telemetry_blindness")
            return self.state

        if consecutive_rejects >= self.thresholds.max_consecutive_rejects:
            self.state = KillSwitchState(active=True, reason="reject_cluster")
            return self.state

        if rolling_error_rate >= self.thresholds.max_error_rate:
            self.state = KillSwitchState(active=True, reason="error_rate_breach")
            return self.state

        if drawdown_pct >= self.thresholds.max_drawdown_pct:
            self.state = KillSwitchState(active=True, reason="drawdown_breach")
            return self.state

        self.state = KillSwitchState(active=False, reason="")
        return self.state
