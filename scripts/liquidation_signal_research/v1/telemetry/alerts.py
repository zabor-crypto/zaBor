"""Alert hook abstraction for risk and runtime incidents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Alert:
    severity: str
    code: str
    message: str


class AlertDispatcher:
    def __init__(self, sink: Optional[Callable[[Alert], None]] = None) -> None:
        self.sink = sink

    def emit(self, severity: str, code: str, message: str) -> None:
        alert = Alert(severity=severity, code=code, message=message)
        if self.sink is not None:
            self.sink(alert)
