"""Simple in-memory metrics registry with snapshot export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricsRegistry:
    counters: Dict[str, float] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = float(value)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
        }
