"""Attribution hooks from raw trigger to decision and order lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AttributionLink:
    episode_id: str
    decision_id: str
    order_intent_id: str
    raw_trigger_event_id: str


class AttributionTracker:
    def __init__(self) -> None:
        self.links: Dict[str, AttributionLink] = {}

    def upsert(self, link: AttributionLink) -> None:
        self.links[link.decision_id] = link

    def get(self, decision_id: str) -> AttributionLink | None:
        return self.links.get(decision_id)
