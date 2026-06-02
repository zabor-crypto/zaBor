"""Offline-only parameter governance and promotion hooks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ParameterBundle:
    bundle_id: str
    config_version: str
    created_at: str
    source_run_id: str
    parameters: Dict[str, float]
    notes: str = ""


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: List[str]


class ParameterGovernance:
    """Persists promoted parameter bundles and blocks unreviewed live mutations."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "parameter_registry.jsonl"

    def evaluate_promotion(
        self,
        *,
        replay_parity_ratio: float,
        min_replay_parity_ratio: float,
        oos_sharpe: float,
        min_oos_sharpe: float,
        max_drawdown: float,
        max_allowed_drawdown: float,
    ) -> PromotionDecision:
        reasons: List[str] = []
        if replay_parity_ratio < min_replay_parity_ratio:
            reasons.append("replay_parity_below_threshold")
        if oos_sharpe < min_oos_sharpe:
            reasons.append("oos_sharpe_below_threshold")
        if max_drawdown > max_allowed_drawdown:
            reasons.append("oos_drawdown_above_threshold")
        return PromotionDecision(approved=len(reasons) == 0, reasons=reasons)

    def persist_bundle(self, bundle_id: str, config_version: str, source_run_id: str, parameters: Dict[str, float], notes: str = "") -> ParameterBundle:
        bundle = ParameterBundle(
            bundle_id=bundle_id,
            config_version=config_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_run_id=source_run_id,
            parameters=parameters,
            notes=notes,
        )
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(bundle), separators=(",", ":"), ensure_ascii=True) + "\n")
        return bundle
