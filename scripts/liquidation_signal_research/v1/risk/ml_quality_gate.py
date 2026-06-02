"""ML-driven trade quality gate — hot-loaded from ml_auto_calibrator output.

This module provides a pre-trade filter that uses a trained logistic regression
model to estimate P(gross_profit > 0) from entry features. Trades below the
learned threshold are rejected.

The model weights are loaded from a JSON file that ml_auto_calibrator.py
writes on a schedule. The engine checks for updates every N seconds and
hot-reloads without restart.

Usage in engine:
    gate = MLQualityGate.from_file(Path("analysis/ml_calibration/ml_calibration_latest.json"))
    reasons = gate.evaluate(decision=decision, features=snapshot.features)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = "ml_auto_calibrator_v1"

FEATURE_NAMES = [
    "score", "liq_notional_log", "flow_imbalance_abs",
    "flow_imbalance_3s_abs", "flow_acceleration",
    "liq_flow_aligned", "liq_momentum",
    "depth_collapse_ratio", "spread_bps",
]


def _float(v: object, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


@dataclass
class MLQualityGate:
    """Pre-trade quality gate using ML model weights."""

    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    threshold: float = 0.5
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    source_path: Optional[Path] = None
    loaded_at: float = 0.0
    n_train: int = 0
    accuracy: float = 0.0
    enabled: bool = False
    # Hot-reload: check file mtime every this many seconds
    _reload_interval_s: float = 300.0
    _last_reload_check: float = 0.0
    _file_mtime: float = 0.0

    @classmethod
    def from_file(cls, path: Path) -> "MLQualityGate":
        """Load model from ml_calibration_latest.json."""
        gate = cls()
        gate.source_path = path
        gate._try_load(path)
        return gate

    def _try_load(self, path: Path) -> bool:
        if not path.exists():
            logger.info("MLQualityGate: file not found: %s", path)
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("MLQualityGate: failed to read %s: %s", path, exc)
            return False

        schema = data.get("schema_version", "")
        if schema != EXPECTED_SCHEMA:
            logger.warning("MLQualityGate: unexpected schema %s", schema)
            return False

        model = data.get("trade_quality_model", {})
        if not model.get("weights"):
            logger.info("MLQualityGate: no weights in model — disabled")
            return False

        self.weights = {k: float(v) for k, v in model["weights"].items()}
        self.bias = float(model.get("bias", 0.0))
        self.threshold = float(model.get("threshold", 0.5))
        self.feature_means = {k: float(v) for k, v in model.get("feature_means", {}).items()}
        self.feature_stds = {k: float(v) for k, v in model.get("feature_stds", {}).items()}
        self.n_train = int(model.get("n_train", 0))
        self.accuracy = float(model.get("accuracy", 0.0))
        self.loaded_at = time.time()
        self._file_mtime = path.stat().st_mtime
        self.enabled = True

        logger.info(
            "MLQualityGate: loaded model (n=%d, acc=%.3f, threshold=%.2f) from %s",
            self.n_train, self.accuracy, self.threshold, path,
        )
        return True

    def maybe_reload(self) -> None:
        """Check if the source file has been updated and reload if so."""
        if self.source_path is None:
            return
        now = time.time()
        if (now - self._last_reload_check) < self._reload_interval_s:
            return
        self._last_reload_check = now

        try:
            current_mtime = self.source_path.stat().st_mtime
        except OSError:
            return

        if current_mtime > self._file_mtime:
            logger.info("MLQualityGate: detected updated calibration file, reloading")
            self._try_load(self.source_path)

    def _extract_features(
        self,
        *,
        score: float,
        features: Mapping[str, object],
    ) -> Dict[str, float]:
        return {
            "score": score,
            "liq_notional_log": math.log1p(
                _float(features.get("liq_total_notional_1s")) / 20_000.0
            ),
            "flow_imbalance_abs": abs(_float(features.get("flow_imbalance_1s"))),
            "flow_imbalance_3s_abs": abs(_float(features.get("flow_imbalance_3s"))),
            "flow_acceleration": _float(features.get("flow_acceleration")),
            "liq_flow_aligned": _float(features.get("liq_flow_aligned")),
            "liq_momentum": _float(features.get("liq_momentum")),
            "depth_collapse_ratio": _float(features.get("depth_collapse_ratio")),
            "spread_bps": _float(features.get("spread_bps")),
        }

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for name in FEATURE_NAMES:
            v = raw.get(name, 0.0)
            m = self.feature_means.get(name, 0.0)
            s = self.feature_stds.get(name, 1.0)
            out[name] = (v - m) / max(s, 1e-8)
        return out

    def predict_proba(
        self,
        *,
        score: float,
        features: Mapping[str, object],
    ) -> float:
        """Return P(gross_profit > 0)."""
        if not self.enabled:
            return 1.0  # disabled → pass everything

        raw = self._extract_features(score=score, features=features)
        norm = self._normalize(raw)
        z = self.bias
        for name in FEATURE_NAMES:
            z += self.weights.get(name, 0.0) * norm.get(name, 0.0)
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))

    def evaluate(
        self,
        *,
        score: float,
        features: Mapping[str, object],
        action: str,
    ) -> List[str]:
        """Return list of rejection reasons (empty = pass).

        Called from the engine decision loop, same pattern as PerAssetCalibrationGate.
        """
        if action not in {"ENTER_LONG", "ENTER_SHORT"}:
            return []

        if not self.enabled:
            return []

        # Hot-reload check
        self.maybe_reload()

        proba = self.predict_proba(score=score, features=features)
        if proba < self.threshold:
            return [f"ml_quality_gate_below_threshold:{proba:.3f}<{self.threshold:.2f}"]

        return []
