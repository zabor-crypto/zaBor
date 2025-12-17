from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bitget_order_plan_runs_in_dry_run(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
app:
  dry_run: true
  log_level: "INFO"
exchange:
  name: "bitget"
  product_type: "USDT-FUTURES"
  symbol: "ETHUSDT"
risk:
  max_notional_usdt: 50
  leverage: 2
plan:
  side: "buy"
  entry_type: "limit"
  entry_price: 0.0
  size_usdt: 25
""".strip(),
        encoding="utf-8",
    )

    script = REPO_ROOT / "scripts" / "bitget_order_plan.py"
    assert script.exists(), f"Missing script: {script}"

    r = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
