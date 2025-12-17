from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (YAML dict).")
    return data


def env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


@dataclass(frozen=True)
class TradePlan:
    side: str
    entry_type: str
    entry_price: float
    size_usdt: float


def validate_plan(plan: TradePlan, max_notional_usdt: float) -> None:
    if plan.side not in {"buy", "sell"}:
        raise ValueError("plan.side must be 'buy' or 'sell'")
    if plan.entry_type not in {"limit", "market"}:
        raise ValueError("plan.entry_type must be 'limit' or 'market'")
    if plan.size_usdt <= 0:
        raise ValueError("plan.size_usdt must be > 0")
    if plan.size_usdt > max_notional_usdt:
        raise ValueError(f"plan.size_usdt exceeds risk.max_notional_usdt ({max_notional_usdt})")
    if plan.entry_type == "limit" and plan.entry_price < 0:
        raise ValueError("plan.entry_price must be >= 0")


def main() -> int:
    parser = argparse.ArgumentParser(description="zaBor example: validate a trade plan (safe dry-run).")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    load_dotenv()

    cfg = load_yaml(Path(args.config))

    log_level = str(cfg.get("app", {}).get("log_level", "INFO"))
    setup_logging(log_level)
    log = logging.getLogger("zaBor")

    cfg_dry_run = bool(cfg.get("app", {}).get("dry_run", True))
    env_dry_run = env_flag("DRY_RUN", "1")
    dry_run = cfg_dry_run and env_dry_run  # both must allow live trading

    exchange = cfg.get("exchange", {})
    symbol = str(exchange.get("symbol", "UNKNOWN"))

    risk = cfg.get("risk", {})
    max_notional = float(risk.get("max_notional_usdt", 0.0))
    if max_notional <= 0:
        raise ValueError("risk.max_notional_usdt must be > 0")

    plan_dict = cfg.get("plan", {})
    plan = TradePlan(
        side=str(plan_dict.get("side", "")),
        entry_type=str(plan_dict.get("entry_type", "")),
        entry_price=float(plan_dict.get("entry_price", 0.0)),
        size_usdt=float(plan_dict.get("size_usdt", 0.0)),
    )
    validate_plan(plan, max_notional)

    log.info("MODE=%s", "DRY_RUN" if dry_run else "LIVE_REQUESTED")
    log.info("SYMBOL=%s", symbol)
    log.info("MAX_NOTIONAL_USDT=%.2f", max_notional)
    log.info("PLAN=%s", plan)

    if dry_run:
        log.warning("Dry-run enabled. No orders will be sent.")
        return 0

    # Live execution path is intentionally blocked until a real adapter is implemented.
    # This prevents accidental trading from a demo script.
    require_env("BITGET_KEY")
    require_env("BITGET_SECRET")
    require_env("BITGET_PASSWORD")

    raise NotImplementedError(
        "Live trading is not implemented in this example script. "
        "Implement exchange logic under exchanges/bitget/ and create a separate script."
    )


if __name__ == "__main__":
    raise SystemExit(main())
