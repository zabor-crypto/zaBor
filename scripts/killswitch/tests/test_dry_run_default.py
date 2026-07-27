#!/usr/bin/env python3
"""Regression tests for the fail-closed dry-run default.

The kill-switch must never place an order unless live mode was requested
explicitly. These tests pin that contract:

 - a config that omits `dry_run` loads as dry_run=True
 - `dry_run: false` is the only way to enable live execution
 - enabling live mode prints a prominent warning
 - there is no CLI flag that can turn on live execution
 - in the default mode no order endpoint is reached

Background: `_parse` previously used `data.get('dry_run', False)`, so a config
that simply did not mention the key executed real orders while the README
documented the opposite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from killswitch import ConfigLoader, build_arg_parser  # noqa: E402

from tests.fakes import create_futures_scenario  # noqa: E402


def _minimal_config(**overrides) -> dict:
    """Smallest config that parses; deliberately omits `dry_run`."""
    cfg = {
        "poll_seconds": 60,
        "exchanges": {},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Default is dry-run
# ---------------------------------------------------------------------------

def test_absent_dry_run_key_defaults_to_true():
    """A config with no `dry_run` key must load as dry-run, not live."""
    cfg = ConfigLoader()._parse(_minimal_config())
    assert cfg.dry_run is True


def test_absent_dry_run_key_announces_the_default(capsys):
    """The operator must be told the default was applied."""
    ConfigLoader()._parse(_minimal_config())
    out = capsys.readouterr().out
    assert "dry_run" in out
    assert "defaulting to dry_run=true" in out


def test_explicit_true_stays_dry_run():
    cfg = ConfigLoader()._parse(_minimal_config(dry_run=True))
    assert cfg.dry_run is True


def test_empty_config_defaults_to_dry_run():
    """Even a fully empty mapping must not produce live mode."""
    cfg = ConfigLoader()._parse({})
    assert cfg.dry_run is True


@pytest.mark.parametrize("falsy", [None, "", 0])
def test_null_or_blank_dry_run_never_silently_enables_live(falsy):
    """A null/blank value is operator ambiguity, not consent to trade live.

    `dry_run:` with no value parses to None in YAML. That must not be read as
    "false"; the only path to live execution is an explicit boolean false.
    """
    cfg = ConfigLoader()._parse(_minimal_config(dry_run=falsy))
    assert cfg.dry_run is True, (
        f"dry_run={falsy!r} must not enable live execution"
    )


# ---------------------------------------------------------------------------
# Live mode requires explicit opt-in, and is loud about it
# ---------------------------------------------------------------------------

def test_live_mode_requires_explicit_false():
    cfg = ConfigLoader()._parse(_minimal_config(dry_run=False))
    assert cfg.dry_run is False


def test_live_mode_prints_prominent_warning(capsys):
    ConfigLoader()._parse(_minimal_config(dry_run=False))
    out = capsys.readouterr().out
    assert "LIVE MODE" in out
    assert "Real orders WILL be placed" in out


def test_dry_run_default_prints_no_live_warning(capsys):
    """The warning must not fire on the safe path — it would train operators
    to ignore it."""
    ConfigLoader()._parse(_minimal_config())
    assert "LIVE MODE" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# No CLI route into live mode
# ---------------------------------------------------------------------------

def test_no_cli_flag_can_enable_live_execution():
    """Live execution is config-only. No CLI flag may bypass that."""
    parser = build_arg_parser()
    flags = {
        opt
        for action in parser._actions
        for opt in action.option_strings
    }
    forbidden = {"--live", "--no-dry-run", "--real", "--execute", "--force-live"}
    assert not (flags & forbidden), (
        f"CLI exposes a live-enabling flag: {flags & forbidden}"
    )


def test_default_cli_invocation_does_not_request_live():
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    assert not getattr(args, "live", False)
    assert not getattr(args, "no_dry_run", False)


# ---------------------------------------------------------------------------
# No order endpoint is reached in the default mode
# ---------------------------------------------------------------------------

def test_no_order_placed_when_dry_run_default():
    """End-to-end intent: with the default config, the exchange sees no order.

    Uses the fake CCXT client, which records every `create_order` call.
    """
    cfg = ConfigLoader()._parse(_minimal_config())
    assert cfg.dry_run is True

    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1},
        {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.5},
    ])

    # Simulate the close path exactly as ActionEngine gates it.
    for pos in client.fetch_positions():
        if cfg.dry_run:
            continue
        client.create_order(
            pos["symbol"], "market",
            "sell" if pos["side"] == "long" else "buy",
            pos["contracts"], params={"reduceOnly": True},
        )

    assert client.get_orders_log() == [], (
        "dry-run mode must not reach the order endpoint"
    )


def test_orders_are_placed_only_once_live_is_opted_into():
    """Counter-test: the guard above passes for the right reason."""
    cfg = ConfigLoader()._parse(_minimal_config(dry_run=False))
    assert cfg.dry_run is False

    client = create_futures_scenario([
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1},
    ])
    for pos in client.fetch_positions():
        if cfg.dry_run:
            continue
        client.create_order(
            pos["symbol"], "market", "sell", pos["contracts"],
            params={"reduceOnly": True},
        )

    assert len(client.get_orders_log()) == 1
