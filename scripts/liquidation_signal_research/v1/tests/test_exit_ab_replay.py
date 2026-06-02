import json

from v1.analysis.exit_ab_replay import (
    StaticExitConfig,
    evaluate_entries,
    load_candidate_from_sweep,
    simulate_live_price_profile_exit,
    simulate_static_exit,
)


def test_simulate_static_exit_tp_hit() -> None:
    pnl_path = [(250, 1.0), (500, 6.0), (750, 3.0)]
    pnl, reason, elapsed = simulate_static_exit(
        pnl_path,
        tp_bps=5.0,
        sl_bps=10.0,
        time_limit_ms=60_000,
    )
    assert reason == "tp_hit"
    assert elapsed == 500
    assert pnl == 6.0


def test_simulate_static_exit_time_limit() -> None:
    pnl_path = [(250, 0.5), (61_000, 2.0)]
    pnl, reason, elapsed = simulate_static_exit(
        pnl_path,
        tp_bps=10.0,
        sl_bps=10.0,
        time_limit_ms=60_000,
    )
    assert reason == "time_limit"
    assert elapsed == 61_000
    assert pnl == 2.0


def test_simulate_live_price_profile_trailing_stop() -> None:
    # MFE=4.0 bps at tick 500 → floor = 4.0 - TRAIL_BPS(1.0) = 3.0.
    # Tick 750: pnl=2.0 < 3.0 → trailing_stop fires.
    pnl_path = [(250, 1.0), (500, 4.0), (750, 2.0), (1000, 1.5)]
    pnl, reason, elapsed = simulate_live_price_profile_exit(
        pnl_path,
        score=0.5,
        liq_notional_1s=0.0,
    )
    assert reason == "trailing_stop"
    assert elapsed == 750
    assert pnl == 2.0


def test_load_candidate_from_sweep(tmp_path) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "report": {
                    "grid_search": {
                        "best": {
                            "tp_bps": 18.0,
                            "sl_bps": 10.0,
                            "time_limit_ms": 90_000,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_candidate_from_sweep(path)
    assert cfg.tp_bps == 18.0
    assert cfg.sl_bps == 10.0
    assert cfg.time_limit_ms == 90_000


def test_evaluate_entries_delta_shape() -> None:
    entries = [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "score": 0.55,
            "liq_notional": 20_000.0,
            "pnl_path": [(250, 1.0), (500, 4.0), (750, 1.0), (1000, -2.0)],
        },
        {
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "score": 0.45,
            "liq_notional": 10_000.0,
            "pnl_path": [(250, -1.0), (500, -4.0), (750, -6.0)],
        },
    ]
    rows, summary = evaluate_entries(
        entries,
        candidate=StaticExitConfig(tp_bps=8.0, sl_bps=5.0, time_limit_ms=60_000, source="test"),
        friction_bps=8.0,
    )
    assert len(rows) == 2
    assert summary["baseline"]["n"] == 2
    assert summary["candidate"]["n"] == 2
    assert summary["delta"]["n"] == 2
    assert "BTCUSDT" in summary["per_symbol"]
