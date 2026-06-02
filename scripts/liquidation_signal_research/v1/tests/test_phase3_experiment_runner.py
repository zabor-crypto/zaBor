from v1.analysis.phase3_experiment_runner import (
    CandidateParams,
    SelectionRules,
    TradeRecord,
    compute_trade_metrics,
    evaluate_asset_candidates,
    trade_passes_candidate,
)


def _trade(
    *,
    ts: int,
    gross: float,
    score: float,
    symbol: str = "BTCUSDT",
) -> TradeRecord:
    return TradeRecord(
        exchange_ts_ms=ts,
        decision_id=f"d{ts}",
        symbol=symbol,
        direction="LONG",
        exit_reason="trailing_stop" if gross > 0 else "hard_stop",
        gross_pnl_bps=gross,
        mfe_bps=max(0.0, gross + 2.0),
        mae_bps=min(0.0, gross - 1.0),
        hold_ms=5000,
        decision_joined=True,
        feature_joined=True,
        score=score,
        expected_edge_bps=16.0,
        liq_total_notional_1s=90_000.0,
        depth_collapse_ratio=0.12,
        spread_bps=3.0,
        flow_imbalance_1s=0.15,
        ret_500_bps=1.2,
    )


def test_compute_trade_metrics_drawdown_and_net() -> None:
    rows = [
        _trade(ts=1, gross=4.0, score=0.45),
        _trade(ts=2, gross=-3.0, score=0.45),
        _trade(ts=3, gross=5.0, score=0.45),
    ]
    out = compute_trade_metrics(rows, friction_bps=1.0)
    assert out["n"] == 3
    assert out["net_mean_bps"] == 1.0
    assert out["net_sum_bps"] == 3.0
    assert out["max_drawdown_bps"] == 4.0


def test_trade_passes_candidate_checks_thresholds() -> None:
    row = _trade(ts=1, gross=3.0, score=0.44)
    ok = CandidateParams(
        min_score=0.40,
        min_expected_edge_bps=12.0,
        min_liq_notional_1s=50_000.0,
        min_depth_collapse_ratio=0.08,
        min_abs_flow_imbalance_1s=0.10,
        max_spread_bps=4.0,
    )
    strict = CandidateParams(
        min_score=0.45,
        min_expected_edge_bps=12.0,
        min_liq_notional_1s=50_000.0,
        min_depth_collapse_ratio=0.08,
        min_abs_flow_imbalance_1s=0.10,
        max_spread_bps=4.0,
    )
    assert trade_passes_candidate(row, ok) is True
    assert trade_passes_candidate(row, strict) is False


def test_evaluate_asset_candidates_promotes_better_filter() -> None:
    rows = [
        _trade(ts=1, gross=-6.0, score=0.36),
        _trade(ts=2, gross=-5.5, score=0.36),
        _trade(ts=3, gross=6.2, score=0.52),
        _trade(ts=4, gross=7.0, score=0.53),
        _trade(ts=5, gross=-4.3, score=0.37),
        _trade(ts=6, gross=8.0, score=0.54),
        _trade(ts=7, gross=-3.5, score=0.37),
        _trade(ts=8, gross=7.1, score=0.55),
        _trade(ts=9, gross=-5.0, score=0.36),
        _trade(ts=10, gross=9.2, score=0.56),
        _trade(ts=11, gross=-4.1, score=0.36),
        _trade(ts=12, gross=8.3, score=0.57),
    ]
    loose = CandidateParams(
        min_score=0.35,
        min_expected_edge_bps=11.0,
        min_liq_notional_1s=20_000.0,
        min_depth_collapse_ratio=0.05,
        min_abs_flow_imbalance_1s=0.08,
        max_spread_bps=6.0,
    )
    strict = CandidateParams(
        min_score=0.50,
        min_expected_edge_bps=11.0,
        min_liq_notional_1s=20_000.0,
        min_depth_collapse_ratio=0.05,
        min_abs_flow_imbalance_1s=0.08,
        max_spread_bps=6.0,
    )
    rules = SelectionRules(
        min_trade_retention_ratio=0.2,
        min_test_trades_abs=1,
        min_net_improvement_bps=0.1,
        min_candidate_test_net_mean_bps=-5.0,
        min_candidate_test_net_sum_bps=-100.0,
        max_drawdown_multiplier=1.5,
        max_drawdown_bps_when_baseline_zero=20.0,
        require_train_non_worse=True,
    )
    report = evaluate_asset_candidates(
        symbol="BTCUSDT",
        rows=rows,
        candidates=[loose, strict],
        friction_bps=1.0,
        train_ratio=0.7,
        min_rows_for_split=8,
        rules=rules,
        top_k=3,
    )
    assert report["selected"]["promoted"] is True
    assert report["selected"]["params"]["min_score"] == 0.5
