from v1.telemetry.shadow_monitors import ShadowMonitor


def test_shadow_monitor_aggregates_distributions() -> None:
    monitor = ShadowMonitor()

    monitor.observe_decision(
        {
            "action": "NO_TRADE",
            "score": 0.12,
            "no_trade_reason": "score_below_threshold;spread_too_wide",
            "risk_reasons": ["book_desynced_or_stale"],
        }
    )
    monitor.observe_decision(
        {
            "action": "ENTER_LONG",
            "score": 0.56,
            "no_trade_reason": None,
            "risk_reasons": [],
        }
    )

    monitor.observe_risk_event({"event": "book_desync"})
    monitor.observe_risk_event({"event": "book_resync_start"})
    monitor.observe_risk_event({"event": "book_resync_ok"})

    summary = monitor.summary()
    assert summary.total_decisions == 2
    assert summary.action_counts["NO_TRADE"] == 1
    assert summary.action_counts["ENTER_LONG"] == 1
    assert summary.no_trade_reason_counts["score_below_threshold"] == 1
    assert summary.no_trade_reason_counts["spread_too_wide"] == 1
    assert summary.risk_reason_counts["book_desynced_or_stale"] == 1
    assert summary.desync_events == 1
    assert summary.resync_attempts == 1
    assert summary.resync_success == 1
    assert summary.resync_failures == 0
