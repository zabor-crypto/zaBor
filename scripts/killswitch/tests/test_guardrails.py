#!/usr/bin/env python3
"""Integration tests for the deploy guardrails: log-only, re-entry cooldown, daily-cap.
Drives _process_regime_guard through a MockAdapter + real SqliteStore."""
import sys, time, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from killswitch import (SqliteStore, PositionStore, TradingLock, MockAdapter, _process_regime_guard,
                        Scope, ExchangeId, AccountType, EquitySnapshot, ExchangeConfig)
from risk_attribution import PositionSnapshot
from regime_guard import RegimeGuardConfig


class _Log:
    def info(self, *a): pass
    def warning(self, *a): pass
    def critical(self, *a): pass
    def error(self, *a): pass


def _setup():
    db = tempfile.mktemp(suffix=".sqlite")
    store = SqliteStore(db); ps = PositionStore(store.conn)
    lock = TradingLock(tempfile.mktemp(suffix=".json"))
    ad = MockAdapter(ExchangeConfig(True, "", "", None, {}))
    ad._mock_macro = (-0.01, -0.01, -0.05)   # BTC down -> long flush allowed
    return db, store, ps, lock, ad


def _scope():
    return Scope(ExchangeId.MOCK, AccountType.FUTURES)


def _pos(sym, side, upnl, margin=24.0, now=None):
    return PositionSnapshot(now or int(time.time()), _scope(), sym, side, 100, 240, 1, 1, None,
                            margin, 10, upnl, upnl / margin, {})


def _eqsnap(now, eq):
    return EquitySnapshot(ts=now, scope=_scope(), equity_usdt=eq, wallet_usdt=eq, upnl_usdt=-40, quality_ok=True, raw={})


def _seed_equity(store, scope, now, peak=1090, lo=1000):
    # 48h of declining equity so dd from peak <= -5% (arms L2)
    for k in range(50):
        ts = now - (50 - k) * 3600
        store.append_snapshot(_eqsnap(ts, peak - (peak - lo) * (k / 49)))
    store.append_snapshot(_eqsnap(now, lo))


def test_log_only_does_not_execute_but_records():
    db, store, ps, lock, ad = _setup(); sc = _scope(); now = int(time.time())
    _seed_equity(store, sc, now)
    cfg = RegimeGuardConfig(enabled=True, log_only=True, persist_cycles=1)
    pos = [_pos(s, "long", -20, now=now) for s in ["A", "B", "C", "D"]]  # cluster of 4
    gstate = {}
    fired = _process_regime_guard(sc, cfg, _eqsnap(now, 1000), pos, ad, store, ps, lock,
                                  dry_run=False, logger=_Log(), notifier=None, guard_state=gstate)
    assert fired
    acts = [r[0] for r in store.conn.execute("SELECT key FROM actions")]
    assert any("logonly_regime" in a for a in acts)   # recorded with log-only marker
    os.remove(db)


def test_daily_cap_suppresses_after_limit():
    db, store, ps, lock, ad = _setup(); sc = _scope(); now = int(time.time())
    _seed_equity(store, sc, now)
    cfg = RegimeGuardConfig(enabled=True, log_only=True, persist_cycles=1, max_closes_per_day=3,
                            l3_cooldown_min=0, l2_cooldown_min=0)
    gstate = {}
    # pre-load closes_today at the cap
    gstate[str(sc)] = {"dd_count": 5, "closes_today": 3, "day": now // 86400, "reentry": {}}
    pos = [_pos(s, "long", -20, now=now) for s in ["A", "B", "C", "D"]]
    fired = _process_regime_guard(sc, cfg, _eqsnap(now, 1000), pos, ad, store, ps, lock,
                                  dry_run=False, logger=_Log(), notifier=None, guard_state=gstate)
    assert not fired   # cap reached -> suppressed
    os.remove(db)


def test_reentry_cooldown_recloses_reappeared_symbol():
    # Re-entry is LIVE-only (log_only=False). It re-closes a symbol that was flushed and re-opened.
    db, store, ps, lock, ad = _setup(); sc = _scope(); now = int(time.time())
    _seed_equity(store, sc, now)
    cfg = RegimeGuardConfig(enabled=True, log_only=False, persist_cycles=1, reentry_cooldown_min=360)
    gstate = {str(sc): {"dd_count": 0, "closes_today": 0, "day": now // 86400,
                        "reentry": {"A|long": now + 3600}}}
    pos = [_pos("A", "long", -2, now=now)]   # A re-opened, only mildly down (no trigger would fire)
    fired = _process_regime_guard(sc, cfg, _eqsnap(now, 1000), pos, ad, store, ps, lock,
                                  dry_run=False, logger=_Log(), notifier=None, guard_state=gstate)
    assert fired
    acts = [r[0] for r in store.conn.execute("SELECT key FROM actions")]
    assert any("reentry" in a for a in acts)
    os.remove(db)


def test_reentry_NOT_enforced_in_log_only():
    # REGRESSION: in log-only nothing is actually flushed, so the re-entry guard must NOT fire (else it
    # mistakes an always-open position for a re-opened one and spams every cycle — the GWEI bug).
    db, store, ps, lock, ad = _setup(); sc = _scope(); now = int(time.time())
    for k in range(50):                       # flat equity -> no L2/L4 trigger
        store.append_snapshot(_eqsnap(now - (50 - k) * 3600, 1000))
    store.append_snapshot(_eqsnap(now, 1000))
    cfg = RegimeGuardConfig(enabled=True, log_only=True, persist_cycles=1, reentry_cooldown_min=360)
    gstate = {str(sc): {"dd_count": 0, "closes_today": 0, "day": now // 86400,
                        "reentry": {"GWEI|short": now + 3600}}}   # stale from earlier
    pos = [_pos("GWEI", "short", 0, now=now)]  # now ~flat
    fired = _process_regime_guard(sc, cfg, _eqsnap(now, 1000), pos, ad, store, ps, lock,
                                  dry_run=False, logger=_Log(), notifier=None, guard_state=gstate)
    assert not fired   # log-only -> re-entry not enforced -> no spam
    os.remove(db)


def test_reentry_window_not_extended_by_enforcement():
    # The re-entry window must anchor to the first flush and NOT renew each time it re-closes.
    db, store, ps, lock, ad = _setup(); sc = _scope(); now = int(time.time())
    _seed_equity(store, sc, now)
    cfg = RegimeGuardConfig(enabled=True, log_only=False, persist_cycles=1, reentry_cooldown_min=360)
    orig_exp = now + 100   # window expires soon
    gstate = {str(sc): {"dd_count": 0, "closes_today": 0, "day": now // 86400,
                        "reentry": {"A|long": orig_exp}}}
    pos = [_pos("A", "long", -2, now=now)]
    _process_regime_guard(sc, cfg, _eqsnap(now, 1000), pos, ad, store, ps, lock,
                          dry_run=False, logger=_Log(), notifier=None, guard_state=gstate)
    # enforcement closed A but must NOT have pushed the expiry forward
    assert gstate[str(sc)]["reentry"].get("A|long") == orig_exp
    os.remove(db)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
