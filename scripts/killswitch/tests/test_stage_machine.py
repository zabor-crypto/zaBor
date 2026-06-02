#!/usr/bin/env python3
"""Unit tests for the stage state machine.

Covers:
 - Stage 2 can fire after Stage 1 despite lower-stage cooldown
 - Stage 3 can fire after any previous stage
 - Stage 1 re-execution is blocked during same-stage cooldown
 - De-escalation is blocked
 - reset() restores NORMAL state
 - Window normalization deduplicates duplicate window specs
"""

import sys
import sqlite3
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage_machine import (
    StageMachine,
    StageState,
    STAGE_NORMAL,
    STAGE_REDUCED_TOP_RISK,
    STAGE_DOMINANT_CLOSED,
    STAGE_ALL_FUTURES_CLOSED,
    STAGE_LOCKDOWN,
    STAGE_NAMES,
)
from killswitch import DrawdownCalculator, SqliteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def machine(conn):
    return StageMachine(conn)


BASE_TS = int(time.time())
SCOPE = "binance_futures"


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_default_state_is_normal(self, machine):
        state = machine.get_state(SCOPE)
        assert state.current_stage == STAGE_NORMAL
        assert state.lock_until is None

    def test_can_execute_stage_1_from_normal(self, machine):
        can, reason = machine.can_execute(SCOPE, 1, BASE_TS)
        assert can is True
        assert reason == ""

    def test_can_execute_stage_3_from_normal(self, machine):
        can, _ = machine.can_execute(SCOPE, 3, BASE_TS)
        assert can is True


# ---------------------------------------------------------------------------
# Stage 1 executed — cooldown active
# ---------------------------------------------------------------------------

class TestStage1Cooldown:

    def _execute_stage1(self, machine):
        lock_until = BASE_TS + 1800  # 30 min
        machine.record_execution(SCOPE, 1, lock_until, "CLOSE_TOP_RISK_CONTRIBUTORS", {}, BASE_TS)

    def test_stage1_re_blocked_during_cooldown(self, machine):
        self._execute_stage1(machine)
        can, reason = machine.can_execute(SCOPE, 1, BASE_TS + 60)
        assert can is False
        assert "cooldown" in reason.lower()

    def test_stage2_allowed_after_stage1(self, machine):
        """Stage 2 must be allowed after Stage 1 even within Stage 1 cooldown."""
        self._execute_stage1(machine)
        can, reason = machine.can_execute(SCOPE, 2, BASE_TS + 60)
        assert can is True, f"Stage 2 should be allowed: {reason}"

    def test_stage3_allowed_after_stage1(self, machine):
        """Stage 3 must be allowed after Stage 1 even within Stage 1 cooldown."""
        self._execute_stage1(machine)
        can, reason = machine.can_execute(SCOPE, 3, BASE_TS + 60)
        assert can is True, f"Stage 3 should be allowed: {reason}"

    def test_stage1_allowed_after_cooldown_expires(self, machine):
        """Same stage can re-execute once cooldown has expired."""
        lock_until = BASE_TS + 100
        machine.record_execution(SCOPE, 1, lock_until, "mode", {}, BASE_TS)
        can, _ = machine.can_execute(SCOPE, 1, BASE_TS + 200)
        assert can is True


# ---------------------------------------------------------------------------
# Full escalation: 1 -> 2 -> 3
# ---------------------------------------------------------------------------

class TestEscalation:

    def test_full_escalation_chain(self, machine):
        """Stages 1 → 2 → 3 should all succeed."""
        ts = BASE_TS

        # Stage 1
        can, _ = machine.can_execute(SCOPE, 1, ts)
        assert can
        machine.record_execution(SCOPE, 1, ts + 3600, "mode1", {}, ts)

        # Stage 2 (immediately, while stage-1 cooldown active)
        ts += 30
        can, _ = machine.can_execute(SCOPE, 2, ts)
        assert can
        machine.record_execution(SCOPE, 2, ts + 7200, "mode2", {}, ts)

        # Stage 3 (immediately, while stage-2 cooldown active)
        ts += 30
        can, _ = machine.can_execute(SCOPE, 3, ts)
        assert can
        machine.record_execution(SCOPE, 3, ts + 21600, "mode3", {}, ts)

        state = machine.get_state(SCOPE)
        assert state.current_stage == 3

    def test_stage2_blocked_after_stage3(self, machine):
        """Cannot de-escalate from stage 3 to stage 2."""
        machine.record_execution(SCOPE, 3, BASE_TS + 21600, "mode3", {}, BASE_TS)
        can, reason = machine.can_execute(SCOPE, 2, BASE_TS + 100)
        assert can is False
        assert "stage 3" in reason.lower() or "current stage" in reason.lower()

    def test_stage1_blocked_after_stage3(self, machine):
        """Cannot de-escalate from stage 3 to stage 1."""
        machine.record_execution(SCOPE, 3, BASE_TS + 21600, "mode3", {}, BASE_TS)
        can, reason = machine.can_execute(SCOPE, 1, BASE_TS + 100)
        assert can is False

    def test_stage3_re_blocked_during_cooldown(self, machine):
        """Stage 3 itself is blocked for repeat during its cooldown."""
        lock_until = BASE_TS + 21600
        machine.record_execution(SCOPE, 3, lock_until, "mode3", {}, BASE_TS)
        can, reason = machine.can_execute(SCOPE, 3, BASE_TS + 100)
        assert can is False
        assert "cooldown" in reason.lower()

    def test_stage3_allowed_after_cooldown(self, machine):
        """Stage 3 can fire again once its cooldown expires."""
        lock_until = BASE_TS + 100
        machine.record_execution(SCOPE, 3, lock_until, "mode3", {}, BASE_TS)
        can, _ = machine.can_execute(SCOPE, 3, BASE_TS + 200)
        assert can is True


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:

    def test_state_stored_correctly(self, machine):
        lock_until = BASE_TS + 3600
        machine.record_execution(SCOPE, 2, lock_until, "test_source",
                                  {"mode": "test"}, BASE_TS)
        state = machine.get_state(SCOPE)
        assert state.current_stage == 2
        assert state.last_stage_ts == BASE_TS
        assert state.last_source == "test_source"
        assert state.lock_until == lock_until
        assert state.details == {"mode": "test"}

    def test_reset_clears_stage(self, machine):
        machine.record_execution(SCOPE, 2, BASE_TS + 3600, "mode", {}, BASE_TS)
        machine.reset(SCOPE, BASE_TS + 100)
        state = machine.get_state(SCOPE)
        assert state.current_stage == STAGE_NORMAL
        assert state.lock_until is None

    def test_multiple_scopes_independent(self, machine):
        """Different scopes have independent state."""
        scope_a = "binance_futures"
        scope_b = "bybit_futures"
        machine.record_execution(scope_a, 2, BASE_TS + 3600, "m", {}, BASE_TS)
        machine.record_execution(scope_b, 1, BASE_TS + 1800, "m", {}, BASE_TS)

        state_a = machine.get_state(scope_a)
        state_b = machine.get_state(scope_b)
        assert state_a.current_stage == 2
        assert state_b.current_stage == 1


# ---------------------------------------------------------------------------
# Window normalization (DrawdownCalculator)
# ---------------------------------------------------------------------------

class TestWindowNormalization:

    def test_parse_window_minutes_string(self):
        assert DrawdownCalculator.parse_window("15m") == 15
        assert DrawdownCalculator.parse_window("5m") == 5

    def test_parse_window_hours_string(self):
        assert DrawdownCalculator.parse_window("1h") == 60
        assert DrawdownCalculator.parse_window("4h") == 240

    def test_parse_window_days_string(self):
        assert DrawdownCalculator.parse_window("1d") == 1440

    def test_parse_window_integer(self):
        assert DrawdownCalculator.parse_window(15) == 15
        assert DrawdownCalculator.parse_window(60) == 60

    def test_parse_window_numeric_string(self):
        assert DrawdownCalculator.parse_window("60") == 60
        assert DrawdownCalculator.parse_window("15") == 15

    def test_deduplicate_windows(self):
        """[15, '15m', 60, '1h'] should deduplicate to 2 entries."""
        from killswitch import ConfigLoader
        result = ConfigLoader._deduplicate_windows([15, "15m", 60, "1h"])
        # Should have exactly 2 unique windows (15 min and 60 min)
        assert len(result) == 2

    def test_deduplicate_keeps_first_occurrence(self):
        """First occurrence is kept when deduplicating."""
        from killswitch import ConfigLoader
        result = ConfigLoader._deduplicate_windows(["15m", "15", 15])
        assert len(result) == 1
        assert result[0] == "15m"

    def test_canonical_windows_unchanged(self):
        """Already canonical windows should pass through unchanged."""
        from killswitch import ConfigLoader
        result = ConfigLoader._deduplicate_windows(["15m", "1h"])
        assert result == ["15m", "1h"]
