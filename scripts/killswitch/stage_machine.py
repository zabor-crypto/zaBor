#!/usr/bin/env python3
"""Persistent per-scope stage state machine for the kill-switch.

Stage semantics:
  0 = NORMAL              — no action taken
  1 = REDUCED_TOP_RISK    — closed top-N risk contributors (partial)
  2 = DOMINANT_CLOSED     — closed dominant loss direction
  3 = ALL_FUTURES_CLOSED  — closed all futures positions
  4 = LOCKDOWN            — global lockdown, manual intervention required

Rules:
  - Escalation (new_stage > current_stage) is always allowed.
  - Same stage re-executes only after lock_until expires.
  - De-escalation (new_stage < current_stage) is blocked.
  - Stage 3+ lock is noted as requiring manual reset.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple

STAGE_NORMAL = 0
STAGE_REDUCED_TOP_RISK = 1
STAGE_DOMINANT_CLOSED = 2
STAGE_ALL_FUTURES_CLOSED = 3
STAGE_LOCKDOWN = 4

STAGE_NAMES = {
    0: "NORMAL",
    1: "REDUCED_TOP_RISK",
    2: "DOMINANT_CLOSED",
    3: "ALL_FUTURES_CLOSED",
    4: "LOCKDOWN",
}

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS scope_stage_state (
    scope TEXT PRIMARY KEY,
    current_stage INTEGER NOT NULL DEFAULT 0,
    last_stage_ts INTEGER NOT NULL DEFAULT 0,
    last_source TEXT,
    lock_until INTEGER,
    updated_ts INTEGER NOT NULL,
    details TEXT
);
"""


@dataclass
class StageState:
    scope: str
    current_stage: int
    last_stage_ts: int
    last_source: Optional[str]
    lock_until: Optional[int]
    updated_ts: int
    details: Optional[dict]


class StageMachine:
    """Persistent stage state machine backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        with self.conn:
            self.conn.execute(_INIT_SQL)

    def get_state(self, scope: str) -> StageState:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT current_stage, last_stage_ts, last_source, lock_until, "
            "updated_ts, details FROM scope_stage_state WHERE scope=?",
            (scope,),
        )
        row = cur.fetchone()
        if not row:
            return StageState(scope, 0, 0, None, None, 0, None)
        details = None
        if row[5]:
            try:
                details = json.loads(row[5])
            except Exception:
                pass
        return StageState(scope, row[0], row[1], row[2], row[3], row[4], details)

    def can_execute(self, scope: str, new_stage: int, now_ts: int) -> Tuple[bool, str]:
        """Return (allowed, reason_if_blocked).

        Higher stage always overrides lower-stage cooldown — escalation is
        never suppressed by a previous stage's lock_until.
        """
        state = self.get_state(scope)

        if new_stage > state.current_stage:
            return True, ""

        if new_stage == state.current_stage:
            if state.lock_until is None or now_ts >= state.lock_until:
                return True, ""
            remaining = state.lock_until - now_ts
            m, s = divmod(remaining, 60)
            return False, (
                f"stage {new_stage} same-stage cooldown active, "
                f"{m}m {s}s remaining"
            )

        # new_stage < current_stage — de-escalation blocked
        return False, (
            f"stage {new_stage} blocked: current stage is "
            f"{state.current_stage} ({STAGE_NAMES.get(state.current_stage, '?')})"
        )

    def record_execution(
        self,
        scope: str,
        stage: int,
        lock_until: int,
        source: str,
        details: dict,
        now_ts: int,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO scope_stage_state
                (scope, current_stage, last_stage_ts, last_source,
                 lock_until, updated_ts, details)
                VALUES (?,?,?,?,?,?,?)""",
                (scope, stage, now_ts, source, lock_until,
                 now_ts, json.dumps(details)),
            )

    def reset(self, scope: str, now_ts: int) -> None:
        """Reset scope to NORMAL (stage 0). Stage 3+ logs a warning."""
        state = self.get_state(scope)
        if state.current_stage >= STAGE_ALL_FUTURES_CLOSED:
            print(
                f"[STAGE MACHINE] Resetting stage {state.current_stage} for {scope} "
                f"— manual intervention confirmation."
            )
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO scope_stage_state
                (scope, current_stage, last_stage_ts, last_source,
                 lock_until, updated_ts, details)
                VALUES (?,0,?,null,null,?,null)""",
                (scope, now_ts, now_ts),
            )
