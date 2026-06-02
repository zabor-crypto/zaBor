#!/usr/bin/env python3
"""File-based trading lock that external bots can read.

Lock format (JSON):
  {
    "locked": true,
    "scope": "bitget_futures",
    "stage": 1,
    "reason": "killswitch_stage_1_close_top_risk_contributors",
    "created_ts": 1710000000,
    "expires_ts": 1710003600,
    "details": {...}
  }

Rules:
  - Stage 1 sets a temporary lock (expires after cooldown).
  - Stage 2 sets a longer lock.
  - Stage 3 sets a long lock that requires manual removal.
  - Stage 3+ locks are NEVER silently deleted by clear().
"""

import json
import os
import time
from typing import Optional


class TradingLock:
    def __init__(self, lock_file: str):
        self.lock_file = lock_file

    def set_lock(
        self,
        scope: str,
        stage: int,
        reason: str,
        expires_in_sec: int,
        details: Optional[dict] = None,
    ) -> dict:
        now = int(time.time())
        data = {
            "locked": True,
            "scope": scope,
            "stage": stage,
            "reason": reason,
            "created_ts": now,
            "expires_ts": now + expires_in_sec,
            "details": details or {},
        }
        with open(self.lock_file, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def is_locked(self) -> bool:
        if not os.path.exists(self.lock_file):
            return False
        try:
            with open(self.lock_file) as f:
                data = json.load(f)
            if not data.get("locked", False):
                return False
            expires = data.get("expires_ts", 0)
            if expires and int(time.time()) > expires:
                return False
            return True
        except Exception:
            return False

    def read(self) -> Optional[dict]:
        if not os.path.exists(self.lock_file):
            return None
        try:
            with open(self.lock_file) as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self, stage: int) -> None:
        """Remove lock file. Stage 3+ refuses — requires manual removal."""
        if stage >= 3:
            print(
                f"[TRADING LOCK] Stage {stage} lock requires manual removal: "
                f"{self.lock_file}"
            )
            return
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
