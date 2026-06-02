#!/usr/bin/env python3
"""Persistent position snapshot store for PnL attribution lookback."""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS position_snapshots (
        ts INTEGER,
        exchange TEXT,
        account TEXT,
        symbol TEXT,
        side TEXT,
        contracts REAL,
        notional_usdt REAL,
        entry_price REAL,
        mark_price REAL,
        liquidation_price REAL,
        margin_usdt REAL,
        leverage REAL,
        unrealized_pnl_usdt REAL,
        pnl_pct_on_margin REAL,
        raw_json TEXT,
        PRIMARY KEY (exchange, account, symbol, side, ts)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_position_snapshots_scope_ts
    ON position_snapshots(exchange, account, ts)
    """,
]


class PositionStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        with self.conn:
            for stmt in _SCHEMA:
                self.conn.execute(stmt)

    def save_many(self, snapshots: List[Any]) -> None:
        """Persist a list of PositionSnapshot objects."""
        with self.conn:
            for s in snapshots:
                self.conn.execute(
                    """INSERT OR IGNORE INTO position_snapshots
                    (ts, exchange, account, symbol, side, contracts, notional_usdt,
                     entry_price, mark_price, liquidation_price, margin_usdt,
                     leverage, unrealized_pnl_usdt, pnl_pct_on_margin, raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        s.ts,
                        s.scope.exchange.value,
                        s.scope.account.value,
                        s.symbol,
                        s.side,
                        s.contracts,
                        s.notional_usdt,
                        s.entry_price,
                        s.mark_price,
                        s.liquidation_price,
                        s.margin_usdt,
                        s.leverage,
                        s.unrealized_pnl_usdt,
                        s.pnl_pct_on_margin,
                        json.dumps(s.raw),
                    ),
                )

    def get_reference_snapshots(
        self, scope: Any, lookback_sec: int, now_ts: int
    ) -> Dict[Tuple[str, str], Any]:
        """Return earliest PositionSnapshot per (symbol, side) within lookback.

        Used as the reference baseline for PnL delta attribution.
        Falls back to the oldest available snapshot inside the window.
        """
        from risk_attribution import PositionSnapshot

        min_ts = now_ts - lookback_sec
        cur = self.conn.cursor()
        cur.execute(
            """SELECT ts, symbol, side, contracts, notional_usdt,
               entry_price, mark_price, liquidation_price, margin_usdt,
               leverage, unrealized_pnl_usdt, pnl_pct_on_margin, raw_json
               FROM position_snapshots
               WHERE exchange=? AND account=? AND ts >= ?
               ORDER BY ts ASC""",
            (scope.exchange.value, scope.account.value, min_ts),
        )
        rows = cur.fetchall()
        seen: Dict[Tuple[str, str], Any] = {}
        for row in rows:
            ts, symbol, side = row[0], row[1], row[2]
            key = (symbol, side)
            if key not in seen:
                raw: dict = {}
                if row[12]:
                    try:
                        raw = json.loads(row[12])
                    except Exception:
                        pass
                seen[key] = PositionSnapshot(
                    ts=ts,
                    scope=scope,
                    symbol=symbol,
                    side=side,
                    contracts=row[3] or 0.0,
                    notional_usdt=row[4] or 0.0,
                    entry_price=row[5],
                    mark_price=row[6],
                    liquidation_price=row[7],
                    margin_usdt=row[8],
                    leverage=row[9],
                    unrealized_pnl_usdt=row[10] or 0.0,
                    pnl_pct_on_margin=row[11],
                    raw=raw,
                )
        return seen
