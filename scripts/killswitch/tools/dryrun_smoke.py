#!/usr/bin/env python3
"""
Dry-Run Smoke Test for Killswitch
Tests live API connections without executing orders.
Validates: connectivity, equity reading, snapshot storage, no false triggers.
"""

import sys
import os
import json
import time
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from killswitch import ConfigLoader, SqliteStore, DrawdownCalculator, RealCCXTAdapter, Scope, ExchangeId, AccountType


class SmokeTestRunner:
    """Runs smoke test on live APIs in dry-run mode"""
    
    def __init__(self, config_path: str, db_path: str):
        self.config_path = config_path
        self.db_path = db_path
        
        # Load config (force dry-run)
        loader = ConfigLoader()
        self.config = loader.load(config_path)
        self.config.dry_run = True  # FORCE DRY-RUN
        
        # Initialize components
        self.store = SqliteStore(db_path)
        self.dd_calc = DrawdownCalculator(self.store)
        
        # Metrics
        self.metrics = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "errors": defaultdict(int),
            "equity_readings": [],
            "max_dd": {}
        })
    
    def run(self, duration_minutes: int, exchanges: List[str] = None, scopes: List[str] = None) -> Dict:
        """
        Run smoke test for specified duration.
        
        Args:
            duration_minutes: How long to run
            exchanges: List of exchanges to test (None = all enabled)
            scopes: List of scopes to test (None = all enabled)
        
        Returns:
            Metrics dict
        """
        print(f"🔥 Killswitch Smoke Test")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Config: {self.config_path}")
        print(f"Database: {self.db_path}")
        print(f"DRY-RUN: {self.config.dry_run} (FORCED)")
        print(f"{'='*60}\n")
        
        # Build adapters
        adapters = {}
        for name, ex_config in self.config.exchanges.items():
            if exchanges and name not in exchanges:
                continue
            if ex_config.enabled:
                try:
                    adapters[name] = RealCCXTAdapter(ex_config, ExchangeId(name))
                    print(f"✓ Connected to {name}")
                except Exception as e:
                    print(f"✗ Failed to connect to {name}: {e}")
        
        if not adapters:
            print("ERROR: No exchanges available")
            return {"error": "no_exchanges"}
        
        print()
        
        # Run monitoring loop
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle_count = 0
        
        while time.time() < end_time:
            cycle_start = time.time()
            cycle_count += 1
            
            print(f"[Cycle {cycle_count}] {datetime.now().strftime('%H:%M:%S')}")
            
            for ex_name, adapter in adapters.items():
                ex_config = self.config.exchanges[ex_name]
                
                for acc_name, acc_config in ex_config.accounts.items():
                    if scopes and acc_name not in scopes:
                        continue
                    if not acc_config.enabled:
                        continue
                    
                    scope = Scope(ExchangeId(ex_name), AccountType(acc_name))
                    scope_key = str(scope)
                    
                    self.metrics[scope_key]["attempts"] += 1
                    
                    try:
                        # Fetch equity
                        snap = adapter.fetch_equity(scope)
                        
                        if snap.quality_ok and snap.equity_usdt > 0:
                            # Success
                            self.metrics[scope_key]["successes"] += 1
                            self.metrics[scope_key]["equity_readings"].append(snap.equity_usdt)
                            
                            # Store snapshot
                            self.store.append_snapshot(snap)
                            
                            # Calculate DD
                            if acc_config.windows:
                                dds = self.dd_calc.compute(scope, snap.ts, acc_config.windows)
                                
                                # Track max DD per window
                                for window, dd in dds.items():
                                    current_max = self.metrics[scope_key]["max_dd"].get(window, 0)
                                    self.metrics[scope_key]["max_dd"][window] = max(current_max, dd)
                                
                                dd_str = ", ".join([f"{w}:{v:.2%}" for w, v in dds.items()])
                                print(f"  {scope_key}: Eq=${snap.equity_usdt:.0f} | DD: {dd_str}")
                            else:
                                print(f"  {scope_key}: Eq=${snap.equity_usdt:.0f}")
                        else:
                            # Bad snapshot
                            self.metrics[scope_key]["failures"] += 1
                            self.metrics[scope_key]["errors"]["bad_snapshot"] += 1
                            print(f"  {scope_key}: BAD SNAPSHOT - {snap.raw}")
                    
                    except Exception as e:
                        # Error
                        self.metrics[scope_key]["failures"] += 1
                        error_type = type(e).__name__
                        self.metrics[scope_key]["errors"][error_type] += 1
                        print(f"  {scope_key}: ERROR - {error_type}: {e}")
            
            print()
            
            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(1, self.config.poll_seconds - elapsed)
            time.sleep(sleep_time)
        
        print(f"\n{'='*60}")
        print(f"Smoke Test Complete")
        print(f"Cycles: {cycle_count}")
        print(f"{'='*60}\n")
        
        return self._build_report()
    
    def _build_report(self) -> Dict:
        """Build final report"""
        report = {
            "timestamp": int(time.time()),
            "config": self.config_path,
            "scopes": {}
        }
        
        for scope_key, data in self.metrics.items():
            attempts = data["attempts"]
            successes = data["successes"]
            failures = data["failures"]
            
            success_rate = successes / attempts if attempts > 0 else 0
            
            equity_readings = data["equity_readings"]
            
            scope_report = {
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "success_rate": success_rate,
                "errors": dict(data["errors"]),
                "max_dd": data["max_dd"]
            }
            
            if equity_readings:
                scope_report["equity"] = {
                    "min": min(equity_readings),
                    "max": max(equity_readings),
                    "median": sorted(equity_readings)[len(equity_readings)//2],
                    "readings": len(equity_readings)
                }
            
            report["scopes"][scope_key] = scope_report
        
        return report
    
    def check_database(self) -> Dict:
        """Validate database integrity"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Count snapshots
        cur.execute("SELECT COUNT(*) FROM snapshots")
        snapshot_count = cur.fetchone()[0]
        
        # Count actions
        cur.execute("SELECT COUNT(*) FROM actions")
        action_count = cur.fetchone()[0]
        
        # Count cooldowns
        cur.execute("SELECT COUNT(*) FROM cooldowns")
        cooldown_count = cur.fetchone()[0]
        
        conn.close()
        
        return {
            "snapshots": snapshot_count,
            "actions": action_count,
            "cooldowns": cooldown_count
        }


def main():
    parser = argparse.ArgumentParser(description="Killswitch Smoke Test (Dry-Run)")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--minutes", type=int, default=15, help="Duration in minutes")
    parser.add_argument("--exchanges", help="Comma-separated exchanges (e.g., binance,bybit)")
    parser.add_argument("--scopes", help="Comma-separated scopes (e.g., futures,spot)")
    parser.add_argument("--db", default="./reports/smoke.sqlite", help="SQLite database path")
    parser.add_argument("--output", default="./reports/dryrun.json", help="Output JSON report")
    
    args = parser.parse_args()
    
    # Parse filters
    exchanges = args.exchanges.split(",") if args.exchanges else None
    scopes = args.scopes.split(",") if args.scopes else None
    
    # Ensure reports directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run smoke test
    runner = SmokeTestRunner(args.config, args.db)
    report = runner.run(args.minutes, exchanges, scopes)
    
    # Check database
    db_stats = runner.check_database()
    report["database"] = db_stats
    
    # Print summary
    print("\n📊 SUMMARY")
    print("="*60)
    for scope, data in report["scopes"].items():
        print(f"\n{scope}:")
        print(f"  Success Rate: {data['success_rate']:.1%}")
        if "equity" in data:
            print(f"  Equity Range: ${data['equity']['min']:.0f} - ${data['equity']['max']:.0f}")
        if data["max_dd"]:
            print(f"  Max DD: {', '.join([f'{w}:{v:.2%}' for w,v in data['max_dd'].items()])}")
        if data["errors"]:
            print(f"  Errors: {data['errors']}")
    
    print(f"\nDatabase:")
    print(f"  Snapshots: {db_stats['snapshots']}")
    print(f"  Actions: {db_stats['actions']}")
    print(f"  Cooldowns: {db_stats['cooldowns']}")
    
    # Write report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: {args.output}")
    
    # Exit codes
    issues = []
    for scope, data in report["scopes"].items():
        if data["success_rate"] < 0.95:
            issues.append(f"{scope}: Low success rate ({data['success_rate']:.1%})")
        
        if "equity" in data and data["equity"]["min"] <= 0:
            issues.append(f"{scope}: Zero equity detected")
    
    if db_stats["actions"] > 0:
        issues.append(f"Found {db_stats['actions']} actions (should be 0 in dry-run)")
    
    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(3)  # Data anomalies
    
    print("\n✅ All checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
