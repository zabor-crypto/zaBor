#!/usr/bin/env python3
"""
Scenario Builder for Killswitch Testing
Generates synthetic equity/balance/position time series for comprehensive testing.
"""

import json
import argparse
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation"""
    name: str
    description: str
    start_equity: float
    minutes: int
    pattern: str
    drop_pct: float = 0.0
    noise_sigma_pct: float = 0.01
    missing_prob: float = 0.0
    bad_snapshot_prob: float = 0.0
    expected_outcome: Optional[Dict] = None


class ScenarioBuilder:
    """Build synthetic test scenarios"""
    
    def __init__(self, base_ts: int = 1706000000):
        self.base_ts = base_ts
    
    def generate(self, config: ScenarioConfig) -> Dict[str, Any]:
        """Generate a complete scenario"""
        timeline = [self.base_ts + (i * 60) for i in range(config.minutes)]
        
        # Generate equity series based on pattern
        if config.pattern == "flat":
            equity_series = self._flat(config)
        elif config.pattern == "step_down":
            equity_series = self._step_down(config)
        elif config.pattern == "gradual_down":
            equity_series = self._gradual_down(config)
        elif config.pattern == "spike_down_recover":
            equity_series = self._spike_down_recover(config)
        elif config.pattern == "noisy":
            equity_series = self._noisy(config)
        else:
            raise ValueError(f"Unknown pattern: {config.pattern}")
        
        # Apply noise if specified
        if config.noise_sigma_pct > 0:
            equity_series = self._add_noise(equity_series, config.noise_sigma_pct)
        
        # Apply missing data
        quality_series = [True] * len(equity_series)
        if config.missing_prob > 0 or config.bad_snapshot_prob > 0:
            equity_series, quality_series = self._apply_gaps(
                equity_series, 
                config.missing_prob, 
                config.bad_snapshot_prob
            )
        
        # Build scenario dict
        scenario = {
            "name": config.name,
            "description": config.description,
            "timeline": timeline,
            "equity_series": equity_series,
            "quality_series": quality_series,
            "config": {
                "start_equity": config.start_equity,
                "pattern": config.pattern,
                "drop_pct": config.drop_pct,
                "minutes": config.minutes
            }
        }
        
        if config.expected_outcome:
            scenario["expected_outcome"] = config.expected_outcome
        
        return scenario
    
    def _flat(self, config: ScenarioConfig) -> List[float]:
        """Flat equity (no drawdown)"""
        return [config.start_equity] * config.minutes
    
    def _step_down(self, config: ScenarioConfig) -> List[float]:
        """Immediate drop, then stable"""
        drop_point = 5  # Drop at 5 minutes
        series = [config.start_equity] * config.minutes
        
        dropped_equity = config.start_equity * (1 - config.drop_pct)
        for i in range(drop_point, config.minutes):
            series[i] = dropped_equity
        
        return series
    
    def _gradual_down(self, config: ScenarioConfig) -> List[float]:
        """Gradual linear decline"""
        series = []
        for i in range(config.minutes):
            progress = i / config.minutes
            equity = config.start_equity * (1 - progress * config.drop_pct)
            series.append(equity)
        return series
    
    def _spike_down_recover(self, config: ScenarioConfig) -> List[float]:
        """Sharp drop then recovery (false positive test)"""
        spike_point = config.minutes // 3
        recovery_point = spike_point + 2
        
        series = [config.start_equity] * config.minutes
        dropped_equity = config.start_equity * (1 - config.drop_pct)
        
        # Spike down
        for i in range(spike_point, recovery_point):
            series[i] = dropped_equity
        
        # Gradual recovery
        for i in range(recovery_point, min(recovery_point + 10, config.minutes)):
            progress = (i - recovery_point) / 10
            series[i] = dropped_equity + (config.start_equity - dropped_equity) * progress
        
        return series
    
    def _noisy(self, config: ScenarioConfig) -> List[float]:
        """Random walk with noise"""
        series = [config.start_equity]
        for i in range(1, config.minutes):
            # Random walk with drift
            drift = -config.drop_pct / config.minutes
            change = (drift + random.gauss(0, config.noise_sigma_pct)) * series[-1]
            new_equity = max(100, series[-1] + change)  # Floor at 100
            series.append(new_equity)
        return series
    
    def _add_noise(self, series: List[float], sigma_pct: float) -> List[float]:
        """Add Gaussian noise to series"""
        noisy = []
        for val in series:
            noise = random.gauss(0, sigma_pct * val)
            noisy.append(max(0, val + noise))
        return noisy
    
    def _apply_gaps(
        self, 
        series: List[float], 
        missing_prob: float, 
        bad_prob: float
    ) -> tuple[List[float], List[bool]]:
        """Apply missing data and bad snapshots"""
        new_series = []
        quality = []
        
        for val in series:
            if random.random() < missing_prob:
                # Missing data point (will be skipped in storage)
                new_series.append(0)
                quality.append(False)
            elif random.random() < bad_prob:
                # Bad snapshot (equity=0 or None)
                new_series.append(0)
                quality.append(False)
            else:
                new_series.append(val)
                quality.append(True)
        
        return new_series, quality


# ==============================================================================
# STANDARD TEST SCENARIOS
# ==============================================================================

def build_standard_scenarios() -> List[Dict[str, Any]]:
    """Build all standard test scenarios"""
    builder = ScenarioBuilder()
    scenarios = []
    
    # 1. Insufficient samples (startup)
    scenarios.append(builder.generate(ScenarioConfig(
        name="startup_insufficient_samples",
        description="Not enough data to calculate DD (should not trigger)",
        start_equity=10000,
        minutes=3,  # Less than min required
        pattern="step_down",
        drop_pct=0.10,
        expected_outcome={
            "tier_triggered": None,
            "reason": "insufficient_samples"
        }
    )))
    
    # 2. Tier A trigger with confirm=2
    scenarios.append(builder.generate(ScenarioConfig(
        name="tier_a_trigger_confirm2",
        description="Tier A should trigger after 2 consecutive confirmations",
        start_equity=10000,
        minutes=20,
        pattern="step_down",
        drop_pct=0.025,  # 2.5% drop
        expected_outcome={
            "tier_triggered": "A",
            "window": "5",
            "after_snapshots": 7  # ~5min window + 2 confirms
        }
    )))
    
    # 3. Tier B trigger (larger drop)
    scenarios.append(builder.generate(ScenarioConfig(
        name="tier_b_trigger_large_drop",
        description="Tier B triggers on sustained 8% drop over 15min",
        start_equity=10000,
        minutes=30,
        pattern="step_down",
        drop_pct=0.085,
        expected_outcome={
            "tier_triggered": "B",
            "window": "15",
            "min_dd": 0.08
        }
    )))
    
    # 4. False spike (should NOT confirm)
    scenarios.append(builder.generate(ScenarioConfig(
        name="false_spike_should_not_confirm",
        description="Brief spike down then recovery (confirm should fail)",
        start_equity=10000,
        minutes=30,
        pattern="spike_down_recover",
        drop_pct=0.05,
        expected_outcome={
            "tier_triggered": None,
            "reason": "confirmation_failed"
        }
    )))
    
    # 5. Gap in data (should not confirm)
    scenarios.append(builder.generate(ScenarioConfig(
        name="gap_in_data_should_not_confirm",
        description="Missing snapshots break confirmation",
        start_equity=10000,
        minutes=30,
        pattern="step_down",
        drop_pct=0.05,
        missing_prob=0.3,  # 30% missing
        expected_outcome={
            "tier_triggered": None,
            "reason": "insufficient_samples_due_to_gaps"
        }
    )))
    
    # 6. Bad snapshots filtered
    scenarios.append(builder.generate(ScenarioConfig(
        name="bad_snapshot_filtered",
        description="Bad snapshots (quality_ok=False) should be ignored",
        start_equity=10000,
        minutes=20,
        pattern="flat",
        bad_snapshot_prob=0.2,
        expected_outcome={
            "tier_triggered": None,
            "snapshots_stored": "reduced"
        }
    )))
    
    # 7. Gradual decline (tests window calculation)
    scenarios.append(builder.generate(ScenarioConfig(
        name="gradual_decline_1h_window",
        description="Slow decline over 1 hour",
        start_equity=10000,
        minutes=75,
        pattern="gradual_down",
        drop_pct=0.10,
        expected_outcome={
            "tier_triggered": "A",
            "window": "60"
        }
    )))
    
    # 8. Noisy market (should be stable with proper filtering)
    scenarios.append(builder.generate(ScenarioConfig(
        name="noisy_market_stable",
        description="Noisy but overall stable (should not trigger)",
        start_equity=10000,
        minutes=30,
        pattern="noisy",
        drop_pct=0.01,  # Minimal drift
        noise_sigma_pct=0.02,
        expected_outcome={
            "tier_triggered": None
        }
    )))
    
    # 9. Multi-window breach (strongest wins)
    scenarios.append(builder.generate(ScenarioConfig(
        name="multi_window_strongest_wins",
        description="Multiple windows breach, strongest ratio should win",
        start_equity=10000,
        minutes=25,
        pattern="step_down",
        drop_pct=0.03,
        expected_outcome={
            "tier_triggered": "A",
            "selection": "strongest_breach"
        }
    )))
    
    # 10. Large crash (Tier B priority)
    scenarios.append(builder.generate(ScenarioConfig(
        name="crash_tier_b_priority",
        description="Both tiers breach, Tier B should win (priority)",
        start_equity=10000,
        minutes=20,
        pattern="step_down",
        drop_pct=0.15,  # 15% crash
        expected_outcome={
            "tier_triggered": "B",
            "reason": "tier_b_takes_priority"
        }
    )))
    
    return scenarios


def build_action_scenarios() -> List[Dict[str, Any]]:
    """Build scenarios for action/execution testing"""
    scenarios = []
    
    # Futures scenarios
    scenarios.append({
        "name": "futures_close_longs_only",
        "description": "One-way mode: close longs, keep shorts",
        "mode": "futures",
        "exchange": "bybit",
        "position_mode": "one-way",
        "positions": [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.5}
        ],
        "action": "CLOSE_LONGS_ONLY",
        "expected_orders": 1,
        "expected_closed": ["BTC/USDT:USDT long"]
    })
    
    scenarios.append({
        "name": "futures_close_all_positions",
        "description": "Close all positions (longs + shorts)",
        "mode": "futures",
        "exchange": "binance",
        "position_mode": "one-way",
        "positions": [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.05},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.0},
            {"symbol": "SOL/USDT:USDT", "side": "long", "contracts": 5.0}
        ],
        "action": "CLOSE_ALL_POSITIONS",
        "expected_orders": 3,
        "expected_closed": "all"
    })
    
    scenarios.append({
        "name": "bybit_hedge_mode_fallback",
        "description": "Bybit hedge mode requires positionIdx",
        "mode": "futures",
        "exchange": "bybit",
        "position_mode": "hedge",
        "positions": [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1, "positionIdx": 1},
            {"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 0.05, "positionIdx": 2}
        ],
        "action": "CLOSE_LONGS_ONLY",
        "expected_orders": 1,
        "requires_position_idx": True
    })
    
    # Spot scenarios
    scenarios.append({
        "name": "spot_sell_blacklist_only",
        "description": "Sell only blacklist coins, keep stables and others",
        "mode": "spot",
        "balances": {
            "USDT": 1000,
            "BTC": 0.1,
            "DOGE": 5000,
            "USDC": 500
        },
        "stables_keep": ["USDT", "USDC"],
        "blacklist": ["DOGE"],
        "action": "SELL_BLACKLIST_ONLY_KEEP_STABLES",
        "expected_orders": 1,
        "expected_sold": ["DOGE"]
    })
    
    scenarios.append({
        "name": "spot_sell_all_non_usdt",
        "description": "Sell everything except stables",
        "mode": "spot",
        "balances": {
            "USDT": 1000,
            "BTC": 0.05,
            "SOL": 10,
            "USDC": 200
        },
        "stables_keep": ["USDT", "USDC"],
        "blacklist": None,
        "action": "SELL_ALL_NON_USDT_KEEP_STABLES",
        "expected_orders": 2,
        "expected_sold": ["BTC", "SOL"]
    })
    
    scenarios.append({
        "name": "spot_multihop_routing",
        "description": "ALT with no direct USDT pair uses BTC routing",
        "mode": "spot",
        "balances": {
            "USDT": 1000,
            "SHIB": 1000000  # Assume SHIB has no SHIB/USDT but has SHIB/BTC
        },
        "pairs_available": ["SHIB/BTC", "BTC/USDT"],
        "stables_keep": ["USDT"],
        "action": "SELL_ALL_NON_USDT_KEEP_STABLES",
        "expected_orders": 2,  # SHIB/BTC then BTC/USDT
        "expected_routing": "SHIB->BTC->USDT"
    })
    
    return scenarios


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate test scenarios for killswitch")
    parser.add_argument("--out", default="tests/scenarios", help="Output directory")
    parser.add_argument("--type", choices=["all", "dd", "action"], default="all", 
                       help="Scenario type to generate")
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    
    if args.type in ["all", "dd"]:
        print("Generating drawdown scenarios...")
        dd_scenarios = build_standard_scenarios()
        for scenario in dd_scenarios:
            filename = output_dir / f"{scenario['name']}.json"
            with open(filename, 'w') as f:
                json.dump(scenario, f, indent=2)
            print(f"  ✓ {filename}")
            generated_count += 1
    
    if args.type in ["all", "action"]:
        print("Generating action scenarios...")
        action_scenarios = build_action_scenarios()
        for scenario in action_scenarios:
            filename = output_dir / f"{scenario['name']}.json"
            with open(filename, 'w') as f:
                json.dump(scenario, f, indent=2)
            print(f"  ✓ {filename}")
            generated_count += 1
    
    print(f"\n✅ Generated {generated_count} scenarios in {output_dir}")


if __name__ == "__main__":
    main()
