"""Synthetic unit contracts; no strategies, live clients, YAML config, or backtests.

All values are fictional and intentionally unrelated to published venue schedules.
The tests validate accounting/software conventions, not economic validity.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.funding_arb_research.src.backtest.costs import (  # noqa: E402
    CostModel,
    FeeSchedule,
    SlippageConfig,
)
from scripts.funding_arb_research.src.backtest.funding_settlement import (  # noqa: E402
    funding_pnl,
    select_predicted_funding,
)
from scripts.funding_arb_research.src.normalization.schema import (  # noqa: E402
    conform_funding,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/synthetic_funding.json").read_text())


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame(FIXTURE["rows"])
        self.panel["timestamp_utc"] = pd.to_datetime(self.panel["timestamp_utc"], utc=True)
        self.times = self.panel["timestamp_utc"].tolist()

    def settle(self, start, end, direction=1, notional=10_000.0):
        return funding_pnl(
            self.panel, direction=direction, notional_usd=notional,
            venue="fictional_venue", symbol="SYNTH", start=start, end=end,
        )

    def test_start_excluded_end_included_and_input_unchanged(self):
        before = self.panel.copy(deep=True)
        entries = self.settle(self.times[0], self.times[2])
        self.assertEqual([e.timestamp for e in entries], self.times[1:3])
        self.assertEqual([e.pnl_usd for e in entries], [20.0, -5.0])
        self.assertTrue(all(e.venue == "fictional_venue" and e.symbol == "SYNTH" for e in entries))
        pd.testing.assert_frame_equal(self.panel, before)

    def test_adjacent_windows_do_not_double_book_shared_boundary(self):
        left = self.settle(self.times[0], self.times[1])
        right = self.settle(self.times[1], self.times[2])
        whole = self.settle(self.times[0], self.times[2])
        self.assertEqual(left + right, whole)
        self.assertEqual(sum(e.pnl_usd for e in whole), 15.0)
        self.assertEqual(len({e.timestamp for e in left + right}), 2)

    def test_opposite_directions_cancel_and_notional_scales(self):
        longs = self.settle(self.times[0], self.times[2])
        shorts = self.settle(self.times[0], self.times[2], direction=-1)
        doubled = self.settle(self.times[0], self.times[2], notional=20_000)
        self.assertEqual([e.pnl_usd for e in shorts], [-20.0, 5.0])
        for long, short, double in zip(longs, shorts, doubled, strict=True):
            self.assertEqual(long.pnl_usd + short.pnl_usd, 0)
            self.assertEqual(double.pnl_usd, 2 * long.pnl_usd)

    def test_empty_and_zero_length_windows_book_nothing(self):
        self.assertEqual(self.settle(self.times[1], self.times[1]), [])
        self.assertEqual(self.settle(self.times[-1], self.times[-1] + pd.Timedelta(hours=1)), [])
        self.panel = self.panel.iloc[:0]
        self.assertEqual(self.settle(self.times[0], self.times[-1]), [])

    def test_zero_notional_has_zero_pnl(self):
        entries = self.settle(self.times[0], self.times[2], notional=0)
        self.assertEqual([e.pnl_usd for e in entries], [0.0, 0.0])

    def test_prediction_is_strictly_prior_and_skips_missing_predictions(self):
        # The 16:00 prediction is not available at exactly 16:00; 08:00 has none.
        self.assertEqual(select_predicted_funding(self.panel, as_of=self.times[2]), 0.003)
        self.assertEqual(select_predicted_funding(self.panel, as_of=self.times[3]), 0.004)
        self.assertIsNone(select_predicted_funding(self.panel, as_of=self.times[0]))

    def test_realized_fallback_and_no_available_history(self):
        for panel in (
            self.panel.drop(columns="predicted_funding_rate"),
            self.panel.assign(predicted_funding_rate=float("nan")),
        ):
            self.assertEqual(select_predicted_funding(panel, as_of=self.times[2]), -0.002)
        self.assertIsNone(select_predicted_funding(self.panel.iloc[:0], as_of=self.times[2]))
        missing = self.panel.assign(predicted_funding_rate=float("nan"), funding_rate=float("nan"))
        self.assertIsNone(select_predicted_funding(missing, as_of=self.times[2]))

    def test_normalization_sorts_before_latest_prediction_selection(self):
        # Selection expects ordered input; the public normalizer supplies it.
        unordered = self.panel.iloc[[2, 0, 3, 1]].copy()
        normalized = conform_funding(unordered)
        self.assertTrue(normalized["timestamp_utc"].is_monotonic_increasing)
        self.assertEqual(select_predicted_funding(normalized, as_of=self.times[3]), 0.004)


class CostTests(unittest.TestCase):
    def setUp(self):
        self.model = CostModel(
            fees={venue: FeeSchedule(**fees) for venue, fees in FIXTURE["fees"].items()},
            slippage=SlippageConfig(**FIXTURE["slippage"]),
            gmx_open_close_bps=9.0,
        )

    def test_fee_roles_rebate_unknown_venue_and_gmx_aliases(self):
        self.assertEqual(self.model.fee_bps("synthetic_a"), 3.0)
        self.assertEqual(self.model.fee_bps("synthetic_a", "maker"), -1.0)
        self.assertEqual(self.model.round_trip_fee_bps("synthetic_a", "maker"), -2.0)
        for venue in ("gmx", "gmx_v2"):
            self.assertEqual(self.model.fee_bps(venue), 9.0)
            self.assertEqual(self.model.round_trip_fee_bps(venue), 18.0)
        with self.assertRaises(KeyError):
            self.model.fee_bps("unknown_synthetic_venue")

    def test_missing_or_nonpositive_depth_or_order_use_fallback(self):
        for order, depth in ((100, None), (100, 0), (100, -1), (0, 1000), (-1, 1000)):
            with self.subTest(order=order, depth=depth):
                self.assertEqual(self.model.slippage_bps(order, depth), 7.0)

    def test_slippage_bps_floor_square_root_scaling_and_depth_monotonicity(self):
        # Fictional k=0.01, size/depth=0.01 => 0.001 fraction = 10 bps.
        self.assertAlmostEqual(self.model.slippage_bps(100, 10_000), 10.0)
        self.assertAlmostEqual(self.model.slippage_bps(400, 10_000), 20.0)
        self.assertAlmostEqual(self.model.slippage_bps(100, 40_000), 5.0)
        self.assertEqual(self.model.slippage_bps(1, 1_000_000), 2.0)

    def test_two_leg_round_trip_charges_all_four_sides_in_bps(self):
        # A: (3 fee + 10 slip) twice; B: (5 fee + 10 slip) twice = 56 bps.
        cost = self.model.two_leg_round_trip_bps(
            "synthetic_a", "synthetic_b", notional=100,
            depth_a=10_000, depth_b=10_000,
        )
        self.assertAlmostEqual(cost, 56.0)
        self.assertAlmostEqual(self.model.entry_cost_bps(
            "synthetic_a", order_notional=100, depth_notional=10_000,
        ), 13.0)
        self.assertAlmostEqual(self.model.exit_cost_bps(
            "synthetic_b", order_notional=100, depth_notional=10_000,
        ), 15.0)


if __name__ == "__main__":
    unittest.main()
