"""MVP-1: Hyperliquid vs Binance funding-interval arbitrage.

Convention used here:

  * Hyperliquid funds hourly, Binance every 8 hours. The strategy compares
    *annualized* expected funding APR per venue, and enters a delta-neutral
    spread if abs(spread_apr) > entry_threshold_apr.
  * The "expected" rate prefers a venue's predicted_funding_rate; if that's
    not available, it falls back to the most recent realized rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..backtest.engine import CloseAll, OpenLeg, StrategyState
from ..backtest.funding_settlement import select_predicted_funding
from ..utils.time import annualize_funding


@dataclass
class HLBinanceState:
    last_sign_flip_at: Optional[pd.Timestamp] = None
    last_sign: int = 0
    last_entry_at: Optional[pd.Timestamp] = None
    cooldown_minutes: int = 5


def _expected_apr(panel: pd.DataFrame, *, as_of: pd.Timestamp, interval_h: int) -> Optional[float]:
    rate = select_predicted_funding(panel, as_of=as_of)
    if rate is None:
        return None
    return annualize_funding(rate, interval_h)


class HLBinanceIntervalStrategy:
    """Stateful callable usable as ``BacktestEngine.run(strategy, ...)``."""

    def __init__(self, *, hl_interval_h: int = 1, binance_interval_h: int = 8):
        self.hl_interval_h = hl_interval_h
        self.binance_interval_h = binance_interval_h
        self._state_per_symbol: dict[str, HLBinanceState] = {}

    def _state(self, symbol: str) -> HLBinanceState:
        st = self._state_per_symbol.get(symbol)
        if st is None:
            st = HLBinanceState()
            self._state_per_symbol[symbol] = st
        return st

    @staticmethod
    def _binance_symbol(coin: str) -> str:
        # BTC -> BTCUSDT, ETH -> ETHUSDT, SOL -> SOLUSDT
        return f"{coin}USDT"

    def __call__(self, state: StrategyState):
        cfg = state.config
        universe: list[str] = list(cfg.get("universe", ["BTC", "ETH", "SOL"]))
        entry_th = float(cfg.get("entry_threshold_apr", 0.12))
        exit_th = float(cfg.get("exit_threshold_apr", 0.02))
        sign_flip_minutes = int(cfg.get("sign_flip_grace_minutes", 15))
        per_trade_pct = float(cfg.get("per_trade_capital_pct", 0.10))

        # If we already have legs open, decide whether to exit.
        if state.open_legs:
            return self._maybe_exit(state, entry_th, exit_th, sign_flip_minutes)

        # Otherwise, evaluate each symbol and pick the strongest signal.
        best_symbol = None
        best_spread = 0.0
        for coin in universe:
            hl_panel = state.funding.get(("hyperliquid", coin), pd.DataFrame())
            bi_panel = state.funding.get(("binance", self._binance_symbol(coin)), pd.DataFrame())
            if hl_panel.empty or bi_panel.empty:
                continue
            hl_apr = _expected_apr(hl_panel, as_of=state.now, interval_h=self.hl_interval_h)
            bi_apr = _expected_apr(bi_panel, as_of=state.now, interval_h=self.binance_interval_h)
            if hl_apr is None or bi_apr is None:
                continue
            spread = hl_apr - bi_apr
            if abs(spread) > abs(best_spread):
                best_spread = spread
                best_symbol = coin

        if best_symbol is None or abs(best_spread) <= entry_th:
            return None

        per_leg_notional = state.equity_usd * per_trade_pct
        binance_sym = self._binance_symbol(best_symbol)
        if best_spread > 0:
            # HL funding > Binance funding -> short HL, long Binance
            legs = [
                OpenLeg(venue="hyperliquid", symbol=best_symbol, direction=-1,
                        notional_usd=per_leg_notional, role=cfg.get("role", "taker"),
                        tag=f"spread_apr={best_spread:.4f}"),
                OpenLeg(venue="binance", symbol=binance_sym, direction=+1,
                        notional_usd=per_leg_notional, role=cfg.get("role", "taker"),
                        tag=f"spread_apr={best_spread:.4f}"),
            ]
        else:
            # HL funding < Binance funding -> long HL, short Binance
            legs = [
                OpenLeg(venue="hyperliquid", symbol=best_symbol, direction=+1,
                        notional_usd=per_leg_notional, role=cfg.get("role", "taker"),
                        tag=f"spread_apr={best_spread:.4f}"),
                OpenLeg(venue="binance", symbol=binance_sym, direction=-1,
                        notional_usd=per_leg_notional, role=cfg.get("role", "taker"),
                        tag=f"spread_apr={best_spread:.4f}"),
            ]
        st = self._state(best_symbol)
        st.last_entry_at = state.now
        st.last_sign = 1 if best_spread > 0 else -1
        st.last_sign_flip_at = None
        return legs

    # ------------------------------------------------------------------ #

    def _maybe_exit(
        self,
        state: StrategyState,
        entry_th: float,
        exit_th: float,
        sign_flip_minutes: int,
    ):
        # Exit decisions are made on the open symbol(s).
        # Our entries are always pairs on a single coin, so just look at the
        # first leg to identify the active coin.
        active = state.open_legs[0]
        coin = active.symbol if active.venue == "hyperliquid" else active.symbol[:-4]
        binance_sym = self._binance_symbol(coin)

        hl_panel = state.funding.get(("hyperliquid", coin), pd.DataFrame())
        bi_panel = state.funding.get(("binance", binance_sym), pd.DataFrame())
        if hl_panel.empty or bi_panel.empty:
            return None

        hl_apr = _expected_apr(hl_panel, as_of=state.now, interval_h=self.hl_interval_h)
        bi_apr = _expected_apr(bi_panel, as_of=state.now, interval_h=self.binance_interval_h)
        if hl_apr is None or bi_apr is None:
            return None
        spread = hl_apr - bi_apr

        st = self._state(coin)
        cur_sign = 1 if spread > 0 else (-1 if spread < 0 else 0)

        if cur_sign != st.last_sign and cur_sign != 0:
            if st.last_sign_flip_at is None:
                st.last_sign_flip_at = state.now
            elif (state.now - st.last_sign_flip_at) >= pd.Timedelta(minutes=sign_flip_minutes):
                return CloseAll(reason="sign_flip_persistent")
        else:
            st.last_sign_flip_at = None

        if abs(spread) < exit_th:
            return CloseAll(reason="spread_below_exit")

        return None
