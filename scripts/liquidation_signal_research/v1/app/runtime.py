"""Runtime configuration contracts for continuation v1 app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from v1.contracts.event_types import MVP_SYMBOLS, MANDATORY_STREAMS


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    tick_size: float
    lot_step: float
    min_qty: float
    max_spread_bps: float


@dataclass
class RuntimeConfig:
    symbols: List[str] = field(default_factory=lambda: list(MVP_SYMBOLS))
    streams: List[str] = field(default_factory=lambda: list(MANDATORY_STREAMS))
    data_root: Path = Path("v1_data/raw")
    telemetry_root: Path = Path("v1_data/telemetry")
    replay_root: Path = Path("v1_data/raw")
    config_version: str = "v1.0.0"
    decision_interval_ms: int = 250
    auto_snapshot_sync: bool = True
    snapshot_depth_limit: int = 1000
    snapshot_fetch_timeout_seconds: float = 6.0
    snapshot_resync_cooldown_ms: int = 1500
    depth_buffer_max_events: int = 5000
    book_stale_after_ms: int = 2000
    execution_paper_mode: bool = True
    binance_api_key: str = ""
    binance_api_secret: str = ""
    execution_base_url: str = "https://fapi.binance.com"
    execution_user_stream_ws_base_url: str = "wss://fstream.binance.com/ws"
    ws_local_addr: str = ""
    ws_socks5_proxy: str = ""
    shadow_mode: bool = False
    enable_paper_calibration_gate: bool = False
    calibration_overrides_path: Optional[Path] = None

    # ── Performance tuning ────────────────────────────────────────────────────

    # Raw capture writer queue depth.  Lower values reduce peak RSS; at 100
    # events/sec, 10 000 provides ~100 s of back-pressure for startup bursts.
    capture_queue_size: int = 10_000

    # Flush raw partition data to OS every N appended records (FLUSH_BLOCK).
    # Lower = more durable but more OS writes.  500 ≈ every 5 s at 100 ev/s.
    store_flush_interval_events: int = 500

    # Telemetry write-buffer flush cadence for high-frequency logs (feature_log,
    # decision_log).  Flush every N writes.  50 ≈ every 3 s at idle cadence.
    telemetry_feature_flush_interval: int = 50
    telemetry_decision_flush_interval: int = 50

    # Verbosity of telemetry logs:
    #   "full"    – log every feature snapshot and every decision (default)
    #   "summary" – log only ENTER_LONG/SHORT decisions; suppress routine NO_TRADEs
    #               and feature snapshots except during liq-active periods
    #   "off"     – suppress feature_log and decision_log entirely
    telemetry_log_level: str = "full"

    # Maximum number of price levels kept per side in the local order book.
    # 0 = unlimited (existing behaviour).  Setting 200 reduces book memory from
    # ~5 MB to ~200 KB per symbol with negligible impact on spread/depth features.
    max_book_depth_levels: int = 0

    # ── Maker-mode investigation (Phases 1-3) ────────────────────────────────
    # Phase 1: paper maker entry simulation.
    #   When True: entry orders are placed at the passive side (bid for LONG, ask
    #   for SHORT) instead of crossing.  Fill is simulated when an aggTrade occurs
    #   at or through the limit price within maker_entry_timeout_ms.  Misses are
    #   logged as maker_miss events and the entry is skipped.  Entry fee uses the
    #   maker rate (MAKER_FEE_RATE) instead of taker rate.
    maker_entry_mode: bool = False
    maker_entry_timeout_ms: int = 300          # cancel if no fill within 300ms

    # Phase 3: maker exit for trailing stops.
    #   When True: trailing_stop exits are treated as maker (passive) orders in
    #   the fee model — price crossing our trailing floor from above is inherently
    #   a passive fill.  Exit fee uses maker rate instead of taker rate for these
    #   exits only.  hard_stop / time_limit / flow_reversal exits remain taker.
    maker_exit_trailing: bool = False

    # ── May 20 2026: regime-soft mode (signal-without-liq validation) ────────
    # When regime_liq_optional=True, the regime gate skips:
    #   - insufficient_liquidation_stress (liq_total_notional_1s threshold)
    #   - liq_flow_misaligned (direction-confirmation requirement)
    #   - liq_in_dead_zone ($50K–$200K cascade band)
    # When regime_session_optional=True, the gate skips outside_us_session.
    # Together these let the signal model run on flow+depth+price alone so we
    # can measure score distribution during forceOrder-dormant periods.
    regime_liq_optional: bool = False
    regime_session_optional: bool = False
    # When True, skip ML quality gate entirely (no calibration loaded, no rejection).
    # Use during regime-soft validation: stale ML calibration from prior data
    # distribution will score all new (non-liq) features at ~0 and block all ENTERs.
    disable_ml_quality_gate: bool = False

    # ── May 21 2026: paper-mode loss-cap override ────────────────────────────
    # Default RiskLimits caps (6 / 4) are calibrated for live mode; during
    # paper-mode data gathering a transient loss cluster can lock the bot out
    # of trading for hours because the cap is self-locking (no entries → no
    # wins → counter never resets). Bump these to large values during paper
    # validation to keep entries flowing.
    risk_max_consecutive_losses: int = 6
    risk_max_symbol_consecutive_losses: int = 4

    # ── May 21 2026: paper-mode kill-switch drawdown override ────────────────
    # Value is a FRACTION (0.08 = 8%), matching all other *_fraction fields on
    # RiskLimits. engine.py stores session_state.drawdown_pct as a fraction too.
    # Default 0.08 is calibrated for live mode (8% nav drawdown = real floor).
    # In paper mode the gate self-locks (drawdown can only recover via new
    # closes; closes are blocked while the gate fires). Set to a high fraction
    # (e.g. 0.99 = 99%) during paper validation so we keep producing outcome
    # data for exit-calibration analysis. Do NOT use >1.0 — it bypasses the gate
    # but signals that the unit convention was misunderstood.
    kill_switch_max_drawdown_pct: float = 0.08


DEFAULT_SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "BTCUSDT": SymbolSpec("BTCUSDT", tick_size=0.1, lot_step=0.001, min_qty=0.001, max_spread_bps=3.0),
    "ETHUSDT": SymbolSpec("ETHUSDT", tick_size=0.01, lot_step=0.001, min_qty=0.001, max_spread_bps=4.0),
    "SOLUSDT": SymbolSpec("SOLUSDT", tick_size=0.001, lot_step=0.01, min_qty=0.01, max_spread_bps=5.0),
    "HYPEUSDT": SymbolSpec("HYPEUSDT", tick_size=0.0001, lot_step=0.1, min_qty=0.1, max_spread_bps=7.0),
}
