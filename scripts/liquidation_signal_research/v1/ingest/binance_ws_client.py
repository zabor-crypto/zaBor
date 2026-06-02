"""Binance websocket ingestion client for mandatory v1 streams."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple

import websockets

try:
    import orjson as _orjson
    def _json_loads(s: str) -> dict:  # type: ignore[return]
        return _orjson.loads(s)
    def _json_dumps(obj: object) -> str:
        return _orjson.dumps(obj, option=_orjson.OPT_NON_STR_KEYS).decode()
except ImportError:  # pragma: no cover
    def _json_loads(s: str) -> dict:  # type: ignore[return]
        return json.loads(s)
    def _json_dumps(obj: object) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)

from v1.contracts.event_types import (
    SCHEMA_VERSION_V1,
    STREAM_AGG_TRADE,
    STREAM_DEPTH,
    STREAM_FORCE_ORDER,
    STREAM_MARK_PRICE,
    VENUE_BINANCE_UM,
    RawMarketEvent,
)
from v1.contracts.ids import MonotonicSequencer, make_raw_event_id, stable_payload_hash

logger = logging.getLogger(__name__)

OnRawEvent = Callable[[RawMarketEvent], Optional[Awaitable[None]]]


@dataclass(frozen=True)
class StreamSpec:
    stream: str
    suffix: str
    # If set, subscribe once as a market-wide stream instead of per-symbol.
    global_suffix: Optional[str] = None


STREAM_SPECS = {
    # forceOrder is a market-wide stream on Binance Futures: one subscription
    # delivers liquidations for all symbols, with the symbol inside data["o"]["s"].
    # Per-symbol <sym>@forceOrder subscriptions are silently ignored by Binance
    # when no liquidation occurs for that symbol, making them unreliable.
    STREAM_FORCE_ORDER: StreamSpec(stream=STREAM_FORCE_ORDER, suffix="forceOrder", global_suffix="!forceOrder@arr"),
    STREAM_AGG_TRADE: StreamSpec(stream=STREAM_AGG_TRADE, suffix="trade"),
    STREAM_DEPTH: StreamSpec(stream=STREAM_DEPTH, suffix="depth@100ms"),
    STREAM_MARK_PRICE: StreamSpec(stream=STREAM_MARK_PRICE, suffix="markPrice@1s"),
}


class BinanceWsClient:
    def __init__(
        self,
        *,
        symbols: Iterable[str],
        streams: Iterable[str],
        on_raw_event: OnRawEvent,
        endpoint: str = "wss://fstream.binance.com/stream",
        reconnect_min_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        local_addr: Optional[str] = None,
        socks5_proxy: Optional[str] = None,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.streams = list(streams)
        self.on_raw_event = on_raw_event
        self.endpoint = endpoint
        self.reconnect_min_seconds = reconnect_min_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self.local_addr = local_addr
        self.socks5_proxy = socks5_proxy

        self._running = False
        self._ingest_seq = MonotonicSequencer()
        self._stream_local_seq: Dict[Tuple[str, str], int] = {}

    def _combined_stream_url(self) -> str:
        names = []
        global_streams_added: set = set()
        for symbol in self.symbols:
            lower = symbol.lower()
            for stream in self.streams:
                spec = STREAM_SPECS[stream]
                if spec.global_suffix is not None:
                    if stream not in global_streams_added:
                        names.append(spec.global_suffix)
                        global_streams_added.add(stream)
                else:
                    names.append(f"{lower}@{spec.suffix}")
        return f"{self.endpoint}?streams={'/'.join(names)}"

    @staticmethod
    def _normalize_stream_name(stream_name: str) -> str:
        if "forceOrder" in stream_name:
            return STREAM_FORCE_ORDER
        if "@aggTrade" in stream_name:
            return STREAM_AGG_TRADE
        if "@trade" in stream_name:
            return STREAM_AGG_TRADE
        if "@depth" in stream_name:
            return STREAM_DEPTH
        if "@markPrice" in stream_name:
            return STREAM_MARK_PRICE
        return stream_name

    @staticmethod
    def _extract_symbol(data: dict, stream_name: str) -> str:
        symbol = data.get("s")
        if symbol:
            return str(symbol).upper()
        # !forceOrder@arr delivers symbol in data["o"]["s"], not at top level.
        if "forceOrder" in stream_name and isinstance(data.get("o"), dict):
            sym = data["o"].get("s")
            if sym:
                return str(sym).upper()
        lhs = stream_name.split("@", 1)[0]
        return lhs.upper()

    @staticmethod
    def _extract_exchange_ts_ms(data: dict, stream: str) -> int:
        for key in ("E", "T", "t"):
            if key in data:
                return int(data[key])
        if stream == STREAM_FORCE_ORDER and isinstance(data.get("o"), dict):
            inner = data["o"]
            for key in ("T", "E"):
                if key in inner:
                    return int(inner[key])
        return int(time.time() * 1000)

    async def _emit(self, event: RawMarketEvent) -> None:
        result = self.on_raw_event(event)
        if inspect.isawaitable(result):
            await result

    async def _handle_message(self, conn_id: str, message: str) -> None:
        recv_wall_ts_ns = time.time_ns()
        recv_mono_ts_ns = time.monotonic_ns()

        try:
            payload = _json_loads(message)
            wrapped_stream = str(payload.get("stream", ""))
            data = payload.get("data", payload)
            stream = self._normalize_stream_name(wrapped_stream)
            symbol = self._extract_symbol(data, wrapped_stream)
            exchange_ts_ms = self._extract_exchange_ts_ms(data, stream)
            payload_hash = stable_payload_hash(payload)

            key = (symbol, stream)
            stream_local_seq = self._stream_local_seq.get(key, 0) + 1
            self._stream_local_seq[key] = stream_local_seq

            ingest_seq = self._ingest_seq.next()
            event = RawMarketEvent(
                event_id=make_raw_event_id(
                    VENUE_BINANCE_UM,
                    stream,
                    symbol,
                    exchange_ts_ms,
                    ingest_seq,
                    payload_hash,
                ),
                schema_version=SCHEMA_VERSION_V1,
                venue=VENUE_BINANCE_UM,
                stream=stream,
                symbol=symbol,
                conn_id=conn_id,
                exchange_ts_ms=exchange_ts_ms,
                recv_wall_ts_ns=recv_wall_ts_ns,
                recv_mono_ts_ns=recv_mono_ts_ns,
                ingest_seq=ingest_seq,
                stream_local_seq=stream_local_seq,
                payload_json=_json_dumps(payload),
                payload_hash=payload_hash,
                parse_status="ok",
                gap_flag=False,
            )
            await self._emit(event)
        except Exception:
            ingest_seq = self._ingest_seq.next()
            fallback_hash = stable_payload_hash({"raw": message})
            error_event = RawMarketEvent(
                event_id=make_raw_event_id(
                    VENUE_BINANCE_UM,
                    "parse_error",
                    "UNKNOWN",
                    int(time.time() * 1000),
                    ingest_seq,
                    fallback_hash,
                ),
                schema_version=SCHEMA_VERSION_V1,
                venue=VENUE_BINANCE_UM,
                stream="parse_error",
                symbol="UNKNOWN",
                conn_id=conn_id,
                exchange_ts_ms=int(time.time() * 1000),
                recv_wall_ts_ns=recv_wall_ts_ns,
                recv_mono_ts_ns=recv_mono_ts_ns,
                ingest_seq=ingest_seq,
                stream_local_seq=0,
                payload_json=message,
                payload_hash=fallback_hash,
                parse_status="parse_error",
                gap_flag=True,
            )
            await self._emit(error_event)

    async def _run_via_proxy(self, conn_id: str, url: str) -> None:
        try:
            from aiohttp_socks import ProxyConnector  # type: ignore[import]
            import aiohttp
        except ImportError as exc:
            raise RuntimeError(
                "socks5_proxy requires 'aiohttp' and 'aiohttp-socks': "
                "pip install aiohttp aiohttp-socks"
            ) from exc

        connector = ProxyConnector.from_url(self.socks5_proxy)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(conn_id, msg.data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    async def run_forever(self) -> None:
        self._running = True
        backoff = self.reconnect_min_seconds

        while self._running:
            conn_id = f"ws-{uuid.uuid4().hex[:12]}"
            url = self._combined_stream_url()
            try:
                logger.info("Connecting Binance WS: conn_id=%s", conn_id)
                if self.socks5_proxy:
                    await self._run_via_proxy(conn_id, url)
                    backoff = self.reconnect_min_seconds
                else:
                    connect_kwargs: Dict[str, object] = {
                        "ping_interval": 20,
                        "ping_timeout": 20,
                        "close_timeout": 10,
                        "max_queue": 2048,
                    }
                    if self.local_addr:
                        connect_kwargs["local_addr"] = (self.local_addr, 0)
                    async with websockets.connect(url, **connect_kwargs) as ws:
                        backoff = self.reconnect_min_seconds
                        async for message in ws:
                            if not self._running:
                                break
                            await self._handle_message(conn_id, message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Binance WS disconnected (conn_id=%s): %s. reconnect_in=%.1fs",
                    conn_id,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)

    def stop(self) -> None:
        self._running = False
