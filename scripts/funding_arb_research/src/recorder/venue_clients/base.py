"""Abstract async WS venue client.

Each subclass owns the venue-specific URL, subscribe payload, and
message parser. Reconnect logic, ping handling, and writer dispatch are
factored into this base class so venue subclasses stay small.

Canonical channel names emitted by all venues:

  ``book``    — orderbook diff/snapshot update
  ``trades``  — public trade
  ``mark``    — mark/index/funding tick (mark_price, index_price,
                                          funding_rate, next_funding_time)
  ``funding`` — explicit funding event (rare; most venues bundle into mark)

Required row keys: ``timestamp_utc`` (event time, UTC ISO or pd.Timestamp),
``recv_ts_utc`` (local receipt time). The diff is the publication-latency
proxy.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import orjson

from ..writer import ParquetWriter
from ...utils.logging import get_logger


@dataclass
class VenueClientConfig:
    venue: str
    coins: list[str]
    ws_url: str
    ping_interval_s: float = 20.0
    reconnect_backoff_s: float = 2.0
    reconnect_backoff_max_s: float = 60.0


class VenueClient(ABC):
    """Long-lived async WS client for one venue. ``run()`` never returns."""

    def __init__(self, cfg: VenueClientConfig, writer: ParquetWriter) -> None:
        self.cfg = cfg
        self.writer = writer
        self.log = get_logger(f"funding_arb.recorder.{cfg.venue}")
        self.msgs_received = 0
        self.last_msg_ts: float = 0.0

    @abstractmethod
    def venue_symbol(self, coin: str) -> str: ...

    @abstractmethod
    def subscribe_payloads(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def handle_message(self, msg: dict[str, Any], recv_ts_utc: float) -> None: ...

    def text_ping_payload(self) -> Optional[str]:
        """Override to send a periodic text-frame ping (e.g. Bitget 'ping')."""
        return None

    async def run(self) -> None:
        backoff = self.cfg.reconnect_backoff_s
        while True:
            try:
                await self._run_once()
                backoff = self.cfg.reconnect_backoff_s
            except asyncio.CancelledError:
                self.log.info("cancelled, exiting")
                raise
            except Exception:  # noqa: BLE001
                self.log.exception("ws session error; reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.cfg.reconnect_backoff_max_s)

    async def _run_once(self) -> None:
        self.log.info("connecting to %s", self.cfg.ws_url)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.ws_connect(
                self.cfg.ws_url,
                heartbeat=self.cfg.ping_interval_s,
                autoping=True,
                receive_timeout=self.cfg.ping_interval_s * 3,
            ) as ws:
                for payload in self.subscribe_payloads():
                    await ws.send_json(payload)
                self.log.info("subscribed coins=%s n=%d", self.cfg.coins, len(self.subscribe_payloads()))
                ping_task: Optional[asyncio.Task] = None
                text_ping = self.text_ping_payload()
                if text_ping is not None:
                    ping_task = asyncio.create_task(
                        self._text_ping_loop(ws, text_ping), name=f"{self.cfg.venue}-ping"
                    )
                try:
                    await self._read_loop(ws)
                finally:
                    if ping_task is not None:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

    async def _text_ping_loop(self, ws: aiohttp.ClientWebSocketResponse, payload: str) -> None:
        try:
            while not ws.closed:
                await asyncio.sleep(self.cfg.ping_interval_s)
                await ws.send_str(payload)
        except (asyncio.CancelledError, ConnectionError):
            return
        except Exception:  # noqa: BLE001
            self.log.debug("text-ping loop exiting", exc_info=True)

    async def _read_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for raw in ws:
            if raw.type == aiohttp.WSMsgType.TEXT:
                recv = time.time()
                self.last_msg_ts = recv
                self.msgs_received += 1
                if raw.data == "pong":
                    continue
                try:
                    data = orjson.loads(raw.data)
                except Exception:
                    self.log.warning("non-json text msg: %.200s", raw.data)
                    continue
                try:
                    self.handle_message(data, recv)
                except Exception:  # noqa: BLE001
                    self.log.exception("handle_message failed: %.300s", str(data))
            elif raw.type == aiohttp.WSMsgType.BINARY:
                continue
            elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise RuntimeError(f"ws closed: type={raw.type} data={raw.data}")
