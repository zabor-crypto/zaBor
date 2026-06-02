"""Binance UM futures user-data websocket client for order reconciliation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import urllib.request
from typing import Awaitable, Callable, Dict, Optional

import websockets

from v1._fapi import safe_urlopen

logger = logging.getLogger(__name__)

OnUserData = Callable[[Dict[str, object]], Optional[Awaitable[None]]]


class BinanceUserDataStream:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        ws_base_url: str = "wss://fstream.binance.com/ws",
        keepalive_interval_seconds: int = 1800,
        reconnect_min_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.keepalive_interval_seconds = int(keepalive_interval_seconds)
        self.reconnect_min_seconds = float(reconnect_min_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)

        self._running = False
        self._listen_key: Optional[str] = None
        self._callback: Optional[OnUserData] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    def _listen_key_request(self, method: str, listen_key: Optional[str] = None) -> Dict[str, object]:
        url = f"{self.base_url}/fapi/v1/listenKey"
        data = None
        if listen_key:
            payload = f"listenKey={listen_key}".encode("utf-8")
            data = payload

        request = urllib.request.Request(url=url, data=data, method=method)
        request.add_header("X-MBX-APIKEY", self.api_key)
        if data is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")

        with safe_urlopen(request, self.request_timeout_seconds) as response:
            body = response.read().decode("utf-8")
        if len(body.strip()) == 0:
            return {}
        return json.loads(body)

    async def _create_listen_key(self) -> str:
        payload = await asyncio.to_thread(self._listen_key_request, "POST")
        listen_key = str(payload.get("listenKey", ""))
        if len(listen_key) == 0:
            raise RuntimeError("Failed to create Binance listenKey")
        return listen_key

    async def _keepalive(self, listen_key: str) -> None:
        await asyncio.to_thread(self._listen_key_request, "PUT", listen_key)

    async def _delete_listen_key(self, listen_key: str) -> None:
        try:
            await asyncio.to_thread(self._listen_key_request, "DELETE", listen_key)
        except Exception:
            logger.warning("Failed to delete listenKey", exc_info=True)

    async def _emit(self, payload: Dict[str, object]) -> None:
        if self._callback is None:
            return
        result = self._callback(payload)
        if inspect.isawaitable(result):
            await result

    async def _ws_loop(self) -> None:
        if self._listen_key is None:
            return

        backoff = self.reconnect_min_seconds
        while self._running:
            ws_url = f"{self.ws_base_url}/{self._listen_key}"
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_queue=2048,
                ) as ws:
                    backoff = self.reconnect_min_seconds
                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        await self._emit(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "User-data WS reconnect after error: %s (sleep %.1fs)",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)

    async def _keepalive_loop(self) -> None:
        while self._running:
            await asyncio.sleep(max(60, self.keepalive_interval_seconds // 2))
            if not self._running or self._listen_key is None:
                continue
            try:
                await self._keepalive(self._listen_key)
            except Exception:
                logger.warning("User-data keepalive failed", exc_info=True)

    async def start(self, callback: OnUserData) -> None:
        if self._running:
            return

        self._callback = callback
        self._running = True
        self._listen_key = await self._create_listen_key()
        self._ws_task = asyncio.create_task(self._ws_loop(), name="binance-user-stream-ws")
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="binance-user-stream-keepalive",
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        for task in (self._ws_task, self._keepalive_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *[task for task in (self._ws_task, self._keepalive_task) if task is not None],
            return_exceptions=True,
        )
        self._ws_task = None
        self._keepalive_task = None

        if self._listen_key is not None:
            await self._delete_listen_key(self._listen_key)
        self._listen_key = None
        self._callback = None
