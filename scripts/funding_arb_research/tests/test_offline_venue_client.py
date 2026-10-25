"""Client interface contracts with fake sessions and fictional public frames.

No socket is opened. These tests do not exercise aiohttp server security fixes.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import aiohttp

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.funding_arb_research.src.recorder.venue_clients.base import (  # noqa: E402
    VenueClient,
    VenueClientConfig,
)


class FakeSocket:
    def __init__(self, frames):
        self.frames = frames
        self.sent = []
        self.text_sent = []
        self.closed = False
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.exited = True

    async def send_json(self, payload):
        self.sent.append(payload)

    async def send_str(self, payload):
        self.text_sent.append(payload)

    def __aiter__(self):
        async def frames():
            for frame in self.frames:
                await asyncio.sleep(0)
                yield frame
        return frames()


class FictionalClient(VenueClient):
    def __init__(self):
        super().__init__(VenueClientConfig("fictional", ["SYNTH"], "wss://fixture.invalid"), object())
        self.received = []

    def venue_symbol(self, coin):
        return coin

    def subscribe_payloads(self):
        return [{"op": "subscribe", "channel": "fictional"}]

    def handle_message(self, message, received_at):
        if message.get("reject"):
            raise ValueError("fictional parser rejection")
        self.received.append((message, received_at))


class ClientInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_timeout_subscription_and_real_enum_dispatch(self):
        client = FictionalClient()
        socket = FakeSocket([
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data='{"price": 7}'),
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="pong"),
            SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"ignored"),
        ])
        captured = {}

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                captured["exited"] = True

            def ws_connect(self, url, **kwargs):
                captured["url"] = url
                captured["ws_args"] = kwargs
                return socket

        def session_factory(**kwargs):
            captured["timeout"] = kwargs["timeout"]
            return FakeSession()

        with patch.object(aiohttp, "ClientSession", side_effect=session_factory), patch("time.time", return_value=123.5):
            await client._run_once()
        self.assertIsInstance(captured["timeout"], aiohttp.ClientTimeout)
        self.assertIsNone(captured["timeout"].total)
        self.assertEqual(captured["timeout"].sock_connect, 10)
        self.assertEqual(captured["url"], "wss://fixture.invalid")
        self.assertEqual(captured["ws_args"], {"heartbeat": 20.0, "autoping": True, "receive_timeout": 60.0})
        self.assertEqual(socket.sent, [{"op": "subscribe", "channel": "fictional"}])
        self.assertEqual(client.received, [({"price": 7}, 123.5)])
        self.assertEqual(client.msgs_received, 2)
        self.assertTrue(socket.exited and captured["exited"])

    async def test_bad_json_and_parser_failure_do_not_drop_following_frame(self):
        client = FictionalClient()
        socket = FakeSocket([
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="not JSON"),
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data='{"reject": true}'),
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data='{"price": 11}'),
        ])
        with patch("time.time", return_value=321.0):
            await client._read_loop(socket)
        self.assertEqual(client.msgs_received, 3)
        self.assertEqual(client.received, [({"price": 11}, 321.0)])

    async def test_closed_and_error_frames_surface_for_reconnection(self):
        for kind in [aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR]:
            with self.subTest(kind=kind):
                client = FictionalClient()
                with self.assertRaisesRegex(RuntimeError, "ws closed"):
                    await client._read_loop(FakeSocket([SimpleNamespace(type=kind, data="fictional close")]))

    async def test_ping_task_is_cancelled_when_read_loop_fails(self):
        client = FictionalClient()
        client.text_ping_payload = lambda: "ping"
        socket = FakeSocket([SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data="fixture")])

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def ws_connect(self, *args, **kwargs):
                return socket

        tasks = []
        create_task = asyncio.create_task

        class TrackedTask:
            def __init__(self, task):
                self.task = task
                self.cancel = Mock(wraps=task.cancel)

            def __await__(self):
                return self.task.__await__()

            def done(self):
                return self.task.done()

        def tracked_task(*args, **kwargs):
            task = TrackedTask(create_task(*args, **kwargs))
            tasks.append(task)
            return task

        with patch.object(aiohttp, "ClientSession", return_value=FakeSession()), patch.object(asyncio, "create_task", side_effect=tracked_task):
            with self.assertRaisesRegex(RuntimeError, "ws closed"):
                await client._run_once()
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].done())
        tasks[0].cancel.assert_called_once_with()
        # The read failure arrives before the first scheduled ping.
        self.assertEqual(socket.text_sent, [])
        self.assertTrue(socket.exited)


if __name__ == "__main__":
    unittest.main()
