"""Exercise existing capture code with fictional input and no websocket dependency.

The websocket module is replaced at import and during each capture. No test calls
the recorder CLI, its operations scripts, or a network client. All writes use a
TemporaryDirectory. These are software contracts, not market-quality/economic tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((Path(__file__).parent / "fixtures/synthetic_events.json").read_text())


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import-time stub makes even a missing test patch fail locally, not use a network.
with patch.dict(sys.modules, {"websockets": types.ModuleType("websockets")}):
    recorder = load_module("offline_hl_recorder", "hl_l2_recorder.py")
toxicity = load_module("offline_hl_toxicity", "toxicity.py")


class FixedClock:
    current = datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz):
        return cls.current.astimezone(tz)


class WriterAndReaderTests(unittest.TestCase):
    def test_hour_rotation_and_restart_append_preserve_records(self):
        with tempfile.TemporaryDirectory() as root, patch.object(recorder, "datetime", FixedClock):
            FixedClock.current = datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc)
            writer = recorder.HourlyGzWriter(root)
            self.addCleanup(writer.close)
            writer.write("bbo", "SYNTH/USD", {"sequence": 1})
            FixedClock.current = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
            writer.write("bbo", "SYNTH/USD", {"sequence": 2})
            writer.close()
            restarted = recorder.HourlyGzWriter(root)
            self.addCleanup(restarted.close)
            restarted.write("bbo", "SYNTH/USD", {"sequence": 3})
            restarted.close()

            paths = sorted(Path(root).rglob("*.jsonl.gz"))
            self.assertEqual([p.name for p in paths], ["00.jsonl.gz", "01.jsonl.gz"])
            self.assertTrue(all(p.parent.name == "coin=SYNTH_USD" for p in paths))
            self.assertEqual(list(toxicity.read_gz_lines(paths)), [
                {"sequence": 1}, {"sequence": 2}, {"sequence": 3},
            ])
            self.assertEqual(writer.files, {})
            self.assertEqual(restarted.files, {})

    def test_flush_exposes_complete_json_record_before_close(self):
        with tempfile.TemporaryDirectory() as root:
            writer = recorder.HourlyGzWriter(root)
            try:
                writer.write("trades", "@123", {"sequence": 1})
                writer.flush()
                paths = list(Path(root).rglob("*.jsonl.gz"))
                self.assertEqual(paths[0].parent.name, "coin=at123")
                self.assertEqual(list(toxicity.read_gz_lines(paths)), [{"sequence": 1}])
            finally:
                writer.close()

    def test_reader_keeps_decoded_lines_from_truncated_gzip_and_continues(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            truncated = root / "01.jsonl.gz"
            raw = b'{"sequence":1}\nnot-json\n{"sequence":2}\n'
            truncated.write_bytes(gzip.compress(raw, mtime=0)[:-8])
            invalid = root / "02.jsonl.gz"
            invalid.write_bytes(b"not a gzip file")
            intact = root / "03.jsonl.gz"
            intact.write_bytes(gzip.compress(b'{"sequence":3}\n', mtime=0))
            self.assertEqual(list(toxicity.read_gz_lines([
                intact, invalid, root / "missing.jsonl.gz", truncated,
            ])), [{"sequence": 1}, {"sequence": 2}, {"sequence": 3}])

    def test_asof_lookup_uses_no_future_quote_but_has_no_age_limit(self):
        at = toxicity.make_lookup([100, 200])
        self.assertIsNone(at(99))
        self.assertEqual(at(100), 0)
        self.assertEqual(at(199), 0)
        self.assertEqual(at(200), 1)
        # A characterization of an existing limitation, not a gap-quality gate.
        self.assertEqual(at(1_000_000), 1)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def capture(self, plans):
        """A plan is a connection error or finite list of receive events."""
        stop = asyncio.Event()
        health_requested = asyncio.Event()
        clock_ms = FIXTURE["start_ms"]
        backoffs, subscriptions, attempts = [], [], []
        real_sleep = asyncio.sleep

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            writer = recorder.HourlyGzWriter(root)
            self.addCleanup(writer.close)

            class Connection:
                def __init__(self, plan):
                    self.plan = plan

                async def __aenter__(self):
                    if isinstance(self.plan, Exception):
                        raise self.plan
                    self.events = iter(self.plan)
                    return self

                async def __aexit__(self, *args):
                    return False

                async def send(self, raw):
                    subscriptions.append(json.loads(raw))

                async def recv(self):
                    nonlocal clock_ms
                    event = next(self.events, None)
                    if event is not None:
                        clock_ms = event["recv_ms"]
                        return event.get("raw", json.dumps(event.get("frame")))
                    health_requested.set()
                    while not (root / "health.json").exists():
                        await real_sleep(0)
                    stop.set()
                    raise asyncio.CancelledError

            def connect(*args, **kwargs):
                attempts.append((args, kwargs))
                if not plans:
                    # Stop an unexpected retry without hanging or opening a socket.
                    stop.set()
                    raise AssertionError("unexpected connection attempt")
                return Connection(plans.pop(0))

            async def sleep(delay):
                if delay == recorder.HEALTH_EVERY_S:
                    await health_requested.wait()
                    health_requested.clear()
                else:
                    backoffs.append(delay)
                    await real_sleep(0)

            async_api = types.SimpleNamespace(
                create_task=asyncio.create_task, wait_for=asyncio.wait_for,
                CancelledError=asyncio.CancelledError, sleep=sleep,
            )
            with (
                patch.object(recorder, "OUT", str(root)),
                patch.object(recorder, "COINS", [FIXTURE["coin"]]),
                patch.object(recorder, "WS_URL", "wss://offline.invalid"),
                patch.object(recorder, "websockets", types.SimpleNamespace(connect=connect)),
                patch.object(recorder, "asyncio", async_api),
                patch.object(recorder, "time", types.SimpleNamespace(time=lambda: clock_ms / 1000)),
                patch.object(recorder, "HourlyGzWriter", return_value=writer),
                patch.object(writer, "flush", wraps=writer.flush) as flush,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                await asyncio.wait_for(recorder.run(stop), timeout=2)
                await real_sleep(0)  # deliver cancellation to the health task
                self.assertEqual(flush.call_count, 1)
            self.assertEqual(writer.files, {})
            records = {}
            for stream in recorder.STREAMS:
                records[stream] = list(toxicity.read_gz_lines(root.glob(f"{stream}/**/*.gz")))
            health = json.loads((root / "health.json").read_text())
        return records, health, backoffs, subscriptions, attempts

    async def test_control_frames_do_not_count_and_receive_gap_is_preserved(self):
        controls = [{"recv_ms": FIXTURE["start_ms"], "frame": frame} for frame in (
            {"channel": "subscriptionResponse"}, {"channel": "pong"},
            {"channel": "error", "data": "fictional error"},
            {"channel": "trades", "data": []}, {"channel": "bbo", "data": []},
            {"channel": "other", "data": {}},
        )]
        records, health, delays, subscriptions, attempts = await self.capture([
            controls + FIXTURE["events"],
        ])
        self.assertEqual(delays, [])
        self.assertEqual(len(attempts), 1)
        self.assertEqual({s["subscription"]["type"] for s in subscriptions}, set(recorder.STREAMS))
        self.assertTrue(all(s["subscription"]["coin"] == FIXTURE["coin"] for s in subscriptions))
        self.assertTrue(all(s["method"] == "subscribe" for s in subscriptions))
        self.assertEqual(health["counts"], {
            "bbo:SYNTH/USD": 2, "l2Book:SYNTH/USD": 1, "trades:SYNTH/USD": 1,
        })
        self.assertEqual(health["channels"], {
            "bbo:SYNTH/USD": 1704067321000,
            "l2Book:SYNTH/USD": 1704067321200,
            "trades:SYNTH/USD": 1704067321100,
        })
        self.assertEqual(health["start_ts_ms"], FIXTURE["start_ms"])
        self.assertAlmostEqual(health["uptime_s"], 121.2)
        self.assertEqual(health["ts_ms"], 1704067321200)
        for stream, indexes in {"bbo": [0, 1], "trades": [2], "l2Book": [3]}.items():
            self.assertEqual(records[stream], [
                {"recv_ts_ms": FIXTURE["events"][i]["recv_ms"],
                 "d": FIXTURE["events"][i]["frame"]["data"]}
                for i in indexes
            ])
        self.assertEqual(records["bbo"][1]["recv_ts_ms"] - records["bbo"][0]["recv_ts_ms"], 120900)

    async def test_reconnect_backoff_resets_and_resubscribes_after_bad_json(self):
        records, health, delays, subscriptions, attempts = await self.capture([
            OSError("fictional connection failure 1"),
            OSError("fictional connection failure 2"),
            [FIXTURE["events"][0], {"recv_ms": 1704067200200, "raw": "not-json"}],
            FIXTURE["events"][1:],
        ])
        self.assertEqual(delays, [1, 2, 1])
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(subscriptions), 6)
        self.assertEqual(subscriptions[:3], subscriptions[3:])
        self.assertEqual(health["counts"]["bbo:SYNTH/USD"], 2)
        self.assertEqual(len(records["bbo"]), 2)


if __name__ == "__main__":
    unittest.main()
