#!/usr/bin/env python3
"""
Hyperliquid L2 book + BBO + trades recorder (HL market-making research).

Subscribes per coin to: l2Book (20 levels), bbo, trades.
Output: gzip JSONL, hourly-rotated files partitioned as
  <out>/<stream>/dt=YYYY-MM-DD/coin=<COIN>/<HH>.jsonl.gz
Each record carries recv_ts_ms (local) alongside the exchange payload so
latency and gap analysis stay possible. Reconnects with backoff; per-channel
last-message watchdog state written to <out>/health.json every 30s.

An isolated, append-only recorder discipline:
no trading, bounded universe. Disk estimate ~2-4 GB/day at 12 coins.
"""
import asyncio, gzip, json, os, signal, sys, time
from datetime import datetime, timezone

import websockets

WS_URL = os.environ.get("HL_WS_URL", "wss://api.hyperliquid.xyz/ws")
OUT = os.environ.get("HL_MM_OUT", "./data")
COINS = os.environ.get(
    "HL_MM_COINS",
    "BTC,ETH,SOL,HYPE,XRP,DOGE,SUI,ZEC,AVAX,LINK,PURR/USDC,@107",
).split(",")
STREAMS = ("l2Book", "bbo", "trades")
HEALTH_EVERY_S = 30
FLUSH_EVERY_S = 5


class HourlyGzWriter:
    """One gz file per (stream, coin, hour); append-only, rotated on hour."""

    def __init__(self, root):
        self.root = root
        self.files = {}  # (stream, coin) -> (hour_key, fh)

    def _path(self, stream, coin, dt):
        safe = coin.replace("/", "_").replace("@", "at")
        d = os.path.join(self.root, stream, f"dt={dt:%Y-%m-%d}", f"coin={safe}")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{dt:%H}.jsonl.gz")

    def write(self, stream, coin, obj):
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y%m%d%H")
        cur = self.files.get((stream, coin))
        if cur is None or cur[0] != hour_key:
            if cur is not None:
                cur[1].close()
            fh = gzip.open(self._path(stream, coin, now), "at", compresslevel=5)
            self.files[(stream, coin)] = (hour_key, fh)
            cur = self.files[(stream, coin)]
        cur[1].write(json.dumps(obj, separators=(",", ":")) + "\n")

    def flush(self):
        for _, fh in self.files.values():
            fh.flush()

    def close(self):
        for _, fh in self.files.values():
            fh.close()
        self.files.clear()


async def run(stop_evt):
    writer = HourlyGzWriter(OUT)
    last_msg = {}  # (stream, coin) -> recv_ts_ms
    msg_count = {}
    start_ms = int(time.time() * 1000)  # for uptime-based activity rates (universe manager)

    async def health_loop():
        while not stop_evt.is_set():
            await asyncio.sleep(HEALTH_EVERY_S)
            writer.flush()
            now_ms = int(time.time() * 1000)
            snap = {
                "ts_ms": now_ms,
                "start_ts_ms": start_ms,
                "uptime_s": (now_ms - start_ms) / 1000.0,
                "coins": list(COINS),
                "channels": {f"{s}:{c}": last_msg.get((s, c)) for s in STREAMS for c in COINS},
                "counts": {f"{s}:{c}": msg_count.get((s, c), 0) for s in STREAMS for c in COINS},
            }
            tmp = os.path.join(OUT, "health.json.tmp")
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, os.path.join(OUT, "health.json"))

    health_task = asyncio.create_task(health_loop())
    backoff = 1
    while not stop_evt.is_set():
        try:
            async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=20, max_size=2**23) as ws:
                sys.stderr.write(f"{datetime.now(timezone.utc).isoformat()} connected\n")
                backoff = 1
                for coin in COINS:
                    for stream in STREAMS:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": stream, "coin": coin},
                        }))
                while not stop_evt.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    recv_ms = int(time.time() * 1000)
                    m = json.loads(raw)
                    ch = m.get("channel")
                    if ch == "subscriptionResponse" or ch == "pong":
                        continue
                    if ch == "error":
                        sys.stderr.write(f"{datetime.now(timezone.utc).isoformat()} ws error: {m}\n")
                        continue
                    data = m.get("data")
                    if ch == "trades" and isinstance(data, list) and data:
                        coin = data[0].get("coin", "?")
                        writer.write("trades", coin, {"recv_ts_ms": recv_ms, "d": data})
                    elif ch in ("l2Book", "bbo") and isinstance(data, dict):
                        coin = data.get("coin", "?")
                        writer.write(ch, coin, {"recv_ts_ms": recv_ms, "d": data})
                    else:
                        continue
                    last_msg[(ch, coin)] = recv_ms
                    msg_count[(ch, coin)] = msg_count.get((ch, coin), 0) + 1
        except asyncio.CancelledError:
            break
        except Exception as e:
            sys.stderr.write(f"{datetime.now(timezone.utc).isoformat()} reconnect after: {type(e).__name__}: {e}\n")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
    health_task.cancel()
    writer.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    stop_evt = asyncio.Event()
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)
    sys.stderr.write(f"start out={OUT} coins={COINS}\n")
    loop.run_until_complete(run(stop_evt))


if __name__ == "__main__":
    main()
