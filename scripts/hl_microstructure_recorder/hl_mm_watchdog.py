#!/usr/bin/env python3
"""Watchdog for hl-mm-recorder (HL market-making research). Systemd-timer oneshot.

Independent of the recorder process; checks that fresh data actually lands:
  1. HEALTH    — ./data/health.json exists and is < WD_HEALTH_AGE_S old.
  2. CHANNELS  — no channel's last-msg ts older than WD_CHANNEL_STALE_S
                 (trades on quiet long-tail coins are exempt via WD_TRADES_OK).
  3. FILES     — newest *.jsonl.gz mtime < WD_MAX_FILE_AGE_S.
  4. SERVICE   — systemctl is-active hl-mm-recorder.
  5. DISK      — free space >= WD_MIN_FREE_GB.

Telegram alert on FAIL (cooldown), single RECOVERED message on recovery.
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (via EnvironmentFile), overrides below.
Exit 0 pass / 1 fail.
"""
import glob, json, os, shutil, subprocess, sys, time, urllib.parse, urllib.request

DATA = os.environ.get("HL_MM_DATA", "./data")
UNIT = os.environ.get("HL_MM_UNIT", "hl-mm-recorder")
HEALTH_AGE_S = int(os.environ.get("WD_HEALTH_AGE_S", "180"))
CHANNEL_STALE_S = int(os.environ.get("WD_CHANNEL_STALE_S", "600"))
MAX_FILE_AGE_S = int(os.environ.get("WD_MAX_FILE_AGE_S", "300"))
MIN_FREE_GB = float(os.environ.get("WD_MIN_FREE_GB", "10"))
COOLDOWN_S = int(os.environ.get("WD_ALERT_COOLDOWN_S", "1800"))
STATE = os.environ.get("WD_STATE_FILE", "./data/watchdog_state.json")
# quiet long-tail markets legitimately print no trades/bbo-changes for hours;
# l2Book pushes ~7-8s snapshots regardless of activity → it is the strict gate
QUIET_OK_S = int(os.environ.get("WD_TRADES_QUIET_OK_S", "21600"))

def tg(msg):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("no telegram creds; msg:", msg); return
    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{tok}/sendMessage?" +
            urllib.parse.urlencode({"chat_id": chat, "text": msg}), timeout=15).read()
    except Exception as e:
        print("telegram send failed:", e)

fails = []
now = time.time()

# 1+2 health + channels
hp = os.path.join(DATA, "health.json")
try:
    h = json.load(open(hp))
    if now - h["ts_ms"]/1000 > HEALTH_AGE_S:
        fails.append(f"health.json stale ({int(now - h['ts_ms']/1000)}s)")
    else:
        chans = h.get("channels") or {}
        # l2Book is the strict liveness gate: HL pushes ~7-8s snapshots for every
        # subscribed coin regardless of activity, so a fresh l2Book proves the
        # subscription is alive. bbo/trades legitimately go silent for hours on
        # ultra-illiquid long-tail coins (e.g. @-index spot pairs with <10 quote
        # changes/day) — that is market quiet, not a fault. So bbo/trades are only
        # flagged when their coin's l2Book is ALSO stale (a genuine coin dropout);
        # otherwise fresh l2Book vetoes the false positive.
        def l2_fresh(coin):
            ts = chans.get(f"l2Book:{coin}")
            return ts is not None and (now - ts/1000) <= CHANNEL_STALE_S
        stale = []
        for ch, ts in chans.items():
            if ts is None:
                continue  # not yet seen since restart; freshness gate covers real loss
            age = now - ts/1000
            if ch.startswith("l2Book:"):
                limit = CHANNEL_STALE_S
            else:
                # bbo:COIN / trades:COIN — veto if that coin's l2Book is still flowing
                coin = ch.split(":", 1)[1]
                if l2_fresh(coin):
                    continue
                limit = QUIET_OK_S
            if age > limit:
                stale.append(f"{ch}:{int(age)}s")
        if stale:
            fails.append("stale channels: " + ", ".join(stale[:8]))
except Exception as e:
    fails.append(f"health.json unreadable: {e}")

# 3 newest file
files = glob.glob(f"{DATA}/*/dt=*/coin=*/*.jsonl.gz")
if not files:
    fails.append("no data files")
else:
    newest = max(os.path.getmtime(f) for f in files)
    if now - newest > MAX_FILE_AGE_S:
        fails.append(f"newest file {int(now-newest)}s old")

# 4 service
try:
    r = subprocess.run(["systemctl", "is-active", UNIT], capture_output=True, text=True)
    if r.stdout.strip() != "active":
        fails.append(f"unit {UNIT} = {r.stdout.strip()}")
except Exception:
    pass

# 5 disk
free_gb = shutil.disk_usage(DATA).free / 2**30
if free_gb < MIN_FREE_GB:
    fails.append(f"disk free {free_gb:.1f}G < {MIN_FREE_GB}G")

# alert with cooldown / recovery
st = {}
try: st = json.load(open(STATE))
except Exception: pass
if fails:
    print("FAIL:", "; ".join(fails))
    if now - st.get("last_alert", 0) > COOLDOWN_S:
        tg("🔴 hl-mm-recorder WATCHDOG:\n" + "\n".join(fails))
        st["last_alert"] = now
    st["failing"] = True
    json.dump(st, open(STATE, "w"))
    sys.exit(1)
else:
    if st.get("failing"):
        tg("🟢 hl-mm-recorder RECOVERED")
    json.dump({"failing": False, "last_alert": st.get("last_alert", 0)}, open(STATE, "w"))
    print("OK")
