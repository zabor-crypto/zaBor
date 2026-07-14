#!/usr/bin/env python3
"""
hl-mm-recorder universe rotation manager (HL market-making research).

Automatically prunes dead coins and (optionally) adds active ones to the
recorder's subscription universe, then reloads the service. Runs from a
systemd timer (daily). Independent of the recorder process.

WHY THIS EXISTS
---------------
The universe was hand-picked by HL dayNtlVlm, which is a TRAP for HL spot:
observed 2026-07-04, @109 WOW/USDC reported $68M/day volume yet ticked BBO
10x in 18h (dead), while @107 HYPE/USDC reported $27/day yet is the most
active spot book we record. HL spot volume is anti-correlated with real quote
activity (wash/stale). So:

  * REMOVAL (spot + perp): driven ONLY by observed BBO/trade activity from the
    recorder's own data/health.json. This is the authoritative, trustworthy signal.
  * PERP ADD: candidates ranked by real perp dayNtlVlm, but each is
    PROBE-CONFIRMED (short live WS subscription must show real ticks) before it
    is committed. Volume is only a prior; the probe is the gate.
  * SPOT ADD: never by volume. Off by default (enable_spot_probe_add) and, when
    on, uses the same live-probe gate against an operator-provided candidate pool.
  * pinned_core coins never rotate out -> keeps the research/G1 dataset stable.

SAFETY
------
  * Never writes an empty universe or one missing the pinned core.
  * Removals use hysteresis (N consecutive dead evals) except hard-dead (~0
    activity over a full uptime window), which are removed immediately.
  * Removed coins enter a cooldown (no re-add for cooldown_days) to stop flapping.
  * Restart is rate-limited (min_hours_between_restarts) and only happens if the
    universe actually changed. Atomic drop-in write + validate + daemon-reload.
  * --dry-run (default) prints the plan and changes nothing. --apply commits.

Files (all under BASE=/opt/hl_mm):
  universe_config.json     policy (operator-edited)
  data/health.json         recorder activity (input)
  universe_state.json      hysteresis + cooldown + last-restart state
  universe_audit.log       append-only decision log
  /etc/systemd/system/hl-mm-recorder.service.d/universe.conf   (written)
"""
import argparse, asyncio, json, os, re, subprocess, sys, time
from datetime import datetime, timezone

BASE = os.environ.get("HL_MM_BASE", "/opt/hl_mm")
CONF = os.path.join(BASE, "universe_config.json")
HEALTH = os.path.join(BASE, "data", "health.json")
STATE = os.path.join(BASE, "universe_state.json")
AUDIT = os.path.join(BASE, "universe_audit.log")
DROPIN = os.environ.get(
    "HL_MM_DROPIN",
    "/etc/systemd/system/hl-mm-recorder.service.d/universe.conf")
UNIT = os.environ.get("HL_MM_UNIT", "hl-mm-recorder")
WS_URL = os.environ.get("HL_WS_URL", "wss://api.hyperliquid.xyz/ws")
INFO_URL = os.environ.get("HL_INFO_URL", "https://api.hyperliquid.xyz/info")


def now_utc():
    return datetime.now(timezone.utc)


def log_audit(msg):
    line = f"{now_utc().isoformat()} {msg}"
    print(line)
    try:
        with open(AUDIT, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"audit write failed: {e}", file=sys.stderr)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


# ---------- inputs ----------

def read_current_universe():
    """Parse HL_MM_COINS=... from the systemd drop-in. Empty list if absent."""
    try:
        txt = open(DROPIN).read()
    except Exception:
        return []
    m = re.search(r"HL_MM_COINS=([^\n]*)", txt)
    if not m:
        return []
    return [c.strip() for c in m.group(1).split(",") if c.strip()]


def coin_rates(health):
    """Per-coin observed activity: {coin: {'bbo_hr':x,'trade_hr':y}} and uptime_h.
    Uses uptime from the recorder; falls back to None uptime if not present."""
    uptime_h = None
    if health.get("uptime_s"):
        uptime_h = float(health["uptime_s"]) / 3600.0
    elif health.get("start_ts_ms") and health.get("ts_ms"):
        uptime_h = (health["ts_ms"] - health["start_ts_ms"]) / 3.6e6
    counts = health.get("counts") or {}
    coins = health.get("coins") or []
    if not coins:  # derive from channel keys
        coins = sorted({k.split(":", 1)[1] for k in counts})
    rates = {}
    for c in coins:
        if uptime_h and uptime_h > 0:
            bbo = counts.get(f"bbo:{c}", 0) / uptime_h
            tr = counts.get(f"trades:{c}", 0) / uptime_h
        else:
            bbo = tr = None
        rates[c] = {"bbo_hr": bbo, "trade_hr": tr}
    return rates, uptime_h


def hl_info(payload):
    import urllib.request
    req = urllib.request.Request(
        INFO_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def perp_candidates(pool_n):
    """Perp coin names ranked by real dayNtlVlm, descending."""
    try:
        d = hl_info({"type": "metaAndAssetCtxs"})
        meta, ctxs = d[0]["universe"], d[1]
        rows = [(u["name"], float(c.get("dayNtlVlm", 0) or 0))
                for u, c in zip(meta, ctxs)]
        rows.sort(key=lambda x: -x[1])
        return [n for n, _ in rows[:pool_n]]
    except Exception as e:
        log_audit(f"perp_candidates FAILED: {e}")
        return []


# ---------- live probe (defeats the volume trap) ----------

async def _probe(coins, seconds):
    import websockets
    counts = {c: {"bbo": 0, "trades": 0} for c in coins}
    try:
        async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=20,
                                       max_size=2 ** 23) as ws:
            for c in coins:
                for stream in ("bbo", "trades"):
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": stream, "coin": c}}))
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
                except (asyncio.TimeoutError, Exception):
                    break
                m = json.loads(raw)
                ch = m.get("channel")
                data = m.get("data")
                if ch == "bbo" and isinstance(data, dict):
                    c = data.get("coin")
                    if c in counts:
                        counts[c]["bbo"] += 1
                elif ch == "trades" and isinstance(data, list) and data:
                    c = data[0].get("coin")
                    if c in counts:
                        counts[c]["trades"] += 1
    except Exception as e:
        log_audit(f"probe connection FAILED: {e}")
    return counts


def probe(coins, seconds):
    if not coins:
        return {}
    return asyncio.run(_probe(coins, seconds))


# ---------- decision ----------

def decide(cfg, current, rates, uptime_h, state):
    core = list(cfg["pinned_core"])
    act = cfg["activity"]
    keep_or = act.get("keep_rule", "OR").upper() == "OR"
    dead_ctr = dict(state.get("dead_counter", {}))
    cooldown = dict(state.get("cooldown_until", {}))  # coin -> epoch
    now = time.time()

    removed, reasons = [], []
    # never judge if we don't have enough uptime yet
    judged = uptime_h is not None and uptime_h >= act["min_uptime_h"]

    keep = []
    for c in current:
        if c in core:
            keep.append(c)
            continue
        r = rates.get(c, {"bbo_hr": None, "trade_hr": None})
        bbo, tr = r["bbo_hr"], r["trade_hr"]
        if not judged or bbo is None:
            keep.append(c)  # insufficient data -> keep (safe)
            continue
        alive = (bbo >= act["bbo_floor_per_hr"]) or (tr >= act["trade_floor_per_hr"]) \
            if keep_or else \
            (bbo >= act["bbo_floor_per_hr"]) and (tr >= act["trade_floor_per_hr"])
        hard_dead = (bbo <= act["hard_dead_bbo_per_hr"]) and (tr <= act["hard_dead_trade_per_hr"])
        if alive:
            dead_ctr.pop(c, None)
            keep.append(c)
        elif hard_dead:
            removed.append(c)
            reasons.append(f"REMOVE {c} hard-dead bbo={bbo:.1f}/hr trades={tr:.1f}/hr (uptime {uptime_h:.1f}h)")
            cooldown[c] = now + cfg["cooldown_days"] * 86400
            dead_ctr.pop(c, None)
        else:
            n = dead_ctr.get(c, 0) + 1
            dead_ctr[c] = n
            if n >= act["dead_evals_to_remove"]:
                removed.append(c)
                reasons.append(f"REMOVE {c} below-floor {n}x bbo={bbo:.1f}/hr trades={tr:.1f}/hr")
                cooldown[c] = now + cfg["cooldown_days"] * 86400
                dead_ctr.pop(c, None)
            else:
                keep.append(c)
                reasons.append(f"WATCH  {c} below-floor {n}/{act['dead_evals_to_remove']} bbo={bbo:.1f}/hr trades={tr:.1f}/hr")

    # ensure pinned core present
    for c in core:
        if c not in keep:
            keep.append(c)
            reasons.append(f"RESTORE core {c} (was missing)")

    # expire cooldowns
    cooldown = {c: t for c, t in cooldown.items() if t > now}

    # ---- adds (probe-confirmed) ----
    added = []
    addcfg = cfg["add"]
    slots = cfg["target_size"] - len(keep)
    if slots > 0 and addcfg.get("enable_perp_add"):
        pool = [c for c in perp_candidates(addcfg["perp_candidate_pool"])
                if c not in keep and c not in cooldown]
        if pool:
            reasons.append(f"PROBE perp candidates ({addcfg['probe_seconds']}s): {pool[:12]}")
            pc = probe(pool, addcfg["probe_seconds"])
            scored = []
            for c in pool:
                cc = pc.get(c, {"bbo": 0, "trades": 0})
                if cc["bbo"] >= addcfg["probe_bbo_min"] and cc["trades"] >= addcfg["probe_trade_min"]:
                    scored.append((c, cc["bbo"], cc["trades"]))
            scored.sort(key=lambda x: -x[1])
            for c, b, t in scored[:slots]:
                added.append(c)
                reasons.append(f"ADD    {c} probe bbo={b} trades={t} in {addcfg['probe_seconds']}s")
            for c in pool:
                if c in added:
                    continue
                cc = pc.get(c, {"bbo": 0, "trades": 0})
                passed = cc["bbo"] >= addcfg["probe_bbo_min"] and cc["trades"] >= addcfg["probe_trade_min"]
                why = "passed probe, below top slots" if passed else "below probe min"
                reasons.append(f"skip   {c} probe bbo={cc['bbo']} trades={cc['trades']} ({why})")

    new_universe = keep + added
    # enforce max_size (drop lowest-activity non-core if over)
    if len(new_universe) > cfg["max_size"]:
        overflow = len(new_universe) - cfg["max_size"]
        droppable = [c for c in new_universe if c not in core and c not in added]
        droppable.sort(key=lambda c: (rates.get(c, {}).get("bbo_hr") or 0))
        for c in droppable[:overflow]:
            new_universe.remove(c)
            reasons.append(f"TRIM   {c} (over max_size {cfg['max_size']})")

    new_state = {
        "dead_counter": dead_ctr,
        "cooldown_until": cooldown,
        "last_restart": state.get("last_restart", 0),
    }
    return new_universe, removed, added, reasons, new_state


# ---------- apply ----------

def write_universe(coins):
    body = "[Service]\nEnvironment=HL_MM_COINS=" + ",".join(coins) + "\n"
    tmp = DROPIN + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
    os.replace(tmp, DROPIN)


def reload_restart():
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "restart", UNIT], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="commit changes + reload service")
    ap.add_argument("--dry-run", action="store_true", help="print plan only (default)")
    ap.add_argument("--force-restart", action="store_true", help="restart even if rate-limit not elapsed")
    args = ap.parse_args()
    apply = args.apply and not args.dry_run

    cfg = load_json(CONF, None)
    if not cfg:
        log_audit("FATAL: config unreadable")
        sys.exit(2)
    health = load_json(HEALTH, {})
    state = load_json(STATE, {})
    current = read_current_universe()
    if not current:
        log_audit("FATAL: current universe empty/unreadable; refusing to act")
        sys.exit(2)

    rates, uptime_h = coin_rates(health)
    new_universe, removed, added, reasons, new_state = decide(cfg, current, rates, uptime_h, state)

    log_audit(f"eval uptime={uptime_h}h current={len(current)} -> new={len(new_universe)} "
              f"removed={removed} added={added}")
    for r in reasons:
        log_audit("  " + r)

    changed = set(new_universe) != set(current)
    # ---- safety gates ----
    core = set(cfg["pinned_core"])
    if not new_universe or not core.issubset(set(new_universe)):
        log_audit("ABORT: proposed universe empty or missing pinned core; no change")
        sys.exit(1)

    if not changed:
        log_audit("no change")
        # still persist counters/cooldowns
        if apply:
            json.dump(new_state, open(STATE, "w"), indent=2)
        return

    if not apply:
        log_audit(f"DRY-RUN: would set universe ({len(new_universe)}): {','.join(new_universe)}")
        return

    # rate-limit restart
    rl = cfg["restart"]["min_hours_between_restarts"] * 3600
    since = time.time() - state.get("last_restart", 0)
    if since < rl and not args.force_restart:
        log_audit(f"CHANGE DEFERRED: last restart {since/3600:.1f}h ago < {rl/3600:.1f}h min; "
                  f"universe.conf NOT written this run")
        json.dump(new_state, open(STATE, "w"), indent=2)  # keep counters advancing
        return

    write_universe(new_universe)
    reload_restart()
    new_state["last_restart"] = time.time()
    json.dump(new_state, open(STATE, "w"), indent=2)
    log_audit(f"APPLIED universe ({len(new_universe)}): {','.join(new_universe)}")


if __name__ == "__main__":
    main()
