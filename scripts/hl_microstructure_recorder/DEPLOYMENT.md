# hl-mm-recorder deployment
**Deployed 2026-07-03 13:25 UTC. Universe hardened 2026-07-04.** Isolated service, no trading, no shared state with other services.

- Unit: `hl-mm-recorder.service` (Restart=always, MemoryMax=1G, Nice=10)
- Script: `/opt/hl_mm/hl_l2_recorder.py` (source of truth: this repo, `recorder/hl_l2_recorder.py`)
- Output: `./data/{l2Book,bbo,trades}/dt=YYYY-MM-DD/coin=<COIN>/<HH>.jsonl.gz`
- Universe (20, v3): BTC ETH SOL HYPE XRP DOGE SUI ZEC AVAX LINK PURR/USDC @107 (=orig 12 `pinned_core`) + @198 @301 @1 + PUMP FET XPL SPX DYDX. Set via systemd drop-in `hl-mm-recorder.service.d/universe.conf` (`HL_MM_COINS=`).
- Health: `./data/health.json` (`ts_ms`, `start_ts_ms`, `uptime_s`, `coins`, per-channel `channels`+`counts`, 30s cadence)
- Retention: cron `hl_mm_retention` deletes *.jsonl.gz older than 21 days (04:17 daily)

## Companion units
- `hl-mm-watchdog.timer` (5min): capture watchdog, Telegram via `/etc/hl-mm/watchdog.env`. Strict on l2Book; bbo/trades vetoed when the coin's l2Book is fresh (07-04 false-alarm fix). Source `recorder/hl_mm_watchdog.py`.
- `hl-mm-universe.timer` (daily 08:20 UTC): auto-rotation. `recorder/hl_universe_manager.py` + `universe_config.json`; state `universe_state.json`, audit `universe_audit.log`. Units in `recorder/systemd/`. **Adds FROZEN by default** (`enable_perp_add=false`, `target_size=20`); removal of dead coins stays active. Manual: `python3 /opt/hl_mm/hl_universe_manager.py --dry-run` (safe preview) / `--apply`.

## Notes
- l2Book cadence ~7-8s per coin (interval snapshots, not per-delta); bbo event-driven (~5-10Hz majors) — sim drives off bbo+trades, l2Book for depth. HL WS session expires ~every 3h → recorder reconnects in ~1s (normal, logged).
- Universe changes restart the recorder (restart loses nothing; files append). Rotation restarts ≤1/day only when the set changes.
- Backups of every 07-04 change on the VPS: `*.bak_20260704`.
