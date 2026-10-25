# funding-arb VPS deployment

Lives at `/opt/funding_arb` on the VPS. Runs as the `funding-arb` system user under systemd.
Logs go to `/var/log/funding-arb/recorder.log` and `journalctl -u funding-arb-recorder`.

## First-time install

From your dev machine, review a command plan first. The preview does not call SSH or
rsync and does not contact the target:

```bash
VPS=user@your.vps.host ./deploy/sync.sh --install
```

Recheck the target, source tree and owner's authorization before executing it:

```bash
VPS=user@your.vps.host ./deploy/sync.sh --apply --install
```

This copies code, then runs `install_vps.sh`: it installs packages, creates the service
user and venv, sets ownership, registers the systemd unit and starts the recorder.
The SSH user needs write access to the destination and the installer needs sudo.
These are operational effects requiring authorization for that target; the public
example does not authorize deployment. After an authorized install:

```bash
ssh $VPS 'sudo systemctl status funding-arb-recorder'
ssh $VPS 'tail -f /var/log/funding-arb/recorder.log'
```

## Subsequent updates

```bash
VPS=user@your.vps.host ./deploy/sync.sh --restart          # preview only
VPS=user@your.vps.host ./deploy/sync.sh --apply --restart  # execute after approval
```

Applying this plan copies changed code and restarts the unit. The recorder's shutdown
handler attempts to flush its buffers; deployment success does not prove that every
in-flight row was retained. Verify recorder health and the resulting files afterward.

With no action flag, the default is also a preview. `--apply` alone copies code without
installing packages or restarting services. Unknown or conflicting options are rejected
before any deployment command is run.

## Sync boundary

The transfer excludes local `.env` / `.env.*` / `*.env` files, runtime `data/`,
`outputs/`, `reports/`, `logs/`, local environments, Git metadata, agent configuration
and caches. The public
`.env.example` template may be copied. Inspect any other local files before applying:
filename exclusions are not a complete secret scan.

There is no `rsync --delete`: extra remote files and excluded state are preserved.
This also means retired code can remain remotely; inventory and authorize any cleanup
as a separate operation. The script does not change existing ownership during ordinary
sync/restart, suppress preparation errors, or silently acquire permissions.

## Disk

The recorder quota is set by `--quota-gb` in the systemd unit's ExecStart.
Adjust by editing `deploy/funding-arb-recorder.service` and re-syncing with
`--restart`. If you want a different coin universe, also edit `--coins ...`.

## Where data lands on the VPS

```
/opt/funding_arb/data/recorder/<channel>/dt=YYYY-MM-DD/venue=<v>/coin=<c>/*.parquet
```

To pull a day's worth back to your dev box:

```bash
rsync -avz $VPS:/opt/funding_arb/data/recorder/ ./data/recorder_vps/
```

## What's installed

- `funding-arb-recorder.service` — long-lived recorder (Bitget + Binance USDT-M)
- (Future) periodic-collector cron jobs for funding history / MMR archive — added in Step 2 follow-up

## Retirement

Inventory service consumers, code, logs and retained data before preparing a retirement
plan. Confirm recoverable custody and obtain owner approval for the exact stop/delete
scope before executing a separate cleanup. This guide provides no blanket teardown
command for an unknown VPS.
