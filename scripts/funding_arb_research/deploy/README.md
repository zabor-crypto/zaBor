# funding-arb VPS deployment

Lives at `/opt/funding_arb` on the VPS. Runs as the `funding-arb` system user under systemd.
Logs go to `/var/log/funding-arb/recorder.log` and `journalctl -u funding-arb-recorder`.

## First-time install

From your dev machine:

```bash
# 1. push code
VPS=user@your.vps.host ./deploy/sync.sh --install
```

That single command rsyncs the tree, creates the venv, installs deps, registers the
systemd unit, and starts the recorder. After it's done:

```bash
ssh $VPS 'sudo systemctl status funding-arb-recorder'
ssh $VPS 'tail -f /var/log/funding-arb/recorder.log'
```

## Subsequent updates

```bash
VPS=user@your.vps.host ./deploy/sync.sh --restart
```

This rsyncs only the changed files and restarts the unit. The recorder traps SIGINT
and flushes its buffers before exit, so you do not lose in-flight rows.

## Disk

The recorder lives within `--quota-gb 50` (set in the systemd unit's ExecStart).
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

## Removing

```bash
ssh $VPS 'sudo systemctl disable --now funding-arb-recorder'
ssh $VPS 'sudo rm /etc/systemd/system/funding-arb-recorder.service && sudo systemctl daemon-reload'
ssh $VPS 'sudo userdel funding-arb || true'
ssh $VPS 'sudo rm -rf /opt/funding_arb /var/log/funding-arb'
```
