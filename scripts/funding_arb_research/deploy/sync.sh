#!/usr/bin/env bash
# Push the funding_arb tree from this dev machine to the VPS.
#
# Usage:
#   VPS=user@vps.example.com deploy/sync.sh           # rsync only
#   VPS=user@vps.example.com deploy/sync.sh --install  # rsync, then run install_vps.sh
#   VPS=user@vps.example.com deploy/sync.sh --restart  # rsync, then restart service
#
# Excludes recorded data and the local venv so the VPS stays clean.

set -euo pipefail

if [[ -z "${VPS:-}" ]]; then
  echo "set VPS=user@host first (e.g. VPS=root@1.2.3.4 deploy/sync.sh)" >&2
  exit 1
fi

APP_DIR=/opt/funding_arb
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "[sync] rsync $HERE/ -> $VPS:$APP_DIR/"
ssh "$VPS" "mkdir -p $APP_DIR && chown -R \$(id -un):\$(id -gn) $APP_DIR" || true
rsync -avz --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'data/recorder*' \
  --exclude 'data/*.log' \
  --exclude 'outputs/runs' \
  --exclude '*.pyc' \
  "$HERE/" "$VPS:$APP_DIR/"

case "${1:-}" in
  --install)
    echo "[sync] running install_vps.sh on $VPS"
    ssh "$VPS" "sudo bash $APP_DIR/deploy/install_vps.sh"
    ;;
  --restart)
    echo "[sync] restoring ownership + restarting funding-arb-recorder on $VPS"
    # rsync runs as the SSH user (root) and clobbers ownership; chown back to
    # the system user so the systemd unit can write its data + log dirs.
    ssh "$VPS" "sudo chown -R funding-arb:funding-arb $APP_DIR /var/log/funding-arb 2>/dev/null || true && sudo systemctl restart funding-arb-recorder && sleep 2 && sudo systemctl --no-pager status funding-arb-recorder | head -20"
    ;;
  '')
    echo "[sync] done. Pass --install (first time) or --restart (subsequent) to apply."
    ;;
  *)
    echo "unknown flag: $1" >&2; exit 1
    ;;
esac
