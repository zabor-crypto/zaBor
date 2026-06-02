#!/usr/bin/env bash
# One-shot installer for the funding-arb recorder on a fresh Debian/Ubuntu VPS.
# Idempotent: safe to re-run after code updates.
#
# Run as root (or via sudo):
#   sudo bash deploy/install_vps.sh
#
# What it does:
#   1. Creates the funding-arb system user + log dir.
#   2. Sets up /opt/funding_arb with a Python 3.11 virtualenv.
#   3. Installs requirements from the project's requirements.txt.
#   4. Installs and enables the systemd service.
#
# Code is expected to already live at /opt/funding_arb (rsync it in first
# via deploy/sync.sh from your dev box).

set -euo pipefail

APP_DIR=/opt/funding_arb
APP_USER=funding-arb
LOG_DIR=/var/log/funding-arb
SERVICE_FILE=/etc/systemd/system/funding-arb-recorder.service
SOURCE_SERVICE="${APP_DIR}/deploy/funding-arb-recorder.service"

if [[ "$EUID" -ne 0 ]]; then
  echo "must run as root" >&2
  exit 1
fi

if [[ ! -d "$APP_DIR" ]]; then
  echo "expected source tree at $APP_DIR — rsync it from your dev box first" >&2
  exit 1
fi

echo "[1/5] system user + log dir"
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
mkdir -p "$LOG_DIR"
chown -R "$APP_USER:$APP_USER" "$LOG_DIR" "$APP_DIR/data" 2>/dev/null || true
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "[2/5] python + venv"
apt-get update -y >/dev/null
apt-get install -y python3 python3-venv python3-pip rsync >/dev/null
sudo -u "$APP_USER" -H bash -lc "
  set -e
  cd $APP_DIR
  if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
  . .venv/bin/activate
  pip install --upgrade pip wheel
  pip install -r requirements.txt
"

echo "[3/5] install systemd unit"
install -m 0644 "$SOURCE_SERVICE" "$SERVICE_FILE"
systemctl daemon-reload

echo "[4/5] enable + start"
systemctl enable funding-arb-recorder.service
systemctl restart funding-arb-recorder.service

echo "[5/5] status (sleep 5s for first heartbeat)"
sleep 5
systemctl --no-pager --full status funding-arb-recorder.service || true
echo
echo "Tail logs with: journalctl -u funding-arb-recorder -f"
echo "Or: tail -f $LOG_DIR/recorder.log"
