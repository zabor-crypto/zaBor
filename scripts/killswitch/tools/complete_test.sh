#!/bin/bash
# Comprehensive Live Testing - Spot + Futures, Read-Only + Demo

set -e

echo "============================================================"
echo "COMPREHENSIVE LIVE TESTING"
echo "============================================================"
echo ""

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: missing required environment variable: $name"
    return 1
  fi
}

# ============================
# PHASE 1: Spot Testing (Read-Only)
# ============================
echo "PHASE 1: Spot Testing (Read-Only)"
echo "----------------------------------------"

require_env BINANCE_API_KEY
require_env BINANCE_API_SECRET
require_env BYBIT_API_KEY
require_env BYBIT_API_SECRET
require_env BITGET_API_KEY
require_env BITGET_API_SECRET
require_env BITGET_PASSWORD

echo "Credentials loaded from environment"
python3 tools/dryrun_smoke.py --config config.yaml --minutes 10 \
    --db ./reports/smoke_all_readonly.sqlite \
    --output ./reports/all_readonly_$(date +%Y%m%d_%H%M%S).json \
    2>&1 | tee reports/all_readonly_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "PHASE 1 Complete"
echo ""

# ============================
# PHASE 2: Demo Accounts (Dry-Run)
# ============================
echo "PHASE 2: Demo Account Testing"
echo "----------------------------------------"

echo "Reusing currently exported credentials."
echo "For demo-only testing, export demo keys before running this script."

python3 tools/dryrun_smoke.py --config config.yaml --minutes 5 \
    --db ./reports/smoke_demo_dryrun.sqlite \
    --output ./reports/demo_dryrun_$(date +%Y%m%d_%H%M%S).json \
    2>&1 | tee reports/demo_dryrun_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "PHASE 2 Complete"
echo ""

echo "============================================================"
echo "ALL TESTING COMPLETE"
echo "============================================================"
