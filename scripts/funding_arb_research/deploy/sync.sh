#!/usr/bin/env bash
# Preview the funding-arb deployment commands without contacting a server.
# --apply explicitly executes the displayed plan. No mode deletes remote files.

set -euo pipefail

sync_usage() {
  cat <<'USAGE'
Usage: VPS=user@host deploy/sync.sh [--preview | --apply] [--install | --restart]

No flag (or --preview): print commands only; no SSH, rsync or remote writes.
--apply: copy code into /opt/funding_arb, preserving excluded and extra remote files.
--install: additionally run the installer (packages, user, unit, recorder start).
--restart: additionally restart the existing recorder unit and show its status.
--install/--restart without --apply only preview their respective command plans.

Applying a plan requires the owner's authorization for that target and effect.
Remote deletion is not supported; inventory and authorize it separately.
USAGE
}

# These functions can be sourced for pure parsing/planning tests; sourcing runs no plan.
sync_parse_options() {
  SYNC_MODE=preview
  SYNC_MODE_SET=
  SYNC_ACTION=sync
  SYNC_HELP=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preview|--apply)
        if [[ -n "$SYNC_MODE_SET" && "$SYNC_MODE_SET" != "$1" ]]; then
          echo "--preview and --apply are mutually exclusive" >&2
          return 2
        fi
        SYNC_MODE_SET="$1"
        SYNC_MODE="${1#--}"
        ;;
      --install|--restart)
        if [[ "$SYNC_ACTION" != sync ]]; then
          echo "choose only one of --install or --restart" >&2
          return 2
        fi
        SYNC_ACTION="${1#--}"
        ;;
      --help|-h) SYNC_HELP=true ;;
      *) echo "unknown flag: $1" >&2; return 2 ;;
    esac
    shift
  done
}

sync_build_plan() {
  local target="$1"
  # A simple SSH hostname or configured alias, optionally preceded by user@.
  # Reject shell metacharacters, option-like hosts and rsync path syntax.
  if [[ ! "$target" =~ ^([a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "set VPS to a simple SSH alias or user@hostname" >&2
    return 2
  fi
  local app_dir=/opt/funding_arb
  local source_dir
  source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  SYNC_PREPARE=(ssh -- "$target" "mkdir -p $app_dir")
  SYNC_TRANSFER=(
    rsync -avz
    --exclude '.git' --exclude '.venv*' --exclude '__pycache__/'
    --exclude '.claude/' --exclude '.codex/' --exclude 'CLAUDE.md' --exclude 'AGENTS.md'
    --include '.env.example' --exclude '.env' --exclude '.env.*' --exclude '*.env'
    --exclude 'data/' --exclude 'outputs/' --exclude 'reports/' --exclude 'logs/'
    --exclude '*.log' --exclude '*.sqlite*' --exclude '*.db' --exclude '*.pyc'
    -- "$source_dir/" "$target:$app_dir/"
  )
  SYNC_FOLLOWUP=()
  if [[ "$SYNC_ACTION" == install ]]; then
    SYNC_FOLLOWUP=(ssh -- "$target" "sudo bash $app_dir/deploy/install_vps.sh")
  elif [[ "$SYNC_ACTION" == restart ]]; then
    SYNC_FOLLOWUP=(ssh -- "$target" "sudo systemctl restart funding-arb-recorder && sudo systemctl --no-pager --full status funding-arb-recorder")
  fi
}

sync_print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

sync_main() {
  # Parse every argument before computing or executing any deployment action.
  sync_parse_options "$@" || return $?
  if [[ "$SYNC_HELP" == true ]]; then
    sync_usage
    return 0
  fi
  sync_build_plan "${VPS:-}" || return $?
  echo "[$SYNC_MODE] funding-arb deployment command plan"
  sync_print_command "${SYNC_PREPARE[@]}"
  sync_print_command "${SYNC_TRANSFER[@]}"
  if [[ "$SYNC_ACTION" != sync ]]; then
    sync_print_command "${SYNC_FOLLOWUP[@]}"
  fi
  if [[ "$SYNC_MODE" != apply ]]; then
    echo "[preview] No commands executed. Recheck target, source and authorization before --apply."
    return 0
  fi
  "${SYNC_PREPARE[@]}"
  "${SYNC_TRANSFER[@]}"
  if [[ "$SYNC_ACTION" != sync ]]; then
    "${SYNC_FOLLOWUP[@]}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  sync_main "$@"
fi
