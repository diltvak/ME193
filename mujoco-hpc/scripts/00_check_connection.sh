#!/usr/bin/env bash
# Step 1 (run on your LAPTOP): can we reach the cluster?
# Off campus you must be on the Tufts VPN first.
source "$(dirname "$0")/../config.sh"

echo "== 1. DNS: can we find $HPC_HOST? =="
if host "$HPC_HOST" >/dev/null 2>&1; then echo "ok"; else
  echo "FAIL: cannot resolve. Are you on the Tufts VPN (or campus wifi)?"; exit 1; fi

echo; echo "== 2. Port 22 open? =="
if nc -z -G 5 "$HPC_HOST" 22 2>/dev/null; then echo "ok"; else
  echo "FAIL: port 22 blocked. Connect the Tufts VPN."; exit 1; fi

echo; echo "== 3. SSH login (you may be asked for password / Duo) =="
# ControlMaster keeps one authenticated connection open for 10 min,
# so later scripts do not ask for Duo again.
mkdir -p ~/.ssh/sockets
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/%r@%h \
    -o ControlPersist=10m "$HPC_USER@$HPC_HOST" 'echo "logged in as $(whoami) on $(hostname)"' \
  && echo "ok: connected" || { echo "FAIL: ssh login"; exit 1; }

echo; echo "Next: ./scripts/01_push_and_check.sh"
